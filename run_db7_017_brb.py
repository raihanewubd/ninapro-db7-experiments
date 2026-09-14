"""Submit one DB7-017 pilot, or monitor an existing Kaggle run without a push.

GitHub Actions supplies authentication. This standard-library runner validates
the packaged notebook, checks access/quota, and records a push attempt before
contacting Kaggle. An ambiguous push is never retried automatically.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import urllib.request
import zipfile

EXPERIMENT = "DB7-017"
SUBJECTS = [1, 15]
SEEDS = [42]
EPOCHS = 13
EXPECTED_FITS = 34
DATASET = "rayaanraza1/ninapro-db7"
ACCOUNT = "beautifulminnd"
NOTEBOOK = "db7-017-brb-pilot.ipynb"
MANIFEST = "db7-017-manifest.json"
POLL_SECONDS = 90
MONITOR_SECONDS = 18_000
# Conservative allowance for the complete capped session on two devices. This
# is a submission guard, not a claim about Kaggle's accounting conversion.
MINIMUM_QUOTA_HOURS = 10.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(text):
    text = str(text)
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "GH_TOKEN", "GITHUB_TOKEN"):
        secret = os.environ.get(name, "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    # Output URLs can contain temporary signed download credentials.
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", text)


def command(arguments, log, timeout=180):
    """Run only the official Kaggle CLI, retaining sanitized diagnostics."""
    try:
        result = subprocess.run(
            ["kaggle", *arguments], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False,
        )
        code, output = result.returncode, result.stdout or ""
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        code, output = 124, output + "\nCommand timed out.\n"
    output = sanitize(output)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] kaggle {' '.join(arguments)}\n{output}\n")
    print(output, flush=True)
    return code, output


def duration_seconds(value):
    """Kaggle uses protobuf JSON durations, e.g. '3600.25s'."""
    require(isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d{1,9})?s", value),
            "Missing or unrecognized GPU quota duration; refusing submission")
    seconds = float(value[:-1])
    require(math.isfinite(seconds), "Nonfinite GPU quota duration")
    return seconds


def quota_summary(response, minimum_hours=MINIMUM_QUOTA_HOURS):
    quota = response.get("gpuQuota")
    require(isinstance(quota, dict), "GPU quota missing; refusing submission")
    total = duration_seconds(quota.get("totalTimeAllowed"))
    # Protobuf omits a zero-valued duration in some responses.
    used = duration_seconds(quota.get("timeUsed", "0s"))
    require(used <= total, "GPU quota exhausted or inconsistent")
    remaining = (total - used) / 3600
    require(remaining >= minimum_hours,
            f"GPU quota {remaining:.2f} h is below the conservative {minimum_hours:.2f} h launch allowance")
    return dict(remaining_hours=remaining, used_hours=used / 3600,
                total_hours=total / 3600, required_hours=minimum_hours,
                accounting_note="Conservative two-device allowance; actual usage is determined by Kaggle.")


def check_access_and_quota(output):
    code, _ = command(["datasets", "files", DATASET], output / "dataset_access.log")
    require(code == 0, "Authenticated DB7 dataset read failed; nothing was submitted")
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    require(bool(token), "KAGGLE_API_TOKEN is required")
    request = urllib.request.Request(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics",
        data=b"{}", headers={"Authorization": "Bearer " + token,
                              "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
    except Exception as error:
        # Never serialize the request, token, headers, or exception response body.
        raise RuntimeError(f"Authenticated quota read failed ({type(error).__name__}); nothing was submitted") from None
    summary = quota_summary(payload)
    summary.update(checked_utc=utc_now(), dataset=DATASET, dataset_access=True)
    write_json(output / "access_and_quota.json", summary)
    print(f"GPU quota available: {summary['remaining_hours']:.2f} hours", flush=True)


def valid_ref(reference):
    require(re.fullmatch(rf"{ACCOUNT}/db7-017-brb-pilot-[0-9]+-[0-9]+", reference) is not None,
            "Expected the recorded beautifulminnd/db7-017-brb-pilot-RUNID-ATTEMPT reference")
    return reference


def parse_push_url(output):
    """Use Kaggle's actual returned reference instead of assuming the slug."""
    matches = re.findall(r"https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", output)
    references = sorted(set(matches))
    require(len(references) == 1, "Push outcome ambiguous: exactly one returned Kaggle code URL is required; do not push again")
    require(references[0].split("/")[0] == ACCOUNT, "Push returned an unexpected owner; inspect it before any new submission")
    return references[0], "https://www.kaggle.com/code/" + references[0]


def prepare(root, output):
    require(os.environ.get("KAGGLE_USERNAME", "").strip() == ACCOUNT,
            "This pilot is authorized for the beautifulminnd secondary account")
    notebook = root / NOTEBOOK
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    digest = sha256(notebook)
    require(manifest.get("notebook_sha256") == digest, "Notebook SHA-256 differs from the packaged manifest")
    require(manifest.get("experiment_id") == EXPERIMENT, "Manifest experiment ID differs")
    require(manifest.get("subjects") == SUBJECTS and manifest.get("seeds") == SEEDS,
            "Only the authorized S1/S15, seed 42 pilot may be submitted")
    require(manifest.get("neural_fits") == EXPECTED_FITS and manifest.get("epochs") == EPOCHS,
            "Manifest must declare 34 neural fits and 13 full epochs")
    run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    require(run_id.isdigit() and attempt.isdigit(), "GitHub run ID and attempt are required")
    require(attempt == "1", "Re-running a GitHub job must use monitor_ref; a fresh automatic push could duplicate training")
    slug = f"db7-017-brb-pilot-{run_id}-{attempt}"
    reference = valid_ref(f"{ACCOUNT}/{slug}")
    submission = root / "brb_submitted"
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(notebook, submission / "experiment.ipynb")
    shutil.copyfile(root / MANIFEST, submission / MANIFEST)
    write_json(submission / "kernel-metadata.json", dict(
        id=reference, title=slug, code_file="experiment.ipynb", language="python",
        kernel_type="notebook", is_private=True, enable_gpu=True,
        enable_internet=False, machine_shape="NvidiaTeslaT4",
        dataset_sources=[DATASET], competition_sources=[], kernel_sources=[], model_sources=[],
    ))
    provenance = dict(experiment_id=EXPERIMENT, notebook_sha256=digest,
                      package_manifest_sha256=sha256(root / MANIFEST),
                      github_run_id=run_id, github_run_attempt=attempt,
                      github_commit=os.environ.get("GITHUB_SHA"),
                      github_repository=os.environ.get("GITHUB_REPOSITORY"),
                      subjects=SUBJECTS, seeds=SEEDS, neural_fits=EXPECTED_FITS,
                      epochs=EPOCHS, expected_ref=reference, created_utc=utc_now())
    write_json(output / "source_provenance.json", provenance)
    return submission, reference, provenance


def safe_zip_members(archive):
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate ZIP names")
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        require(info.filename and "\\" not in info.filename and not path.is_absolute()
                and ".." not in path.parts and not any(":" in p for p in path.parts),
                "Unsafe ZIP member path")
        require(not stat.S_ISLNK(info.external_attr >> 16), "Symlink ZIP member")
    return names


def validate_archive(path, expected_manifest=None):
    """Require real full-training evidence; smoke manifests never count."""
    with zipfile.ZipFile(path) as archive:
        names = safe_zip_members(archive)
        require(archive.testzip() is None, "Result ZIP CRC check failed")
        def read_json(name):
            require(name in names, f"Missing result file: {name}")
            require(archive.getinfo(name).file_size <= 2_000_000, "Oversized manifest")
            return json.loads(archive.read(name))
        completion = read_json("completion.json")
        require(completion.get("experiment_id") == EXPERIMENT and completion.get("success") is True,
                "Notebook did not certify successful DB7-017 completion")
        require(completion.get("subjects") == SUBJECTS and completion.get("seeds") == SEEDS,
                "Completion scope differs from the authorized pilot")
        require(completion.get("neural_fits") == EXPECTED_FITS and completion.get("meta_completed") == 2,
                "Expected 34 full fits and two completed subject meta-analyses")
        protocol = read_json("run_manifest.json")
        require(protocol.get("experiment_id") == EXPERIMENT and protocol.get("subjects") == SUBJECTS
                and protocol.get("seeds") == SEEDS and protocol.get("epochs") == EPOCHS
                and protocol.get("smoke") is False,
                "Runtime manifest does not describe the full authorized pilot")
        require(protocol.get("window_samples") == 400 and protocol.get("stride_samples") == 20
                and protocol.get("fs") == 2000
                and protocol.get("train_repetitions") == [1, 3, 4, 6]
                and protocol.get("test_repetitions") == [2, 5],
                "Runtime input window, stride or repetition split differs")
        sources = read_json("source_sha256.json")
        require(isinstance(sources, dict) and sources and protocol.get("source_sha256") == sources,
                "Runtime source manifest is missing or inconsistent")
        for filename, digest in sources.items():
            require(re.fullmatch(r"[A-Za-z0-9_]+\.py", filename) and isinstance(digest, str)
                    and re.fullmatch(r"[a-f0-9]{64}", digest), "Invalid source hash entry")
            source_name = "source/" + filename
            require(source_name in names and hashlib.sha256(archive.read(source_name)).hexdigest() == digest,
                    f"Archived source checksum differs: {filename}")
        if expected_manifest is not None:
            require(sources == expected_manifest.get("source_sha256"), "Result source differs from the submitted package")
            require(protocol.get("protocol_sha256") == expected_manifest.get("protocol_sha256")
                    and isinstance(protocol.get("protocol_sha256"), str),
                    "Result protocol hash differs from the submitted package")
        preflight = read_json("preflight.json")
        require(preflight.get("success") is True and preflight.get("gpu_count") == 2
                and preflight.get("smoke_success") is True,
                "Both-GPU preflight and end-to-end smoke evidence are required")
        full_names = [name for name in names if name.startswith("full/") and name.endswith("/fit_manifest.json")]
        require(len(full_names) == EXPECTED_FITS, f"Expected 34 full fit manifests; found {len(full_names)}")
        identities = set()
        for name in full_names:
            fit = read_json(name)
            require(fit.get("subject") in SUBJECTS and fit.get("seed") == 42,
                    f"Unexpected fit scope: {name}")
            require(fit.get("status") == "complete" and fit.get("epochs_run") == EPOCHS,
                    f"Fit did not complete all 13 epochs: {name}")
            identity = (fit["subject"], fit.get("stage"), fit.get("held_out_repetition"), fit.get("arm"))
            require(identity not in identities, f"Duplicate fit identity: {identity}")
            identities.add(identity)
            history_name = name.removesuffix("fit_manifest.json") + "history.csv"
            require(history_name in names, f"Missing full training history: {history_name}")
            history = list(csv.DictReader(io.StringIO(archive.read(history_name).decode("utf-8"))))
            require(len(history) == EPOCHS and {int(row["epoch"]) for row in history} == set(range(1, EPOCHS + 1)),
                    f"History does not contain exactly epochs 1–13: {history_name}")
        expected = {(s, "oof", rep, arm) for s in SUBJECTS for rep in (1, 3, 4, 6) for arm in ("W", "S", "I")}
        expected |= {(s, "final", None, arm) for s in SUBJECTS for arm in ("W", "S", "I", "SI", "WSI")}
        require(identities == expected, "Full fit identities do not match the 24 OOF + 10 final pilot design")
        meta = [read_json(name) for name in names if name.startswith("full/") and name.endswith("/meta_completion.json")]
        require(len(meta) == 2 and {item.get("subject") for item in meta} == set(SUBJECTS)
                and all(item.get("success") is True and item.get("seed") == 42 for item in meta),
                "Missing or failed final meta-learning reports")
        smoke = [name for name in names if name.startswith("smoke/") and name.endswith("/fit_manifest.json")]
    return dict(success=True, experiment_id=EXPERIMENT, archive=Path(path).name,
                archive_sha256=sha256(path), neural_fits=EXPECTED_FITS,
                meta_completed=2, subjects=SUBJECTS, seeds=SEEDS,
                source_sha256=sources, submitted_package_matched=expected_manifest is not None,
                separately_recorded_smoke_manifests=len(smoke), verified_utc=utc_now())


def collect(reference, output):
    # Output retries are read-only and never restart the remote notebook.
    for attempt in range(3):
        code, _ = command(["kernels", "output", reference, "-p", str(output / "kaggle_output"),
                           "--file-pattern", r".*\.(zip|json|log|txt)$", "--force"],
                          output / "download.log", timeout=900)
        if code == 0:
            return
        if attempt < 2:
            time.sleep(15)


def monitor(reference, output, state):
    deadline = time.monotonic() + MONITOR_SECONDS
    failures = 0
    while True:
        code, status = command(["kernels", "status", reference], output / "status.log")
        (output / "last_status.txt").write_text(status, encoding="utf-8")
        failures = failures + 1 if code else 0
        require(failures < 5, "Five status failures; run may still be active. Use monitor_ref, not a new push")
        if code == 0 and re.search(r"\bcomplete\b", status, re.I):
            state.update(state="kaggle_complete", completed_utc=utc_now())
            write_json(output / "launch_state.json", state)
            return
        if code == 0 and re.search(r"\b(error|failed|cancelled|canceled)\b", status, re.I):
            raise RuntimeError("Kaggle reported failure; collecting available outputs and execution logs")
        if time.monotonic() >= deadline:
            raise TimeoutError("Monitoring deadline reached. The saved Kaggle run may still be active; resume with monitor_ref")
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))


def run(root, monitor_ref="", prepare_only=False):
    root = Path(root).resolve()
    output = root / "brb_results"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "launch_state.json"
    if monitor_ref:
        reference = valid_ref(monitor_ref.strip())
        state = dict(experiment_id=EXPERIMENT, state="monitoring_existing", actual_ref=reference,
                     url="https://www.kaggle.com/code/" + reference, created_utc=utc_now(), new_submission=False)
    else:
        require(not state_path.exists(), "This folder already records a launch attempt; use its reference with monitor_ref")
        submission, reference, provenance = prepare(root, output)
        if prepare_only:
            print("Prepared and hash-checked the pilot; no network request or submission.", flush=True)
            return
        check_access_and_quota(output)
        state = dict(provenance, state="push_attempted", new_submission=True,
                     expected_url="https://www.kaggle.com/code/" + reference)
        write_json(state_path, state)
        code, push_output = command(["kernels", "push", "-p", str(submission),
                                     "--accelerator", "NvidiaTeslaT4", "--timeout", "18000"],
                                    output / "push.log", timeout=300)
        try:
            actual_ref, actual_url = parse_push_url(push_output)
            state.update(actual_ref=actual_ref, url=actual_url)
            write_json(state_path, state)
            require(code == 0, "Push failed or timed out; inspect the recorded URL before resuming with monitor_ref")
            require(actual_ref == reference, "Kaggle returned a different slug. Inspect the actual URL; no automatic push retry")
        except Exception:
            state.update(state="push_ambiguous", updated_utc=utc_now())
            write_json(state_path, state)
            # The expected reference may exist even if the response was lost.
            collect(state.get("actual_ref", reference), output)
            raise
        reference = actual_ref
        state["state"] = "submitted"
    write_json(state_path, state)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"DB7-017 pilot: {state['url']}\n\nS1/S15, seed 42; 34 full fits; both T4 GPUs required by notebook preflight.\n")
    try:
        monitor(reference, output, state)
    except Exception as error:
        state.update(state="monitoring_failed", error=sanitize(error), updated_utc=utc_now())
        write_json(state_path, state)
        raise
    finally:
        collect(reference, output)
    archives = sorted((output / "kaggle_output").rglob("db7_brb_*.zip"))
    require(len(archives) == 1, f"Expected one DB7 BRB result archive; found {len(archives)}")
    expected_manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8")) if state["new_submission"] else None
    try:
        result = validate_archive(archives[0], expected_manifest)
    except Exception as error:
        state.update(state="result_validation_failed", error=sanitize(error), updated_utc=utc_now())
        write_json(state_path, state)
        raise
    write_json(output / "verification.json", result)
    state.update(state="verified_complete", verification=result, updated_utc=utc_now())
    write_json(state_path, state)
    print("Verified DB7-017: 34 full 13-epoch fits and two completed meta-learning reports.", flush=True)
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("\nVerified: ZIP CRC, all 34 full histories, both GPUs, smoke preflight and two subject meta-completions.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--monitor-ref", default=os.environ.get("KAGGLE_MONITOR_REF", ""))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.root, args.monitor_ref, args.prepare_only)


if __name__ == "__main__":
    main()
