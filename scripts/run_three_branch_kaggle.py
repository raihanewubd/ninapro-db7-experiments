"""Submit, collect and verify the paired DB7 three-branch study.

The runner itself uses the Python standard library. GitHub Actions installs the
pinned Kaggle CLI; this script never installs anything locally. PREPARE_ONLY=1
writes a reviewable notebook and metadata without contacting Kaggle.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["B0", "C1"]
SEEDS = [42, 43, 44]
SPLITS = ["train", "validation", "test"]
REPOSITORY = "raihanewubd/ninapro-db7-experiments"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_settings(job):
    if job == "smoke":
        return 42, True, [1, 22]
    match = re.fullmatch(r"seed(42|43|44)", job)
    if not match:
        raise ValueError("THREE_BRANCH_JOB must be smoke, seed42, seed43 or seed44")
    return int(match.group(1)), False, list(range(1, 23))


def prepare(job):
    seed, smoke, _ = job_settings(job)
    user = os.environ["KAGGLE_USERNAME"]
    assert re.fullmatch(r"[A-Za-z0-9_-]+", user), "Invalid Kaggle username"
    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    assert run_id.isdigit() and attempt.isdigit()
    slug = f"db7-3b-{run_id}-{attempt}-{job}"
    reference = f"{user}/{slug}"
    source = ROOT / "notebooks" / "db7-three-branch.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    provenance = dict(
        kaggle_ref=reference,
        github_run_id=run_id,
        github_run_attempt=attempt,
        github_commit=os.environ.get("GITHUB_SHA", "local"),
        github_repository=os.environ.get("GITHUB_REPOSITORY", REPOSITORY),
        job=job,
        source_sha256=file_hash(source),
    )
    # Config.SEED remains unchanged: it controls the fixed repetition split.
    override = (
        f"Config.SEED_BASE = {seed}\n"
        f"Config.SMOKE = {smoke!r}\n"
        f"Config.AUTOMATION = {provenance!r}\n"
        "RESULTS_DIRECTORY = run_three_branch_study()\n"
    )
    code_indices = [i for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
    assert code_indices, "Notebook has no code cells"
    notebook["cells"][code_indices[-1]]["source"] = override.splitlines(True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell["outputs"] = []
            cell["execution_count"] = None
    folder = ROOT / "three_branch_submitted" / job
    write_json(folder / "experiment.ipynb", notebook)
    write_json(folder / "kernel-metadata.json", dict(
        id=reference, title=slug, code_file="experiment.ipynb", language="python",
        kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=False,
        machine_shape="NvidiaTeslaT4", dataset_sources=["rayaanraza1/ninapro-db7"],
        kernel_sources=[], competition_sources=[], model_sources=[],
    ))
    write_json(folder / "submission.json", provenance)
    print(f"Prepared {job}: source notebook SHA-256 {provenance['source_sha256']}", flush=True)
    return folder, reference, provenance


def command(arguments, log, timeout=300):
    result = subprocess.run(
        ["kaggle", *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )
    output = result.stdout
    for name in ["KAGGLE_API_TOKEN", "KAGGLE_KEY"]:
        secret = os.environ.get(name, "")
        if secret:
            output = output.replace(secret, "[REDACTED]")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(output + "\n")
    print(output, flush=True)
    return result.returncode, output


def safe_members(archive):
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate archive member names"
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        assert member.filename and "\\" not in member.filename
        assert not path.is_absolute() and ".." not in path.parts
        assert all(":" not in part for part in path.parts)
        assert not stat.S_ISLNK(member.external_attr >> 16), "Symlink member is not allowed"
        yield member, path


def read_csv(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def prediction_key_hash(rows):
    columns = ["subject", "gesture", "native_repetition", "window_start", "window_end"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([[row[column] for column in columns] for row in rows])
    return hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()


def validate_archive(path, job, reference=None, expected_provenance=None):
    seed, smoke, subjects = job_settings(job)
    expected_fits = len(subjects) * len(ARMS)
    with zipfile.ZipFile(path) as archive:
        list(safe_members(archive))
        assert archive.testzip() is None, "ZIP CRC verification failed"
        names = set(archive.namelist())
        assert not any(PurePosixPath(name).name in {"FAILURE.txt", "SUMMARY_FAILURE.txt"} for name in names)
        manifest = json.loads(archive.read("run_manifest.json"))
        completion = json.loads(archive.read("completion.json"))
        preflight = json.loads(archive.read("preflight.json"))
        assert preflight["success"] is True
        assert manifest["arms"] == ARMS and manifest["subjects"] == subjects
        assert manifest["seed_base"] == seed and manifest["smoke"] == smoke
        assert manifest["window_ms"] == 400 and manifest["stride_ms"] == 100
        assert completion["success"] is True and completion["cnn_fits"] == expected_fits
        assert completion["subjects"] == subjects and completion["arms"] == ARMS
        assert completion["seed_base"] == seed
        if reference is not None:
            assert manifest["automation"]["kaggle_ref"] == reference
        if expected_provenance:
            for key, value in expected_provenance.items():
                assert manifest["automation"][key] == value, f"Provenance mismatch: {key}"
        metrics = read_csv(archive, "all_subject_metrics.csv")
        assert len(metrics) == expected_fits * 3, "Incomplete root metric rows"
        expected_metric_keys = {(subject, arm, split) for subject in subjects for arm in ARMS for split in SPLITS}
        assert {(int(row["subject"]), row["arm"], row["split"]) for row in metrics} == expected_metric_keys
        hashes = {}
        for subject in subjects:
            for arm in ARMS:
                prefix = f"S{subject:02}/seed_{seed}/{arm}"
                fit = json.loads(archive.read(f"{prefix}/fit_manifest.json"))
                assert fit["subject"] == subject and fit["arm"] == arm and fit["seed_base"] == seed
                assert fit["status"] == "complete"
                assert fit["parameters"] == {"B0": 552966, "C1": 551542}[arm]
                assert fit["epochs_run"] == 2 if smoke else 20 <= fit["epochs_run"] <= 150
                assert 1 <= fit["selected_epoch"] <= fit["epochs_run"]
                identity = fit["window_key_hashes"]
                assert set(identity) == set(SPLITS)
                assert all(re.fullmatch(r"[a-f0-9]{64}", h) for h in identity.values())
                assert identity == hashes.setdefault(subject, identity), "B0 and C1 window identities differ"
                for filename in ["best_model.pt", "normalizer.npz", "history.csv", "metrics.csv"]:
                    assert f"{prefix}/{filename}" in names, f"Missing {prefix}/{filename}"
                fit_metrics = read_csv(archive, f"{prefix}/metrics.csv")
                assert len(fit_metrics) == 3 and {row["split"] for row in fit_metrics} == set(SPLITS)
                for split in SPLITS:
                    prediction_name = f"{prefix}/{split}/predictions.csv"
                    assert prediction_name in names
                    assert f"{prefix}/{split}/probabilities_embeddings.npz" in names
                    predictions = read_csv(archive, prediction_name)
                    assert predictions, f"Empty predictions: {prediction_name}"
                    assert {int(row["subject"]) for row in predictions} == {subject}
                    assert {int(row["gesture"]) for row in predictions} == set(range(1, 18))
                    assert prediction_key_hash(predictions) == identity[split], f"Prediction identity hash mismatch: {prediction_name}"
        return dict(
            success=True, job=job, subjects=subjects, arms=ARMS, seed_base=seed,
            smoke=smoke, cnn_fits=expected_fits, archive=path.name,
            archive_sha256=file_hash(path), automation=manifest["automation"],
            window_key_hashes=hashes,
        )


def validate(folder, job, reference, provenance):
    archives = list(folder.rglob("db7_three_branch_*.zip"))
    assert len(archives) == 1, f"Expected one result archive, found {len(archives)}"
    result = validate_archive(archives[0], job, reference, provenance)
    write_json(folder / "validation.json", result)
    print(f"Verified {job}: {result['cnn_fits']} fits, archive SHA-256 {result['archive_sha256']}", flush=True)
    return result


def run_job(job):
    folder, reference, provenance = prepare(job)
    if os.environ.get("PREPARE_ONLY") == "1":
        print("PREPARE_ONLY: no submission or network request was made.", flush=True)
        return
    assert os.environ.get("KAGGLE_API_TOKEN"), "KAGGLE_API_TOKEN must be supplied by GitHub Secrets"
    _, smoke, _ = job_settings(job)
    output = ROOT / "three_branch_results" / job
    output.mkdir(parents=True, exist_ok=True)
    state_file = output / "launch_state.json"
    assert not state_file.exists(), (
        "A submission attempt already exists in this folder. Inspect the saved Kaggle URL "
        "and status before retrying; this runner will not submit it twice."
    )
    url = "https://www.kaggle.com/code/" + reference
    write_json(output / "kernel.json", dict(ref=reference, url=url))
    write_json(output / "source_provenance.json", provenance)
    print("Kaggle URL: " + url, flush=True)
    launch_state = dict(ref=reference, url=url, state="push_attempted", created_utc=utc_now())
    write_json(state_file, launch_state)
    try:
        code, _ = command(["kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"], output / "push.log")
        assert code == 0, "Push failed or is ambiguous. Inspect the saved URL; do not blindly submit again."
        launch_state["state"] = "submitted"
        write_json(state_file, launch_state)
        deadline = time.monotonic() + (60 if smoke else 310) * 60
        failures = 0
        while True:
            code, status = command(["kernels", "status", reference], output / "status.log")
            failures = failures + 1 if code else 0
            assert failures < 5, "Repeated status failures; Kaggle may still be active at the saved URL"
            if code == 0 and re.search(r"\bCOMPLETE\b", status.upper()):
                launch_state["state"] = "complete"
                write_json(state_file, launch_state)
                break
            if code == 0 and re.search(r"\b(ERROR|FAILED|CANCELED|CANCELLED)\b", status.upper()):
                raise RuntimeError("Kaggle execution failed; inspect downloaded diagnostics")
            if time.monotonic() > deadline:
                raise TimeoutError("Kaggle may still be running. Check the saved URL before any resubmission.")
            time.sleep(60)
    finally:
        # Output retries never launch or restart training.
        for retry in range(3):
            code, _ = command(
                ["kernels", "output", reference, "-p", str(output), "--file-pattern", r".*\.(zip|log)$", "--force"],
                output / "download.log", timeout=600,
            )
            if code == 0 and list(output.glob("db7_three_branch_*.zip")):
                break
            if retry < 2:
                time.sleep(30)
    validate(output, job, reference, provenance)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def github_request(url, token):
    return urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "db7-three-branch-analysis",
    })


def download_artifact(artifact_id, destination, token):
    # Do not carry the GitHub authorization header to the signed blob URL.
    endpoint = f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"
    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(github_request(endpoint, token), timeout=60)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        location = error.headers["Location"]
        assert location.startswith("https://"), "Artifact redirect must use HTTPS"
        response = urllib.request.urlopen(location, timeout=120)
    temporary = destination.with_suffix(".partial")
    with response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    with zipfile.ZipFile(temporary) as archive:
        list(safe_members(archive))
        assert archive.testzip() is None
    temporary.replace(destination)


def safe_extract(archive, destination):
    resolved_root = destination.resolve()
    for member, path in safe_members(archive):
        target = destination.joinpath(*path.parts)
        assert target.resolve().is_relative_to(resolved_root), "Unsafe extraction destination"
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)


def collect_analysis():
    """Collect only three full seed artifacts, then run the paired analysis."""
    token = os.environ["GH_TOKEN"]
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    assert run_id.isdigit() and attempt.isdigit()
    assert os.environ.get("GITHUB_REPOSITORY", REPOSITORY) == REPOSITORY
    artifacts = []
    for page in range(1, 101):
        endpoint = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        with urllib.request.urlopen(github_request(endpoint, token), timeout=60) as response:
            batch = json.load(response)["artifacts"]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
    downloads = ROOT / "three_branch_analysis_downloads"
    collected = ROOT / "collectedruns"
    downloads.mkdir(parents=True, exist_ok=True)
    collected.mkdir(parents=True, exist_ok=True)
    verifications = []
    for seed in SEEDS:
        job = f"seed{seed}"
        name = f"db7-three-branch-{run_id}-{attempt}-{job}"
        matches = [a for a in artifacts if a["name"] == name and not a["expired"]]
        assert len(matches) == 1, f"Expected one unexpired artifact named {name}"
        outer_path = downloads / f"{job}.zip"
        download_artifact(matches[0]["id"], outer_path, token)
        with zipfile.ZipFile(outer_path) as outer:
            members = [m for m, _ in safe_members(outer) if PurePosixPath(m.filename).name.startswith("db7_three_branch_") and m.filename.endswith(".zip")]
            assert len(members) == 1, f"Expected one full study ZIP in {name}"
            inner_path = downloads / f"{job}-study.zip"
            with outer.open(members[0]) as source, inner_path.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
        checked = validate_archive(inner_path, job, expected_provenance={
            "github_run_id": run_id, "github_run_attempt": attempt,
            "github_commit": os.environ["GITHUB_SHA"], "job": job,
            "source_sha256": file_hash(ROOT / "notebooks" / "db7-three-branch.ipynb"),
        })
        with zipfile.ZipFile(inner_path) as inner:
            safe_extract(inner, collected / job)
        checked["github_artifact_id"] = matches[0]["id"]
        verifications.append(checked)
        print(f"Collected and verified {job}: 44 full fits", flush=True)
    hashes = [entry["window_key_hashes"] for entry in verifications]
    assert hashes[0] == hashes[1] == hashes[2], "Window identities differ between seeds"
    source_hashes = {entry["automation"]["source_sha256"] for entry in verifications}
    assert len(source_hashes) == 1, "Source notebooks differ between seeds"
    write_json(collected / "collection_verification.json", dict(
        success=True, cnn_fits=132, seeds=SEEDS, arms=ARMS,
        github_run_id=run_id, github_run_attempt=attempt, sources=verifications,
    ))
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "analyze_three_branch_results.py"),
        "--roots", str(collected), "--output", str(ROOT / "three_branch_analysis"),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-analysis", action="store_true")
    args = parser.parse_args()
    if args.collect_analysis:
        collect_analysis()
    else:
        run_job(os.environ.get("THREE_BRANCH_JOB", "smoke"))


if __name__ == "__main__":
    main()
