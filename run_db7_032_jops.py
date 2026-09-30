"""Submit DB7-032 once, or resume monitoring, then verify Kaggle outputs.

The runner never retries a kernel submission: an ambiguous push is recorded and
requires monitoring the existing kernel. Status/output reads can be retried.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "db7_032_action_results"


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def call(arguments: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Run the authenticated Kaggle CLI without exposing secrets in logs."""
    result = subprocess.run(
        ["kaggle", *arguments], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout,
    )
    message = result.stdout
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    print(message, flush=True)
    return result.returncode, message


def read_with_retry(arguments: list[str]) -> tuple[int, str]:
    """Retry only read operations; never submit a duplicate kernel."""
    assert arguments[:2] in (["kernels", "status"], ["kernels", "output"])
    for attempt in range(4):
        code, message = call(arguments)
        if code == 0:
            return code, message
        if attempt < 3:
            time.sleep(15 * (attempt + 1))
    return code, message


def verify_source_package(manifest: dict) -> Path:
    notebook = ROOT / manifest["notebook"]
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == manifest["notebook_sha256"]
    for relative, expected in manifest["source_files_sha256"].items():
        source = (ROOT / relative).resolve()
        assert source.is_relative_to(ROOT), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative
    return notebook


def verify_result_archive(path: Path, manifest: dict) -> dict:
    """The actual archive, not the CLI exit code, determines download success."""
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, 'Output ZIP failed CRC verification.'
        done = json.loads(archive.read('completion.json'))
        assert done['success'] and done['experiment_id'] == 'DB7-032'
        assert done['subjects'] == list(range(1, 21)) and done['seeds'] == [42, 43, 44]
        assert done['final_brb_fits'] == 60 and done['validation_brb_fits'] == 1200
        assert done['structure_candidates'] == 300
        assert done['test_window_seed_evaluations'] == 695163
        assert done['devices_used'] == ['cuda:0', 'cuda:1']
        assert done['all_parameter_groups_searched'] and not done['test_used_for_selection']
        assert isinstance(done['all_parameter_groups_updated'], bool)
        sources = json.loads(archive.read('source_hashes.json'))
        for name in ('db7_030_hudgins.py', 'db7_031_brb.py', 'db7_032_jops_engine.py', 'db7_032_jops.py'):
            assert sources[name] == manifest['source_files_sha256']['scripts/' + name], name
        for name in ('summary.json', 'REPORT.md', 'subject_seed_metrics.csv',
                     'subject_gesture_metrics.csv', 'subject_accuracy.png', 'gesture_recall.png'):
            assert archive.getinfo(name).file_size > 0, name
        return done


def download_results(reference: str, manifest: dict) -> Path:
    """Accept a verified ZIP even if optional Kaggle API work returns HTTP429."""
    destination = OUT / 'kaggle'
    destination.mkdir(exist_ok=True)
    for attempt in range(4):
        code, message = call(['kernels', 'output', reference, '-p', str(destination),
                              '--file-pattern', r'^db7_032_results\.zip$', '--force'])
        archives = list(destination.rglob('db7_032_results.zip'))
        if len(archives) == 1:
            try:
                verify_result_archive(archives[0], manifest)
            except (AssertionError, KeyError, ValueError, OSError, zipfile.BadZipFile) as exc:
                print('Result archive not yet valid:', type(exc).__name__, str(exc), flush=True)
            else:
                if code:
                    print('Kaggle CLI reported an error, but the complete archive passed CRC, '
                          'source identity and experiment completion checks.', flush=True)
                save_json(OUT / 'download_verification.json', {
                    'archive_verified': True, 'cli_exit_code': code, 'attempt': attempt + 1,
                    'rate_limit_reported': '429' in message, 'kernel': reference})
                return archives[0]
        if attempt < 3:
            time.sleep(60)
    raise RuntimeError('No verified complete result archive. Resume monitoring this kernel; do not retrain.')


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "db7-032-manifest.json").read_text(encoding="utf-8"))
    notebook = verify_source_package(manifest)
    assert os.environ.get("KAGGLE_USERNAME") == "beautifulminnd"

    reference = os.environ.get("KAGGLE_MONITOR_REF", "").strip()
    if reference:
        assert re.fullmatch(r"beautifulminnd/db7-032-jops-[0-9]+", reference)
    else:
        assert os.environ.get("GITHUB_RUN_ATTEMPT") == "1", (
            "Use monitor_ref for an existing kernel; do not submit a duplicate."
        )
        reference = "beautifulminnd/db7-032-jops-" + os.environ["GITHUB_RUN_ID"]
        stage = ROOT / "db7_032_submission"
        stage.mkdir(exist_ok=True)
        (stage / "experiment.ipynb").write_bytes(notebook.read_bytes())
        save_json(stage / "kernel-metadata.json", {
            "id": reference, "title": reference.split("/")[-1],
            "code_file": "experiment.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [manifest["raw_dataset"]],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        })
        save_json(OUT / "launch.json", {
            "state": "push_attempted", "kernel": reference, "manifest": manifest,
        })
        code, message = call([
            "kernels", "push", "-p", str(stage),
            "--accelerator", "NvidiaTeslaT4", "--timeout", "43200",
        ])
        if code or "successfully pushed" not in message.lower() or "not valid dataset sources" in message.lower():
            raise RuntimeError("Push failed or ambiguous; inspect launch.json and the kernel before resubmission.")

    launch = {"state": "submitted_or_monitoring", "kernel": reference, "manifest": manifest}
    save_json(OUT / "launch.json", launch)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(f"DB7-032: https://www.kaggle.com/code/{reference}\n")

    deadline = time.monotonic() + 18000
    while True:
        code, message = read_with_retry(["kernels", "status", reference])
        if code:
            raise RuntimeError("Status failed after retries. Resume with monitor_ref; do not resubmit.")
        if re.search(r"\bcomplete\b", message, re.I):
            break
        if re.search(r"\b(error|failed|cancelled)\b", message, re.I):
            launch["state"] = "kaggle_failed"
            save_json(OUT / "launch.json", launch)
            read_with_retry([
                "kernels", "output", reference, "-p", str(OUT / "kaggle"),
                "--file-pattern", r".*(db7_032_results\.zip|\.log|\.json|\.txt)$",
            ])
            raise RuntimeError("Kaggle experiment failed; inspect downloaded logs.")
        if time.monotonic() > deadline:
            launch["state"] = "monitor_timeout_kernel_may_still_run"
            save_json(OUT / "launch.json", launch)
            raise TimeoutError("Kaggle may still be running. Use monitor_ref instead of submitting again.")
        time.sleep(60)

    result_archive = download_results(reference, manifest)
    with zipfile.ZipFile(result_archive) as archive:
        assert archive.testzip() is None, "Output ZIP failed CRC verification."
        done = json.loads(archive.read("completion.json"))
        assert done["success"]
        assert done["subjects"] == list(range(1, 21)) and done["seeds"] == [42, 43, 44]
        assert done["final_brb_fits"] == 60 and done["validation_brb_fits"] == 1200
        assert done["structure_candidates"] == 300
        assert done["test_window_seed_evaluations"] == 695163
        assert done["devices_used"] == ["cuda:0", "cuda:1"]
        assert done["all_parameter_groups_searched"] and not done["test_used_for_selection"]
        # A valid DE search may stagnate. Report observed changes, never invent
        # successful optimization by requiring every parameter to have moved.
        assert isinstance(done["all_parameter_groups_updated"], bool)
        summary = OUT / "summary"
        summary.mkdir(exist_ok=True)
        for name in (
            "completion.json", "summary.json", "REPORT.md",
            "subject_seed_metrics.csv", "subject_gesture_metrics.csv",
            "subject_accuracy.png", "gesture_recall.png",
        ):
            (summary / name).write_bytes(archive.read(name))
    save_json(OUT / "verification.json", {
        "completion": done, "kernel": reference,
        "result_zip_sha256": hashlib.sha256(result_archive.read_bytes()).hexdigest(),
        "notebook_sha256": manifest["notebook_sha256"],
    })
    launch["state"] = "verified_complete"
    save_json(OUT / "launch.json", launch)
    print("DB7-032 VERIFIED COMPLETE", reference, flush=True)


if __name__ == "__main__":
    main()
