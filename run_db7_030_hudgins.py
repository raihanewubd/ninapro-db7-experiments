"""Create the compact trace dataset, submit DB7-030, monitor and verify results."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import zipfile

root = Path(__file__).resolve().parent
out = root / "db7_030_action_results"
out.mkdir(exist_ok=True)
manifest = json.loads((root / "db7-030-manifest.json").read_text())
trace = root / "data/db7-030-si-i-trace.csv.gz"
notebook = root / "db7-030-hudgins-td4.ipynb"
assert hashlib.sha256(trace.read_bytes()).hexdigest() == manifest["trace_sha256"]
assert hashlib.sha256(notebook.read_bytes()).hexdigest() == manifest["notebook_sha256"]
assert os.environ.get("KAGGLE_USERNAME") == "beautifulminnd"
requested = os.environ.get("KAGGLE_MONITOR_REF", "").strip()
existing_dataset = os.environ.get("KAGGLE_EXISTING_DATASET_REF", "").strip()


def call(args, timeout=900):
    process = subprocess.run(["kaggle", *args], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=timeout)
    output = process.stdout
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        if os.environ.get(key):
            output = output.replace(os.environ[key], "[REDACTED]")
    print(output, flush=True)
    return process.returncode, output


if requested:
    if not re.fullmatch(r"beautifulminnd/[a-z0-9-]+", requested):
        raise ValueError("Invalid existing Kaggle reference")
    ref = requested
    dataset_ref = "existing submission; read its notebook metadata"
else:
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise RuntimeError("Retry monitoring with KAGGLE_MONITOR_REF; do not create a duplicate notebook")
    run_id = os.environ["GITHUB_RUN_ID"]
    dataset_ref = existing_dataset or f"beautifulminnd/db7-030-si-i-trace-{run_id}"
    if not re.fullmatch(r"beautifulminnd/db7-030-si-i-trace-[0-9]+", dataset_ref):
        raise ValueError("Invalid existing trace dataset reference")
    ref = f"beautifulminnd/db7-030-hudgins-{run_id}"
    stage = root / "db7_030_submission"
    if not existing_dataset:
        dataset = stage / "trace_dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace, dataset / trace.name)
        (dataset / "dataset-metadata.json").write_text(json.dumps({
            "title": f"DB7 030 SI I frozen trace {run_id}",
            "id": dataset_ref, "licenses": [{"name": "other"}],
            "isPrivate": True}))
        (out / "launch.json").write_text(json.dumps({"status": "dataset_upload_attempted",
            "dataset": dataset_ref, "kernel": ref, "manifest": manifest}, indent=2))
        code, message = call(["datasets", "create", "-p", str(dataset), "-q"], timeout=1800)
        if code:
            raise RuntimeError("Dataset upload failed. Inspect launch.json before retrying.")
    # Kaggle dataset creation is asynchronous; attaching it before READY silently
    # drops the source while still reporting a successful kernel push.
    dataset_deadline = time.monotonic() + 1800
    while True:
        code, message = call(["datasets", "status", dataset_ref])
        if code == 0 and re.search(r"\bready\b", message, re.I):
            break
        if re.search(r"\b(failed|deleted)\b", message, re.I):
            raise RuntimeError("Trace dataset processing failed")
        if time.monotonic() > dataset_deadline:
            raise TimeoutError("Trace dataset is still processing; retry with KAGGLE_EXISTING_DATASET_REF")
        time.sleep(30)
    kernel = stage / "kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook, kernel / "experiment.ipynb")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": ref.split("/")[1], "code_file": "experiment.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [manifest["raw_dataset"], dataset_ref],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}))
    (out / "launch.json").write_text(json.dumps({"status": "kernel_push_attempted",
        "dataset": dataset_ref, "kernel": ref, "manifest": manifest}, indent=2))
    code, message = call(["kernels", "push", "-p", str(kernel)], timeout=1800)
    if code or "successfully pushed" not in message.lower() or "not valid dataset sources" in message.lower():
        raise RuntimeError("Ambiguous kernel push. Inspect launch.json before retrying.")

(out / "launch.json").write_text(json.dumps({"status": "submitted_or_monitoring",
    "dataset": dataset_ref, "kernel": ref, "manifest": manifest}, indent=2))
summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_file:
    with open(summary_file, "a", encoding="utf-8") as file:
        file.write(f"DB7-030 notebook: https://www.kaggle.com/code/{ref}\n")

deadline = time.monotonic() + 14400
while True:
    code, message = call(["kernels", "status", ref])
    if code:
        raise RuntimeError("Kaggle status call failed")
    if re.search(r"\bcomplete\b", message, re.I):
        break
    if re.search(r"\b(error|failed|cancelled)\b", message, re.I):
        call(["kernels", "output", ref, "-p", str(out / "kaggle")])
        raise RuntimeError("Kaggle notebook failed")
    if time.monotonic() > deadline:
        raise TimeoutError("Notebook still active; rerun workflow with KAGGLE_MONITOR_REF")
    time.sleep(60)

code, message = call(["kernels", "output", ref, "-p", str(out / "kaggle")], timeout=1800)
if code:
    raise RuntimeError("Kaggle output download failed")
archives = list((out / "kaggle").rglob("db7_030_results.zip"))
if len(archives) != 1:
    raise RuntimeError(f"Expected one result archive, found {len(archives)}")
with zipfile.ZipFile(archives[0]) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Corrupt result ZIP")
    completion = json.loads(archive.read("completion.json"))
    if (not completion["success"] or completion["subjects"] != list(range(1, 21))
        or completion["test_window_seed_evaluations"] != 695163
        or completion["neural_fits"] != 0):
        raise RuntimeError("Result completion check failed")
    for name in ("REPORT.md", "complementarity_summary.csv", "candidate_preference_diagnostic.csv", "feature_combination_summary.csv",
                 "subject_seed_complementarity.csv", "subject_gesture_complementarity.csv"):
        target = out / "summary" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
(out / "verification.json").write_text(json.dumps({"completion": completion,
    "result_zip_sha256": hashlib.sha256(archives[0].read_bytes()).hexdigest(),
    "kernel": ref, "dataset": dataset_ref}, indent=2))
print("DB7-030 verified complete", flush=True)
