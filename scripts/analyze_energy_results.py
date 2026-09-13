"""Analyze C1/C2A/C2D energy ablations without treating windows as independent trials.

Example:
    python analyze_energy_results.py --roots /kaggle/working/results --output /kaggle/working/analysis

No packages are installed. NumPy and pandas are required; matplotlib is optional.
The default expected study is 22 subjects x 3 seeds x 3 arms = 198 fits.
Partial folders remain useful, but are explicitly reported as incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTITY = ["subject", "seed_base", "gesture", "native_repetition", "window_start", "window_end"]
FIT_KEYS = ["arm", "subject", "seed_base"]
SPLITS = ("train", "validation", "test")
PHASES = ("early", "middle", "late", "unknown")
COMPARISONS = (("C1", "C2A"), ("C1", "C2D"), ("C2A", "C2D"))
PAIR_KEYS = ["comparison", "reference", "candidate"]
FEATURE_COLUMNS = FIT_KEYS + ["gesture", "phase", "feature", "correct_windows", "wrong_windows",
                             "correct_mean", "wrong_mean", "mean_wrong_minus_correct",
                             "correct_median", "wrong_median", "median_wrong_minus_correct"]


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_csv(output: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(output / name, index=False)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_folder(folder: Path, split: str) -> Path:
    names = (split, "val") if split == "validation" else (split,)
    return next((folder / name for name in names if (folder / name).is_dir()), folder / split)


def identity_from_manifest(manifest: dict) -> tuple[str, int, int]:
    return str(manifest["arm"]), int(manifest["subject"]), int(manifest["seed_base"])


def normalize_predictions(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {"cnn_prediction": "prediction", "predicted_gesture": "prediction",
               "cnn_confidence": "confidence", "repetition": "native_repetition"}
    for original, canonical in aliases.items():
        if canonical not in frame and original in frame:
            frame[canonical] = frame[original]
    for key in FIT_KEYS:
        if key in frame and not frame[key].eq(manifest[key]).all():
            raise ValueError(f"Prediction {key} disagrees with fit manifest")
        frame[key] = manifest[key]
    required = IDENTITY + ["prediction", "confidence"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction table is empty")
    for key in IDENTITY + ["prediction"]:
        values = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all():
            raise ValueError(f"{key} must contain finite integers")
        frame[key] = values.astype("int64")
    if not frame.gesture.between(1, 17).all() or not frame.prediction.between(1, 17).all():
        raise ValueError("Gesture and prediction must use Exercise B labels 1..17")
    if not frame.native_repetition.between(1, 6).all():
        raise ValueError("Native repetitions must be 1..6")
    if not (frame.window_end - frame.window_start).eq(800).all():
        raise ValueError("Every window must contain 800 samples (400 ms)")
    if frame.duplicated(IDENTITY).any():
        raise ValueError("Duplicate window identities within a split")
    frame["confidence"] = pd.to_numeric(frame.confidence, errors="raise")
    if not np.isfinite(frame.confidence).all() or not frame.confidence.between(0, 1).all():
        raise ValueError("Confidence must be finite and between zero and one")
    frame["correct"] = frame.prediction.eq(frame.gesture)
    frame["wrong"] = ~frame.correct
    frame["high_confidence_wrong"] = frame.wrong & frame.confidence.ge(0.90)
    if "phase" not in frame:
        if "phase_fraction" in frame:
            values = pd.to_numeric(frame.phase_fraction, errors="raise")
            if not values.between(0, 1).all():
                raise ValueError("phase_fraction must lie in [0, 1]")
            frame["phase"] = np.where(values < 1 / 3, "early", np.where(values < 2 / 3, "middle", "late"))
        else:
            frame["phase"] = "unknown"
    frame["phase"] = frame.phase.fillna("unknown").astype(str).str.lower()
    frame.loc[~frame.phase.isin(PHASES), "phase"] = "unknown"
    columns = FIT_KEYS + [c for c in IDENTITY if c not in FIT_KEYS]
    columns += ["prediction", "confidence", "correct", "wrong", "high_confidence_wrong", "phase"]
    for optional in ("endpoint_ms", "phase_fraction"):
        if optional in frame:
            frame[optional] = pd.to_numeric(frame[optional], errors="raise")
            if not np.isfinite(frame[optional]).all():
                raise ValueError(f"{optional} must be finite when supplied")
            if optional == "endpoint_ms" and frame[optional].lt(0).any():
                raise ValueError("Time since the refined-label boundary cannot be negative")
            columns.append(optional)
    return frame[columns]


def check_arrays(path: Path, frame: pd.DataFrame) -> dict:
    """Validate row order using probabilities; never unpickle object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        probability_key = next((k for k in ("probabilities", "probability", "probs", "cnn") if k in archive), None)
        embedding_key = next((k for k in ("embeddings", "embedding") if k in archive), None)
        if probability_key is None or embedding_key is None:
            raise ValueError("NPZ needs probabilities and embeddings arrays")
        probability = archive[probability_key]
        embedding = archive[embedding_key]
        if probability.shape != (len(frame), 17) or not np.isfinite(probability).all():
            raise ValueError("Probability shape/values are invalid")
        if (probability < -1e-7).any() or not np.allclose(probability.sum(1), 1, atol=2e-4):
            raise ValueError("Probability rows must sum to one")
        if not np.array_equal(probability.argmax(1) + 1, frame.prediction.to_numpy()):
            raise ValueError("NPZ probability order/predictions disagree with CSV")
        if not np.allclose(probability.max(1), frame.confidence.to_numpy(), atol=2e-4):
            raise ValueError("NPZ probabilities disagree with CSV confidence")
        if "y_true" in archive and not np.array_equal(archive["y_true"], frame.gesture.to_numpy() - 1):
            raise ValueError("NPZ true-label order disagrees with CSV")
        if embedding.ndim != 2 or len(embedding) != len(frame) or embedding.shape[1] < 1 or not np.isfinite(embedding).all():
            raise ValueError("Embedding shape/values are invalid")
        y = frame.gesture.to_numpy() - 1
        one_hot = np.eye(17)[y]
        return {"nll": float(-np.log(np.maximum(probability[np.arange(len(y)), y], 1e-12)).mean()),
                "multiclass_brier": float(np.square(probability - one_hot).sum(1).mean()),
                "embedding_dimensions": int(embedding.shape[1])}


def grouped_errors(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys, observed=True, dropna=False).agg(
        windows=("correct", "size"), correct=("correct", "sum"), wrong=("wrong", "sum"),
        high_confidence_wrong=("high_confidence_wrong", "sum"), mean_confidence=("confidence", "mean")).reset_index()
    result["accuracy_percent"] = 100 * result.correct / result.windows
    result["error_percent"] = 100 * result.wrong / result.windows
    return result


def streaks(frame: pd.DataFrame, sample_rate: float) -> dict:
    frame = frame.sort_values("window_end")
    ends = frame.window_end.to_numpy()
    step = float(pd.Series(np.diff(ends)).mode().iloc[0]) if len(ends) > 1 else np.nan
    longest = current = 0
    longest_span = current_start = 0.0
    for index, (end, wrong) in enumerate(zip(ends, frame.wrong)):
        if index and np.isfinite(step) and end - ends[index - 1] > step * 1.01:
            current = 0
        if wrong:
            if current == 0:
                current_start = float(end)
            current += 1
            longest = max(longest, current)
            longest_span = max(longest_span, float(end) - current_start)
        else:
            current = 0
    return {"windows": len(frame), "wrong": int(frame.wrong.sum()),
            "accuracy_percent": 100 * float(frame.correct.mean()), "all_wrong": bool(frame.wrong.all()),
            "all_correct": bool(frame.correct.all()), "longest_error_streak_windows": longest,
            "longest_error_endpoint_span_ms": 1000 * longest_span / sample_rate,
            "observed_modal_stride_ms": 1000 * step / sample_rate}


def checkpoint_history(folder: Path, manifest: dict) -> dict:
    result = {"selected_epoch": manifest.get("selected_epoch"), "epochs_run": manifest.get("epochs_run"),
              "parameters": manifest.get("parameters")}
    history = pd.read_csv(folder / "history.csv")
    if history.empty:
        raise ValueError("history.csv is empty")
    result["history_rows"] = len(history)
    epoch_key = next((k for k in ("epoch", "epoch_number") if k in history), None)
    if epoch_key is not None and manifest.get("selected_epoch") is not None:
        selected = history[pd.to_numeric(history[epoch_key]) == int(manifest["selected_epoch"])]
        if len(selected) != 1:
            raise ValueError("Selected epoch does not identify exactly one history row")
        for label, aliases in {
            "selected_epoch_train_accuracy_percent": ("train_accuracy", "train_acc", "training_accuracy"),
            "selected_epoch_validation_accuracy_percent": ("validation_accuracy", "val_accuracy", "val_acc"),
            "selected_epoch_train_loss": ("train_loss", "training_loss"),
            "selected_epoch_validation_loss": ("validation_loss", "val_loss"),
        }.items():
            key = next((k for k in aliases if k in selected), None)
            if key is not None:
                value = float(selected.iloc[0][key])
                result[label] = value * 100 if "accuracy" in label and 0 <= value <= 1 else value
    return result


def append_feature_contrasts(folder: Path, frame: pd.DataFrame, output: Path) -> dict:
    """Descriptive test-only comparisons; never call them train-scaled effect sizes."""
    path = folder.parent / "input_features.csv"
    feature_frame = pd.read_csv(path)
    missing = [key for key in IDENTITY if key not in feature_frame]
    if missing:
        raise ValueError(f"Input-feature identities missing: {missing}")
    if feature_frame.duplicated(IDENTITY).any():
        raise ValueError("Duplicate identities in input_features.csv")
    features = [key for key in feature_frame if re.fullmatch(r"(?:emg|acc|gyro|mag)_\d+_.+", key)
                or re.match(r"(?:energy|rms|relative_change)_", key)]
    if "stimulus_disagreement_fraction" in feature_frame:
        features.append("stimulus_disagreement_fraction")
    if not features:
        raise ValueError("No recognized diagnostic feature columns")
    joined = frame.merge(feature_frame[IDENTITY + features], on=IDENTITY, how="left", indicator=True, validate="one_to_one")
    matched = joined["_merge"].eq("both")
    audit = {**{key: frame.iloc[0][key] for key in FIT_KEYS}, "source": str(path), "test_windows": len(frame),
             "matched_feature_windows": int(matched.sum()), "unmatched_test_windows": int((~matched).sum()),
             "features": len(features), "eligible_groups": 0, "contrast_rows": 0}
    joined = joined[matched].copy()
    joined[features] = joined[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    coverage_rows = []
    tables = []
    for (gesture, phase), part in joined.groupby(["gesture", "phase"]):
        right = part.loc[part.correct, features]
        wrong = part.loc[~part.correct, features]
        eligible = phase != "unknown" and len(right) >= 3 and len(wrong) >= 3
        fit_id = {key: frame.iloc[0][key] for key in FIT_KEYS}
        coverage_rows.append({**fit_id, "gesture": gesture, "phase": phase, "correct_windows": len(right),
                              "wrong_windows": len(wrong), "eligible": eligible})
        if not eligible:
            continue
        table = pd.DataFrame({"correct_windows": right.count(), "wrong_windows": wrong.count(),
                              "correct_mean": right.mean(), "wrong_mean": wrong.mean(),
                              "correct_median": right.median(), "wrong_median": wrong.median()})
        table = table[(table.correct_windows >= 3) & (table.wrong_windows >= 3)].rename_axis("feature").reset_index()
        if table.empty:
            continue
        for key, value in {**fit_id, "gesture": gesture, "phase": phase}.items():
            table[key] = value
        table["mean_wrong_minus_correct"] = table.wrong_mean - table.correct_mean
        table["median_wrong_minus_correct"] = table.wrong_median - table.correct_median
        tables.append(table[FEATURE_COLUMNS])
        audit["eligible_groups"] += 1
    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined.to_csv(output / "matched_test_feature_contrasts.csv", mode="a", header=False, index=False)
        audit["contrast_rows"] = len(combined)
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(output / "feature_contrast_group_coverage.csv", mode="a", header=False, index=False)
    return audit


def recovery_table(pairs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = pairs.groupby(PAIR_KEYS + keys, dropna=False).agg(
        windows=("recovered", "size"), recovered=("recovered", "sum"), harmed=("harmed", "sum"),
        both_wrong=("both_wrong", "sum"), both_correct=("both_correct", "sum"),
        reference_accuracy_percent=("correct_reference", lambda values: 100 * values.mean()),
        candidate_accuracy_percent=("correct_candidate", lambda values: 100 * values.mean())).reset_index()
    table["net_recovered"] = table.recovered - table.harmed
    table["delta_pp"] = table.candidate_accuracy_percent - table.reference_accuracy_percent
    return table


def pairwise_analysis(tests: pd.DataFrame, output: Path, subjects: list[int], seeds: list[int], issue):
    paired_frames, pairing = [], []
    for reference, candidate in COMPARISONS:
        comparison = f"{reference}_to_{candidate}"
        base = tests[tests.arm.eq(reference)]
        treatment = tests[tests.arm.eq(candidate)]
        if base.empty or treatment.empty:
            continue
        common = IDENTITY + ["prediction", "confidence", "correct", "phase"]
        timed = "endpoint_ms" in base and "endpoint_ms" in treatment and base.endpoint_ms.notna().all() and treatment.endpoint_ms.notna().all()
        if timed:
            common.append("endpoint_ms")
        joined = base[common].merge(treatment[common], on=IDENTITY, how="outer",
                                   suffixes=("_reference", "_candidate"), indicator=True, validate="one_to_one")
        for (subject, seed), part in joined.groupby(["subject", "seed_base"]):
            counts = part["_merge"].value_counts()
            pairing.append(dict(comparison=comparison, reference=reference, candidate=candidate,
                                subject=subject, seed_base=seed, matched_windows=int(counts.get("both", 0)),
                                reference_only_windows=int(counts.get("left_only", 0)),
                                candidate_only_windows=int(counts.get("right_only", 0))))
        if not joined["_merge"].eq("both").all():
            issue(comparison, "Exact test-window identity mismatch; this comparison is withheld instead of analyzing an intersection", "error")
            continue
        if not joined.phase_reference.eq(joined.phase_candidate).all():
            issue(comparison, "Matched windows disagree on phase; this comparison is withheld", "error")
            continue
        if timed and not np.array_equal(joined.endpoint_ms_reference, joined.endpoint_ms_candidate):
            issue(comparison, "Matched windows disagree on time since the refined-label boundary", "error")
            continue
        pairs = joined.drop(columns="_merge").copy()
        for arm_label in ["reference", "candidate"]:
            pairs[f"correct_{arm_label}"] = pairs[f"correct_{arm_label}"].astype(bool)
        pairs["reference"], pairs["candidate"], pairs["comparison"] = reference, candidate, comparison
        pairs["recovered"] = ~pairs.correct_reference & pairs.correct_candidate
        pairs["harmed"] = pairs.correct_reference & ~pairs.correct_candidate
        pairs["both_wrong"] = ~pairs.correct_reference & ~pairs.correct_candidate
        pairs["both_correct"] = pairs.correct_reference & pairs.correct_candidate
        pairs["phase"] = pairs.phase_reference
        if timed:
            pairs["endpoint_ms"] = pairs.endpoint_ms_reference
            pairs["endpoint_bin_start_ms"] = (np.floor(pairs.endpoint_ms / 100) * 100).astype(int)
        else:
            issue(comparison, "Endpoint timing is unavailable; no timing is inferred from test-window order")
        paired_frames.append(pairs)
    pairing_df = pd.DataFrame(pairing, columns=PAIR_KEYS + ["subject", "seed_base", "matched_windows", "reference_only_windows", "candidate_only_windows"])
    write_csv(output, "window_pairing_audit.csv", pairing_df)
    if not paired_frames:
        return pd.DataFrame(), pd.DataFrame(), pairing_df
    pairs = pd.concat(paired_frames, ignore_index=True)
    assert not pairs.duplicated(["comparison"] + IDENTITY).any()
    pairs.to_csv(output / "paired_test_windows.csv.gz", index=False, compression="gzip")
    paired_subjects = pd.DataFrame()
    for filename, keys in [
        ("paired_subject_recovery.csv", ["subject", "seed_base"]),
        ("paired_gesture_repetition_recovery.csv", ["subject", "seed_base", "gesture", "native_repetition"]),
        ("paired_phase_recovery.csv", ["subject", "seed_base", "phase"]),
    ]:
        table = recovery_table(pairs, keys)
        write_csv(output, filename, table)
        if filename == "paired_subject_recovery.csv":
            paired_subjects = table
    if "endpoint_bin_start_ms" in pairs:
        timed = pairs[pairs.endpoint_bin_start_ms.notna()]
        write_csv(output, "paired_endpoint_100ms_recovery.csv", recovery_table(timed, ["subject", "seed_base", "endpoint_bin_start_ms"]))
    persistent_keys = PAIR_KEYS + [key for key in IDENTITY if key != "seed_base"]
    persistent = pairs.groupby(persistent_keys).agg(available_seeds=("seed_base", "nunique"),
        both_wrong_seeds=("both_wrong", "sum"), recovered_seeds=("recovered", "sum"), harmed_seeds=("harmed", "sum")).reset_index()
    persistent["both_wrong_in_all_expected_seeds"] = persistent.available_seeds.eq(len(seeds)) & persistent.both_wrong_seeds.eq(len(seeds))
    write_csv(output, "persistent_paired_window_errors.csv", persistent)

    summaries, averages, seed_tables = [], [], []
    rng = np.random.default_rng(20260913)
    for (comparison, reference, candidate), table in paired_subjects.groupby(PAIR_KEYS):
        by_seed = table.groupby("seed_base").agg(subjects=("subject", "nunique"),
            reference_accuracy_percent=("reference_accuracy_percent", "mean"),
            candidate_accuracy_percent=("candidate_accuracy_percent", "mean"), delta_pp=("delta_pp", "mean"),
            recovered=("recovered", "sum"), harmed=("harmed", "sum"), both_wrong=("both_wrong", "sum")).reset_index()
        for key, value in zip(PAIR_KEYS, (comparison, reference, candidate)):
            by_seed[key] = value
        seed_tables.append(by_seed)
        per_subject = table.groupby("subject").agg(seeds=("seed_base", "nunique"),
            reference_accuracy_percent=("reference_accuracy_percent", "mean"),
            candidate_accuracy_percent=("candidate_accuracy_percent", "mean"), delta_pp=("delta_pp", "mean")).reset_index()
        per_subject = per_subject[per_subject.seeds.eq(len(seeds)) & per_subject.subject.isin(subjects)].copy()
        for key, value in zip(PAIR_KEYS, (comparison, reference, candidate)):
            per_subject[key] = value
        averages.append(per_subject)
        delta = per_subject.delta_pp.to_numpy()
        low = high = np.nan
        if len(delta) >= 2:
            distribution = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(1)
            low, high = np.quantile(distribution, [.025, .975])
        summaries.append(dict(comparison=comparison, reference=reference, candidate=candidate,
            fully_matched_subjects=len(per_subject), seeds=len(seeds),
            reference_accuracy_percent=per_subject.reference_accuracy_percent.mean(),
            candidate_accuracy_percent=per_subject.candidate_accuracy_percent.mean(), mean_gain_pp=delta.mean() if len(delta) else np.nan,
            bootstrap_low_pp=low, bootstrap_high_pp=high, subjects_better=int((delta > 0).sum()),
            subjects_worse=int((delta < 0).sum()), subjects_tied=int((delta == 0).sum()),
            recovered=int(table.recovered.sum()), harmed=int(table.harmed.sum()), both_wrong=int(table.both_wrong.sum()),
            net_recovered=int(table.net_recovered.sum()), min_seed_gain_pp=by_seed.delta_pp.min(),
            max_seed_gain_pp=by_seed.delta_pp.max(), all_expected_subjects=len(per_subject) == len(subjects)))
    write_csv(output, "comparison_summary.csv", pd.DataFrame(summaries))
    write_csv(output, "comparison_by_seed.csv", pd.concat(seed_tables, ignore_index=True))
    write_csv(output, "paired_subject_seed_averages.csv", pd.concat(averages, ignore_index=True))
    return pairs, paired_subjects, pairing_df


def energy_feature_summary(output: Path):
    contrasts = pd.read_csv(output / "matched_test_feature_contrasts.csv")
    if contrasts.empty:
        write_csv(output, "matched_energy_feature_contrasts.csv", contrasts)
        write_csv(output, "energy_feature_summary.csv", pd.DataFrame(columns=["arm", "feature", "groups"]))
        return
    energy = contrasts[contrasts.feature.str.contains(r"energy|rms|mav|relative|change|ratio|delta|log_amp", case=False, regex=True)]
    write_csv(output, "matched_energy_feature_contrasts.csv", energy)
    summary = energy.groupby(["arm", "feature"]).agg(subjects=("subject", "nunique"), groups=("gesture", "size"),
        median_group_wrong_minus_correct=("median_wrong_minus_correct", "median"),
        mean_group_wrong_minus_correct=("mean_wrong_minus_correct", "mean")).reset_index()
    write_csv(output, "energy_feature_summary.csv", summary)


def analyze(roots: list[Path], output: Path, subjects: list[int], seeds: list[int], arms: list[str], make_plots: bool = True) -> dict:
    if any(output.resolve() == root.resolve() or output.resolve() in root.resolve().parents for root in roots):
        raise ValueError("Analysis output must not equal or contain an input root")
    output.mkdir(parents=True, exist_ok=True)
    expected = set(itertools.product(arms, subjects, seeds))
    json_write(output / "analysis_status.json", {"study_complete": False, "analysis_in_progress": True})
    (output / "RESULTS.md").write_text("# Analysis in progress\n\nA full-study result has not yet been verified for this analysis run.\n", encoding="utf-8")
    # Replace prior optional tables even if this run has fewer or no usable fits.
    for name in ("subject_metrics.csv", "subject_gesture_repetition_errors.csv", "phase_errors.csv", "accuracy_by_arm_seed.csv",
                 "seed_spread_by_subject.csv", "confusion_counts.csv", "paired_subject_recovery.csv", "paired_gesture_repetition_recovery.csv",
                 "paired_phase_recovery.csv", "persistent_paired_window_errors.csv", "paired_subject_seed_averages.csv",
                 "comparison_summary.csv", "comparison_by_seed.csv", "paired_endpoint_100ms_recovery.csv",
                 "endpoint_100ms_errors.csv", "persistent_all_arm_window_errors.csv"):
        write_csv(output, name, pd.DataFrame(columns=FIT_KEYS))
    pd.DataFrame(columns=IDENTITY).to_csv(output / "paired_test_windows.csv.gz", index=False, compression="gzip")
    write_csv(output, "matched_test_feature_contrasts.csv", pd.DataFrame(columns=FEATURE_COLUMNS))
    write_csv(output, "feature_contrast_group_coverage.csv", pd.DataFrame(columns=FIT_KEYS + ["gesture", "phase", "correct_windows", "wrong_windows", "eligible"]))
    issues: list[dict] = []
    def issue(path: Path | str, detail: str, severity: str = "warning") -> None:
        issues.append({"path": str(path), "severity": severity, "detail": detail})

    candidates: dict[tuple, list[tuple[Path, dict]]] = {}
    for root in roots:
        if not root.is_dir():
            issue(root, "Input root does not exist", "error")
            continue
        for path in root.rglob("fit_manifest.json"):
            if output.resolve() in path.resolve().parents:
                continue
            try:
                manifest = json_read(path)
                key = identity_from_manifest(manifest)
                manifest.update(dict(zip(FIT_KEYS, key)))
                if not any(p.resolve() == path.resolve() for p, _ in candidates.get(key, [])):
                    candidates.setdefault(key, []).append((path, manifest))
            except Exception as exc:
                issue(path, f"Unreadable fit identity: {exc}", "error")

    unique = []
    for key, entries in candidates.items():
        if len(entries) > 1:
            consistent = True
            for split in SPLITS:
                files = [split_folder(path.parent, split) / "predictions.csv" for path, _ in entries]
                hashes = {digest(path) for path in files if path.is_file()}
                consistent &= len(hashes) <= 1
            if not consistent:
                issue(str(key), "Conflicting duplicate fit identities; excluded instead of choosing a result", "error")
                continue
            entries.sort(key=lambda item: sum(p.is_file() for s in SPLITS for p in
                (split_folder(item[0].parent, s) / "predictions.csv", split_folder(item[0].parent, s) / "probabilities_embeddings.npz")), reverse=True)
            issue(str(key), f"Found {len(entries)} compatible copies; used {entries[0][0].parent}")
        unique.append(entries[0])

    inventory, summaries, history_rows, test_frames, trial_rows, calibration_rows, feature_audits = [], [], [], [], [], [], []
    assignments = {}
    grid_hashes = {}
    for manifest_path, manifest in sorted(unique, key=lambda item: identity_from_manifest(item[1])):
        folder = manifest_path.parent
        fit_id = {k: manifest[k] for k in FIT_KEYS}
        if identity_from_manifest(manifest) not in expected:
            issue(manifest_path, "Fit lies outside the requested subjects/seeds/arms; excluded from this study summary")
            continue
        inventory_row = {**fit_id, "folder": str(folder), "complete": True, "usable_test": False}
        if manifest.get("status", "complete") != "complete":
            inventory_row["complete"] = False
            issue(manifest_path, "Fit manifest is not marked complete")
        summary = {**fit_id}
        split_ids = {}
        for name in ("metrics.json", "history.csv"):
            if not (folder / name).is_file():
                inventory_row["complete"] = False
                issue(folder / name, "Required fit artifact missing")
        if (folder / "metrics.json").is_file():
            try:
                json_read(folder / "metrics.json")
            except Exception as exc:
                inventory_row["complete"] = False
                issue(folder / "metrics.json", f"Invalid metrics JSON: {exc}", "error")
        if (folder / "history.csv").is_file():
            try:
                history_rows.append({**fit_id, **checkpoint_history(folder, manifest)})
            except Exception as exc:
                inventory_row["complete"] = False
                issue(folder / "history.csv", str(exc), "error")
        for split in SPLITS:
            split_dir = split_folder(folder, split)
            path = split_dir / "predictions.csv"
            if not path.is_file():
                inventory_row["complete"] = False
                issue(path, "Prediction split not yet available")
                continue
            try:
                frame = normalize_predictions(pd.read_csv(path), manifest)
                grid_hash = hashlib.sha256(frame[[key for key in IDENTITY if key != "seed_base"]].to_csv(index=False).encode()).hexdigest()
                if "window_key_hashes" in manifest and grid_hash != manifest["window_key_hashes"][split]:
                    raise ValueError("Prediction identities disagree with the saved split hash")
                grid_hashes.setdefault((manifest["subject"], split), set()).add(grid_hash)
                expected_count = manifest.get(f"{split}_windows")
                if expected_count is not None and len(frame) != int(expected_count):
                    raise ValueError(f"Expected {expected_count} windows, found {len(frame)}")
                split_ids[split] = set(map(tuple, frame[["gesture", "native_repetition"]].drop_duplicates().to_numpy()))
                summary[f"{split}_windows"] = len(frame)
                summary[f"{split}_accuracy_percent"] = 100 * float(frame.correct.mean())
                arrays_path = split_dir / "probabilities_embeddings.npz"
                if arrays_path.is_file():
                    try:
                        calibration_rows.append({**fit_id, "split": split, **check_arrays(arrays_path, frame)})
                    except Exception as exc:
                        inventory_row["complete"] = False
                        issue(arrays_path, str(exc), "error")
                else:
                    inventory_row["complete"] = False
                    issue(arrays_path, "Probability/embedding artifact missing; CSV analysis remains available")
                if split == "test":
                    inventory_row["usable_test"] = True
                    test_frames.append(frame)
                    feature_path = folder.parent / "input_features.csv"
                    if feature_path.is_file():
                        try:
                            feature_audit = append_feature_contrasts(folder, frame, output)
                            feature_audits.append(feature_audit)
                            if feature_audit["unmatched_test_windows"]:
                                issue(feature_path, "Some test windows lack matching input diagnostics; contrasts use matched rows only")
                        except Exception as exc:
                            issue(feature_path, f"Input feature diagnostics unavailable for this fit: {exc}")
                    else:
                        issue(feature_path, "Optional test feature diagnostics not available")
                    if frame.phase.eq("unknown").any():
                        issue(path, "Some phases are unknown; not inferred from the observed test-window count")
                    for (gesture, repetition), part in frame.groupby(["gesture", "native_repetition"]):
                        trial_rows.append({**fit_id, "gesture": gesture, "native_repetition": repetition,
                                           **streaks(part, float(manifest.get("sample_rate_hz", manifest.get("fs_emg", 2000))))})
                    reps_per_gesture = frame.groupby("gesture").native_repetition.nunique()
                    if len(reps_per_gesture) != 17 or not reps_per_gesture.eq(1).all():
                        inventory_row["complete"] = False
                        issue(path, "Expected one test repetition for each of all 17 gestures", "error")
            except Exception as exc:
                inventory_row["complete"] = False
                issue(path, f"Invalid prediction split: {exc}", "error")
        for split, expected_repetitions in (("train", 4), ("validation", 1), ("test", 1)):
            if split in split_ids:
                counts = pd.Series([gesture for gesture, _ in split_ids[split]]).value_counts()
                if len(counts) != 17 or not counts.eq(expected_repetitions).all():
                    inventory_row["complete"] = False
                    issue(folder, f"{split} must contain {expected_repetitions} native repetitions per gesture for all 17 gestures", "error")
        for left, right in itertools.combinations(split_ids, 2):
            if split_ids[left] & split_ids[right]:
                inventory_row["complete"] = False
                issue(folder, f"Repetition leakage: {left} and {right} share gesture/repetition identities", "error")
        if len(split_ids) == 3:
            assignment = {split: sorted(list(values)) for split, values in split_ids.items()}
            assignments.setdefault(manifest["subject"], []).append((folder, json.dumps(clean_json(assignment), sort_keys=True)))
        inventory.append(inventory_row)
        summaries.append(summary)

    for subject, versions in assignments.items():
        if len({value for _, value in versions}) > 1:
            issue(f"subject {subject}", "Repetition allocation changes between arms/seeds; not the stipulated fixed-split comparison", "error")
    for (subject, split), versions in grid_hashes.items():
        if len(versions) != 1:
            issue(f"subject {subject}/{split}", "Exact window grid differs across arms/seeds", "error")
    inventory_df = pd.DataFrame(inventory, columns=FIT_KEYS + ["folder", "complete", "usable_test"])
    complete_keys = {tuple(row[k] for k in FIT_KEYS) for row in inventory if row["complete"]}
    missing = pd.DataFrame(sorted(expected - complete_keys), columns=FIT_KEYS)
    summary_df = pd.DataFrame(summaries)
    history_df = pd.DataFrame(history_rows)
    if not summary_df.empty:
        if not history_df.empty:
            summary_df = summary_df.merge(history_df, on=FIT_KEYS, how="left", validate="one_to_one")
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if f"{left}_accuracy_percent" in summary_df and f"{right}_accuracy_percent" in summary_df:
                summary_df[f"{left}_minus_{right}_pp"] = summary_df[f"{left}_accuracy_percent"] - summary_df[f"{right}_accuracy_percent"]
    write_csv(output, "fit_inventory.csv", inventory_df)
    write_csv(output, "missing_or_incomplete_fits.csv", missing)
    write_csv(output, "generalization_gaps.csv", summary_df)
    write_csv(output, "probability_diagnostics.csv", pd.DataFrame(calibration_rows))
    write_csv(output, "repetition_failures.csv", pd.DataFrame(trial_rows))
    write_csv(output, "feature_input_audit.csv", pd.DataFrame(feature_audits))

    subject_metrics = pd.DataFrame()
    arm_seed = pd.DataFrame()
    pairs = pd.DataFrame()
    paired_subjects = pd.DataFrame()
    pairing_df = pd.DataFrame(columns=PAIR_KEYS + ["subject", "seed_base", "matched_windows", "reference_only_windows", "candidate_only_windows"])
    figures = []
    if test_frames:
        tests = pd.concat(test_frames, ignore_index=True)
        subject_metrics = grouped_errors(tests, FIT_KEYS)
        gesture_metrics = grouped_errors(tests, FIT_KEYS + ["gesture", "native_repetition"])
        phase_metrics = grouped_errors(tests, FIT_KEYS + ["phase"])
        arm_seed = subject_metrics.groupby(["arm", "seed_base"]).agg(
            subjects=("subject", "nunique"), mean_subject_accuracy_percent=("accuracy_percent", "mean"),
            sd_subject_accuracy_pp=("accuracy_percent", "std"), windows=("windows", "sum"), wrong=("wrong", "sum")).reset_index()
        arm_seed["pooled_window_accuracy_percent"] = 100 * (1 - arm_seed.wrong / arm_seed.windows)
        arm_seed["all_expected_subjects"] = arm_seed.subjects.eq(len(subjects))
        seed_spread = subject_metrics.groupby(["arm", "subject"]).agg(
            available_seeds=("seed_base", "nunique"), mean_accuracy_percent=("accuracy_percent", "mean"),
            seed_sd_pp=("accuracy_percent", "std"), min_accuracy_percent=("accuracy_percent", "min"),
            max_accuracy_percent=("accuracy_percent", "max")).reset_index()
        seed_spread["seed_range_pp"] = seed_spread.max_accuracy_percent - seed_spread.min_accuracy_percent
        confusion = tests.groupby(FIT_KEYS + ["gesture", "prediction"]).size().rename("windows").reset_index()
        write_csv(output, "subject_metrics.csv", subject_metrics)
        write_csv(output, "subject_gesture_repetition_errors.csv", gesture_metrics)
        write_csv(output, "phase_errors.csv", phase_metrics)
        write_csv(output, "accuracy_by_arm_seed.csv", arm_seed)
        write_csv(output, "seed_spread_by_subject.csv", seed_spread)
        write_csv(output, "confusion_counts.csv", confusion)
        pairs, paired_subjects, pairing_df = pairwise_analysis(tests, output, subjects, seeds, issue)
        if "endpoint_ms" in tests:
            timed = tests[tests.endpoint_ms.notna()].copy()
            timed["endpoint_bin_start_ms"] = (np.floor(timed.endpoint_ms / 100) * 100).astype(int)
            write_csv(output, "endpoint_100ms_errors.csv", grouped_errors(timed, FIT_KEYS + ["endpoint_bin_start_ms"]))
        wide = tests.pivot(index=IDENTITY, columns="arm", values="wrong").reindex(columns=arms)
        all_wrong = wide.notna().all(axis=1) & wide.fillna(False).all(axis=1)
        persistent = wide.reset_index()[IDENTITY].copy()
        persistent["all_arms_available"] = wide.notna().all(axis=1).to_numpy()
        persistent["all_arms_wrong"] = all_wrong.to_numpy()
        keys = [key for key in IDENTITY if key != "seed_base"]
        persistent = persistent.groupby(keys).agg(available_seeds=("seed_base", "nunique"),
            complete_arm_seeds=("all_arms_available", "sum"), all_arms_wrong_seeds=("all_arms_wrong", "sum")).reset_index()
        persistent["all_arms_wrong_in_all_expected_seeds"] = persistent.complete_arm_seeds.eq(len(seeds)) & persistent.all_arms_wrong_seeds.eq(len(seeds))
        write_csv(output, "persistent_all_arm_window_errors.csv", persistent)
        if make_plots:
            try:
                figures = plots(output, subject_metrics, gesture_metrics, phase_metrics, paired_subjects)
            except ImportError:
                issue("matplotlib", "Optional plotting library unavailable; CSV and Markdown analysis still complete")
            except Exception as exc:
                issue("matplotlib", f"Could not produce optional figures: {exc}")

    energy_feature_summary(output)
    status = {"expected_fits": len(expected), "discovered_unique_fit_identities": len(candidates),
              "usable_fit_folders": len(inventory), "complete_expected_fits": len(expected & complete_keys),
              "usable_test_fits": int(inventory_df.usable_test.sum()) if len(inventory_df) else 0,
              "missing_or_incomplete_expected_fits": len(expected - complete_keys),
              "study_complete": not (expected - complete_keys), "expected_subjects": subjects,
              "expected_seed_bases": seeds, "expected_arms": arms, "roots": [str(p) for p in roots],
              "paired_windows_across_seeds": len(pairs[IDENTITY].drop_duplicates()) if len(pairs) else 0,
              "paired_prediction_instances_by_comparison": pairs.groupby("comparison").size().to_dict() if len(pairs) else {},
              "figures": figures,
              "fits_with_matched_feature_diagnostics": len(feature_audits),
              "feature_contrast_rows": sum(row["contrast_rows"] for row in feature_audits),
              "warnings": sum(row["severity"] == "warning" for row in issues), "errors": sum(row["severity"] == "error" for row in issues)}
    if status["errors"]:
        status["study_complete"] = False
    write_csv(output, "analysis_issues.csv", pd.DataFrame(issues, columns=["path", "severity", "detail"]))
    json_write(output / "analysis_status.json", clean_json(status))
    report(output, status, arm_seed, paired_subjects, pairing_df, issues)
    return status


def plots(output: Path, subjects: pd.DataFrame, gestures: pd.DataFrame, phases: pd.DataFrame, paired: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    names = []
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for arm, part in subjects.groupby("arm"):
        group = part.groupby("subject").accuracy_percent.agg(["mean", "min", "max"])
        ax.errorbar(group.index, group["mean"], yerr=[group["mean"]-group["min"], group["max"]-group["mean"]],
                    marker="o", label=arm, capsize=3)
    ax.set(xlabel="Subject", ylabel="Test accuracy (%)", title="Subject accuracy: mean and range across available seeds")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(directory / "subject_accuracy.png", dpi=160); plt.close(fig)
    names.append("figures/subject_accuracy.png")
    known = phases[phases.phase.isin(PHASES[:3])]
    if not known.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm, part in known.groupby("arm"):
            values = part.groupby("phase").error_percent.mean().reindex(PHASES[:3])
            ax.plot(values.index, values, marker="o", label=arm)
        ax.set(xlabel="Repetition-relative phase", ylabel="Mean subject/seed error (%)", title="Phase error rates (available fits)")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(directory / "phase_errors.png", dpi=160); plt.close(fig)
        names.append("figures/phase_errors.png")
    for arm, part in gestures.groupby("arm"):
        grid = part.pivot_table(index="subject", columns="gesture", values="error_percent", aggfunc="mean").reindex(columns=range(1, 18))
        fig, ax = plt.subplots(figsize=(11, max(4.5, len(grid) * .27)))
        im = ax.imshow(grid, vmin=0, vmax=100, aspect="auto", cmap="Reds")
        ax.set_xticks(range(17), range(1, 18)); ax.set_yticks(range(len(grid)), grid.index)
        ax.set(xlabel="Gesture", ylabel="Subject", title=f"{arm}: test error percentage, mean across available seeds")
        fig.colorbar(im, ax=ax, label="Error (%)"); fig.tight_layout()
        filename = f"gesture_errors_{''.join(c for c in arm if c.isalnum())}.png"
        fig.savefig(directory / filename, dpi=160); plt.close(fig)
        names.append(f"figures/{filename}")
    for comparison, part in paired.groupby("comparison") if not paired.empty else []:
        values = part.groupby("subject")[["recovered", "harmed"]].sum()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(values))
        ax.bar(x-.2, values.recovered, width=.4, label="Reference errors recovered")
        ax.bar(x+.2, values.harmed, width=.4, label="New candidate errors")
        ax.set_xticks(x, values.index); ax.set(xlabel="Subject", ylabel="Matched window predictions, summed across seeds",
            title=f"{comparison}: recovery and harm (repeated, overlapping windows)")
        filename = f"recovery_harm_{comparison}.png"
        ax.legend(); fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)
        names.append(f"figures/{filename}")
    endpoint_path = output / "endpoint_100ms_errors.csv"
    if endpoint_path.is_file():
        endpoint = pd.read_csv(endpoint_path)
        if "endpoint_bin_start_ms" in endpoint and not endpoint.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for arm, part in endpoint.groupby("arm"):
                group = part.groupby("endpoint_bin_start_ms").agg(wrong=("wrong", "sum"), windows=("windows", "sum"))
                ax.plot(group.index + 50, 100 * group.wrong / group.windows, label=arm)
            ax.set(xlabel="Window endpoint since refined-label boundary (ms; 100 ms bins)",
                   ylabel="Pooled error (%)", title="Label-relative time; not a physiological onset measurement")
            ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(directory / "endpoint_errors.png", dpi=160); plt.close(fig)
            names.append("figures/endpoint_errors.png")
    return names


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "No eligible results are available yet."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("unavailable")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(output: Path, status: dict, arm_seed: pd.DataFrame, paired: pd.DataFrame, pairing: pd.DataFrame, issues: list[dict]) -> None:
    def read(name):
        path = output / name
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    summary = read("comparison_summary.csv")
    by_seed = read("comparison_by_seed.csv")
    state = "COMPLETE" if status["study_complete"] else "PARTIAL / INCOMPLETE"
    lines = ["# DB7 energy-ablation results", "", f"**Status: {state}.** "
             f"{status['complete_expected_fits']} of {status['expected_fits']} expected fits passed artifact checks.", "",
             "C1 is the original three-branch reference; C2A adds amplitude-only energy cues; C2D adds energy and relative-change cues. "
             "C1 versus C2A asks whether explicit amplitude helps. C1 versus C2D tests the full energy/change addition. "
             "C2A versus C2D is the direct comparison of the two energy treatments; it does not by itself identify a physiological cause.", "",
             "All arms retain Exercise B labels 1–17, 400 ms windows and 100 ms output stride. Repetition assignments are fixed while "
             "model seeds change. Accuracy is recomputed from predictions; subjects are weighted equally and seeds averaged within subject.", ""]
    if not summary.empty and "mean_gain_pp" in summary:
        lines += ["## Did the energy additions improve accuracy?", "",
                  markdown_table(summary, ["comparison", "fully_matched_subjects", "reference_accuracy_percent",
                      "candidate_accuracy_percent", "mean_gain_pp", "bootstrap_low_pp", "bootstrap_high_pp",
                      "subjects_better", "subjects_worse"]), ""]
        for row in summary.to_dict("records"):
            if pd.isna(row["mean_gain_pp"]):
                continue
            movement = "increased" if row["mean_gain_pp"] > 0 else "decreased" if row["mean_gain_pp"] < 0 else "did not change"
            lines.append(f"**{row['comparison']}**: mean accuracy {movement} by {abs(row['mean_gain_pp']):.3f} percentage points. "
                         f"The candidate recovered {int(row['recovered']):,} reference errors and introduced {int(row['harmed']):,} new errors "
                         f"(net {int(row['net_recovered']):+,} corrected prediction instances across the same windows and seeds).")
        lines += ["", "Intervals use 20,000 paired subject-bootstrap draws after averaging seeds within each subject. "
                  "They are conditional on the fixed repetition split and available seed set, and are not adjusted for the three comparisons. "
                  "Overlapping windows and repeated-seed predictions are not independent trials. Partial arms must not be ranked as a controlled complete study.", ""]
    else:
        lines += ["No exact-grid paired comparison is available yet. Missing or mismatched grids are not silently reduced to an intersection.", ""]
    lines += ["## Does the effect repeat across seeds?", "",
              markdown_table(arm_seed, ["arm", "seed_base", "subjects", "mean_subject_accuracy_percent", "pooled_window_accuracy_percent", "all_expected_subjects"]), ""]
    if not by_seed.empty and "delta_pp" in by_seed:
        lines += [markdown_table(by_seed, ["comparison", "seed_base", "subjects", "delta_pp", "recovered", "harmed", "both_wrong"]), ""]
        for comparison, group in by_seed.groupby("comparison"):
            lines.append(f"{comparison}: seed-specific mean gains range from {group.delta_pp.min():+.3f} to {group.delta_pp.max():+.3f} pp; "
                         f"{int(group.delta_pp.gt(0).sum())} of {len(group)} seeds improved. This repeats optimization on the same test repetitions.")
    spread = read("seed_spread_by_subject.csv")
    if not spread.empty and "seed_range_pp" in spread:
        table = spread.groupby("arm").agg(median_subject_seed_range_pp=("seed_range_pp", "median"),
            maximum_subject_seed_range_pp=("seed_range_pp", "max")).reset_index()
        lines += ["", markdown_table(table, list(table.columns))]

    phase = read("paired_phase_recovery.csv")
    lines += ["", "## Were early errors fixed, and were correct windows damaged?", ""]
    if not phase.empty and "phase" in phase:
        rows = []
        for keys, group in phase.groupby(PAIR_KEYS + ["phase"]):
            windows = int(group.windows.sum())
            recovered, harmed = int(group.recovered.sum()), int(group.harmed.sum())
            both = int(group.both_wrong.sum())
            rows.append(dict(zip(PAIR_KEYS + ["phase"], keys)) | dict(windows=windows,
                reference_error_percent=100*(recovered+both)/windows,
                candidate_error_percent=100*(harmed+both)/windows,
                recovered=recovered, harmed=harmed, net_recovered=recovered-harmed))
        pooled_phase = pd.DataFrame(rows)
        write_csv(output, "phase_recovery_summary.csv", pooled_phase)
        lines += [markdown_table(pooled_phase, ["comparison", "phase", "windows", "reference_error_percent",
                  "candidate_error_percent", "recovered", "harmed", "net_recovered"]), ""]
        for comparison, group in pooled_phase.groupby("comparison"):
            early = group[group.phase.eq("early")]
            later = group[group.phase.isin(["middle", "late"])]
            if len(early):
                row = early.iloc[0]
                lines.append(f"{comparison}: early-window errors were {row.reference_error_percent:.2f}% for the reference and "
                             f"{row.candidate_error_percent:.2f}% for the candidate; {int(row.recovered):,} were recovered and "
                             f"{int(row.harmed):,} previously correct early predictions became wrong.")
            if len(later):
                lines.append(f"In middle/late windows, {comparison} introduced {int(later.harmed.sum()):,} errors while recovering "
                             f"{int(later.recovered.sum()):,}; net change {int(later.net_recovered.sum()):+,} corrections. "
                             "The phase table preserves middle and late separately.")
        lines += ["", "These percentages pool prediction instances within each phase; they are not independent-sample estimates. "
                  "Early/middle/late refer to thirds of the refined-label repetition, not measured muscle onset."]
    else:
        lines.append("Paired phase data are unavailable; unknown phases are not inferred.")
    endpoint = read("paired_endpoint_100ms_recovery.csv")
    if not endpoint.empty and "endpoint_bin_start_ms" in endpoint:
        bins = sorted(endpoint.endpoint_bin_start_ms.unique())
        lines += ["", "A separate [100 ms endpoint-bin table](paired_endpoint_100ms_recovery.csv) measures elapsed time "
                  "from the refined-label boundary to the prediction window endpoint. Bins are [start, start+100) ms. "
                  f"Observed bins begin at {min(bins):.0f} ms; no accuracy is inferred before the first available window. "
                  "This time is not a physiological onset measurement. [Per-arm errors](endpoint_100ms_errors.csv) retain denominators."]
    else:
        lines += ["", "Endpoint timing was unavailable; no 100 ms timing result was fabricated from window order."]

    generalization = read("generalization_gaps.csv")
    lines += ["", "## Training, validation and test behavior", ""]
    metrics = ["train_accuracy_percent", "validation_accuracy_percent", "test_accuracy_percent", "train_minus_validation_pp", "train_minus_test_pp", "validation_minus_test_pp"]
    available = [name for name in metrics if name in generalization]
    if available and not generalization.empty:
        table = generalization.groupby("arm")[available].mean().reset_index()
        lines += [markdown_table(table, ["arm"] + available), "",
                  "These are saved evaluation-mode predictions at the selected validation-loss checkpoint. Training-mode epoch-history "
                  "accuracy may differ. A small gain with a larger validation/test gap is not evidence that adding features solved generalization."]
    subjects = read("subject_metrics.csv")
    if not subjects.empty and "accuracy_percent" in subjects:
        means = subjects.groupby(["arm", "subject"]).accuracy_percent.mean().reset_index()
        cohorts = []
        for name, ids in [("S1–S20", list(range(1,21))), ("S21–S22", [21,22]), ("S21", [21]), ("S22", [22])]:
            for arm, part in means[means.subject.isin(ids)].groupby("arm"):
                cohorts.append(dict(cohort=name, arm=arm, subjects=part.subject.nunique(), accuracy_percent=part.accuracy_percent.mean()))
        cohort = pd.DataFrame(cohorts)
        write_csv(output, "cohort_accuracy.csv", cohort)
        lines += ["", markdown_table(cohort, list(cohort.columns)), "", "The S21/S22 group contains only two people; retain their individual results."]

    lines += ["", "## Which subjects and gestures still fail?", "",
              "[Subject × gesture × seed error counts](subject_gesture_repetition_errors.csv) retain correct, wrong and total windows. "
              "[Paired gesture/repetition recoveries](paired_gesture_repetition_recovery.csv) show precisely which treatment fixed or harmed each trial. "
              "Each gesture has one test repetition per subject, so a difficult cell does not establish a generally difficult gesture."]
    trials = read("repetition_failures.csv")
    if not trials.empty and "all_wrong" in trials:
        table = trials.groupby("arm").agg(trial_prediction_instances=("subject", "size"),
            whole_trial_wrong=("all_wrong", "sum"), perfect_trials=("all_correct", "sum")).reset_index()
        lines += ["", markdown_table(table, list(table.columns)), "", "Trial instances repeat the same physical repetition under each model seed."]
    persistent = read("persistent_paired_window_errors.csv")
    if not persistent.empty and "both_wrong_in_all_expected_seeds" in persistent:
        table = persistent.groupby("comparison").both_wrong_in_all_expected_seeds.sum().rename("physical_windows_both_wrong_every_seed").reset_index()
        lines += ["", markdown_table(table, list(table.columns))]
    all_persistent = read("persistent_all_arm_window_errors.csv")
    if not all_persistent.empty and "all_arms_wrong_in_all_expected_seeds" in all_persistent:
        n = int(all_persistent.all_arms_wrong_in_all_expected_seeds.sum())
        lines += ["", f"**{n:,} physical windows were wrong for every arm in every expected seed.** "
                  "[Persistent failures](persistent_all_arm_window_errors.csv) identify their subject, gesture, repetition and endpoints. "
                  "This is a failure set for the tested models, not a physiological information ceiling."]

    lines += ["", "## Do the energy features differ between correct and wrong cases?", "",
              f"Matched input diagnostics cover {status['fits_with_matched_feature_diagnostics']} fits and "
              f"{status['feature_contrast_rows']:,} feature/group contrasts. "
              "[Energy-feature contrasts](matched_energy_feature_contrasts.csv) include RMS, amplitude, energy and relative-change descriptors when exported. "
              "[Per-feature summary](energy_feature_summary.csv) aggregates only the same named feature across eligible groups.", "",
              "Comparisons are made within subject, seed, gesture and known phase, requiring at least three finite correct and three finite wrong windows "
              "per feature. Means and medians remain in each feature's units; do not rank differently scaled features by their raw difference. "
              "All-correct and all-wrong groups cannot enter these contrasts; [coverage](feature_contrast_group_coverage.csv) makes that exclusion visible. "
              "The same physical inputs repeat across seeds. Associations and treatment gains do not prove a physiological mechanism. "
              "Stimulus disagreement is label-derived and must remain a diagnostic, never an inference feature.", "",
              "## Interpretation and next decision", "",
              "A consistent C2A–C1 gain supports testing the amplitude addition on different repetition folds. "
              "A consistent C2D–C2A gain supports the relative-change treatment beyond the amplitude control. "
              "If early recoveries are offset by middle/late damage or seed-dependent losses, revise that treatment before scaling it. "
              "If neither improves, this specific energy representation is unsupported; it does not imply EMG contains no useful information.", "",
              "These repetitions were inspected previously. This study is exploratory and within subject; three seeds are not repetition cross-validation. "
              "Do not tune on these outputs and then present the same test split as a new unbiased evaluation. No gate or 97% result is implied.", "",
              "[Window pairing audit](window_pairing_audit.csv), [fit inventory](fit_inventory.csv), "
              "[missing/incomplete fits](missing_or_incomplete_fits.csv) and [analysis issues](analysis_issues.csv) document verification."]
    for name in status["figures"]:
        lines += ["", f"![{Path(name).stem.replace('_', ' ')}]({name})"]
    if issues:
        lines += ["", f"Recorded issues: {status['warnings']} warnings and {status['errors']} errors. "
                  "Invalid or missing required artifacts prevent a complete-study claim."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path, required=True, help="One or more extracted result roots")
    parser.add_argument("--output", type=Path, required=True, help="Analysis destination")
    parser.add_argument("--expected-subjects", nargs="+", type=int, default=list(range(1, 23)))
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--expected-arms", nargs="+", default=["C1", "C2A", "C2D"])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero unless every expected fit passes checks")
    args = parser.parse_args()
    status = analyze(args.roots, args.output, sorted(set(args.expected_subjects)), sorted(set(args.expected_seeds)),
                     list(dict.fromkeys(args.expected_arms)), not args.no_plots)
    print(json.dumps(clean_json(status), indent=2, allow_nan=False))
    return 2 if args.require_complete and not status["study_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
