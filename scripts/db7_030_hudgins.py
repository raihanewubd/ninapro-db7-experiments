"""DB7-030: Hudgins TD4 complementarity against frozen DB7-016 SI/I outputs.

The neural predictions are inputs, never re-fitted. Every TD4 fitting or
selection decision uses repetitions 1/3/4/6; repetitions 2/5 are evaluated
only after those choices are fixed. Overlapping windows stay in their trial.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from scipy.stats import rankdata

FS = 2000
WINDOW = 400
TRAIN_STEP = 100  # 50 ms: reduce duplicated training evidence
TEST_STEP = 20  # 10 ms: match the frozen SI/I prediction grid
TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)
FAMILIES = ("MAV", "WL", "ZC", "SSC")
THRESHOLD_RATIOS = (0.005, 0.01, 0.02)
KEY = ["subject", "gesture", "native_repetition", "window_start", "window_end"]


def find_root() -> Path:
    for parent in (Path("/kaggle/input"), Path("D:/NinaProDB7")):
        if parent.exists():
            found = sorted(parent.rglob("S1_E1_A1.mat"))
            if found:
                return found[0].parent.parent
    raise FileNotFoundError("NinaPro DB7 E1 recordings are not attached")


def load_subject(root: Path, subject: int):
    candidates = list(root.rglob(f"S{subject}_E1_A1.mat"))
    if len(candidates) != 1:
        raise ValueError(f"S{subject}: expected one E1 file, found {len(candidates)}")
    data = loadmat(candidates[0], variable_names=["emg", "restimulus", "rerepetition"])
    emg = np.asarray(data["emg"], dtype=np.float32)
    if emg.shape[1] != 12 and emg.shape[0] == 12:
        emg = emg.T
    labels = np.asarray(data["restimulus"]).ravel().astype(int)
    repetitions = np.asarray(data["rerepetition"]).ravel().astype(int)
    if emg.shape[1] != 12 or len(emg) != len(labels) or len(emg) != len(repetitions):
        raise ValueError(f"S{subject}: invalid EMG/annotation alignment")
    if set(np.unique(labels)) != set(range(18)):
        raise ValueError(f"S{subject}: E1 labels 0..17 are not present")
    sos = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")
    b, a = iirnotch(50, 30, fs=FS)
    edges = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    runs = []
    for start, end in zip(edges[:-1], edges[1:]):
        gesture = int(labels[start])
        if gesture == 0:
            continue
        unique_rep = np.unique(repetitions[start:end])
        if len(unique_rep) != 1 or unique_rep[0] not in (*TRAIN_REPS, *TEST_REPS):
            raise ValueError(f"S{subject} gesture {gesture}: invalid repetition")
        filtered = sosfiltfilt(sos, emg[start:end], axis=0)
        filtered = filtfilt(b, a, filtered, axis=0).astype(np.float32)
        runs.append((gesture, int(unique_rep[0]), int(start), filtered))
    counts = pd.Series([(g, r) for g, r, _, _ in runs]).value_counts()
    if len(counts) != 102 or not (counts == 1).all():
        raise ValueError(f"S{subject}: expected 17 gestures x 6 repetitions")
    digest = hashlib.sha256(emg.tobytes()).hexdigest()
    return runs, {"file": str(candidates[0]), "raw_emg_sha256": digest}


def threshold_scale(runs, fitting_repetitions=TRAIN_REPS):
    """Estimate channel amplitudes using only the repetitions used for fitting."""
    train_signal = np.concatenate([np.abs(x) for _, r, _, x in runs
                                   if r in fitting_repetitions])
    return np.maximum(np.percentile(train_signal, 95, axis=0), 1e-12)


def extract(runs, subject, split, threshold):
    features, rows = [], []
    step = TRAIN_STEP if split == "train" else TEST_STEP
    reps = TRAIN_REPS if split == "train" else TEST_REPS
    for gesture, rep, absolute_start, signal in runs:
        if rep not in reps or len(signal) < WINDOW:
            continue
        # Shape: windows x channels x 400 samples. Each run is handled alone.
        windows = np.lib.stride_tricks.sliding_window_view(signal, WINDOW, axis=0)[::step]
        mav = np.abs(windows).mean(axis=-1)
        difference = np.diff(windows, axis=-1)
        wl = np.abs(difference).sum(axis=-1)
        zc = ((windows[..., :-1] * windows[..., 1:] < 0) &
              (np.abs(difference) >= threshold[None, :, None])).sum(axis=-1)
        ssc = ((difference[..., :-1] * difference[..., 1:] < 0) &
               (np.maximum(np.abs(difference[..., :-1]),
                           np.abs(difference[..., 1:])) >= threshold[None, :, None])).sum(axis=-1)
        features.append(np.concatenate([mav, wl, zc, ssc], axis=1).astype(np.float32))
        offsets = np.arange(len(windows)) * step
        rows.extend((subject, gesture, rep, absolute_start + int(k), absolute_start + int(k) + WINDOW)
                    for k in offsets)
    frame = pd.DataFrame(rows, columns=KEY)
    if frame.duplicated(KEY).any():
        raise AssertionError("Duplicate feature windows")
    return np.concatenate(features), frame


def columns(mask):
    return np.concatenate([np.arange(12 * i, 12 * (i + 1)) for i in range(4) if mask & (1 << i)])


def fit_lda(x, y):
    """Train-only robust scaling and regularized pooled-covariance LDA."""
    x = np.asarray(x, dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    scale = np.maximum(scale, np.maximum(np.max(np.abs(x), axis=0) * 1e-6, 1e-9))
    z = (x - center) / scale
    classes = np.arange(1, 18)
    counts = np.bincount(y, minlength=18)[1:]
    if np.any(counts == 0):
        raise ValueError("LDA training is missing a gesture")
    means = np.stack([z[y == g].mean(axis=0) for g in classes])
    residual = z - means[y - 1]
    covariance = residual.T @ residual / max(len(z) - len(classes), 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.9 * covariance + 0.1 * diagonal
    covariance.flat[::len(covariance) + 1] += 1e-5
    weights = np.linalg.solve(covariance, means.T)
    bias = -0.5 * np.sum(means * weights.T, axis=1) + np.log(counts / counts.sum())
    return center, scale, weights, bias


def predict(model, x):
    center, scale, weights, bias = model
    score = ((np.asarray(x, dtype=np.float64) - center) / scale) @ weights + bias
    score -= score.max(axis=1, keepdims=True)
    prob = np.exp(score)
    prob /= prob.sum(axis=1, keepdims=True)
    return prob.argmax(axis=1).astype(np.int16) + 1, prob


def cv_score(fold_features, frame, mask):
    """Each fold has features extracted with its own fitting-only threshold."""
    y = frame.gesture.to_numpy(int)
    rep = frame.native_repetition.to_numpy(int)
    ix = columns(mask)
    scores = []
    for held in TRAIN_REPS:
        train, val = rep != held, rep == held
        x = fold_features[held]
        pred, _ = predict(fit_lda(x[train][:, ix], y[train]), x[val][:, ix])
        scores.append(float(np.mean(pred == y[val])))
    return float(np.mean(scores))


def check_trace_coverage(subject_trace, test_meta, subject):
    """Require exactly one prediction per raw test window for each saved seed."""
    if set(subject_trace.seed.unique()) != {42, 43, 44}:
        raise AssertionError(f"S{subject}: expected exactly seeds 42, 43 and 44")
    expected = pd.MultiIndex.from_frame(test_meta[KEY])
    for seed, group in subject_trace.groupby("seed"):
        actual = pd.MultiIndex.from_frame(group[KEY])
        if actual.has_duplicates or len(actual) != len(expected):
            raise AssertionError(f"S{subject} seed {seed}: duplicate/missing test windows")
        if len(expected.difference(actual)) or len(actual.difference(expected)):
            raise AssertionError(f"S{subject} seed {seed}: raw and saved test keys differ")


def subset_complementarity(joined, prediction, margin):
    """Count useful TD4 evidence and harmful switches for one subject and seed."""
    truth = joined.gesture.to_numpy(int)
    si = joined.si_pred.to_numpy(int)
    inertial = joined.i_pred.to_numpy(int)
    si_correct, i_correct = si == truth, inertial == truth
    good = prediction == truth
    shared = ~si_correct & ~i_correct
    recovery, harm = ~si_correct & i_correct, si_correct & ~i_correct
    switch = (si != inertial) & (margin > 0)
    recovered, harmed = int((switch & recovery).sum()), int((switch & harm).sum())
    return dict(windows=len(truth), td_correct=int(good.sum()),
        td_accuracy=float(good.mean()), si_correct=int(si_correct.sum()),
        si_accuracy=float(si_correct.mean()), shared_wrong=int(shared.sum()),
        shared_recovered=int((shared & good).sum()),
        si_wrong_td_correct=int((~si_correct & good).sum()),
        si_correct_td_wrong=int((si_correct & ~good).sum()),
        recovery_opportunities=int(recovery.sum()), harm_opportunities=int(harm.sum()),
        td_correct_on_recovery=int((recovery & good).sum()),
        td_correct_on_harm=int((harm & good).sum()),
        switches=int(switch.sum()), switch_recovered=recovered, switch_harmed=harmed,
        switch_net_corrected=recovered - harmed,
        switch_accuracy=float((si_correct.sum() + recovered - harmed) / len(truth)),
        si_i_td_oracle_correct=int((si_correct | i_correct | good).sum()))


def subject_study(root, subject, trace, out):
    runs, source = load_subject(root, subject)
    scale = threshold_scale(runs)
    # A validation repetition must not affect the ZC/SSC amplitude thresholds.
    fold_scales = {held: threshold_scale(runs, tuple(r for r in TRAIN_REPS if r != held))
                   for held in TRAIN_REPS}
    threshold_cv = []
    training = {}
    cv_features = {}
    for ratio in THRESHOLD_RATIOS:
        x, meta = extract(runs, subject, "train", scale * ratio)
        training[ratio] = x
        cv_features[ratio] = {}
        for held in TRAIN_REPS:
            fold_x, fold_meta = extract(runs, subject, "train", fold_scales[held] * ratio)
            if not fold_meta.equals(meta):
                raise AssertionError("Fold feature windows changed with the noise threshold")
            cv_features[ratio][held] = fold_x
        score = cv_score(cv_features[ratio], meta, 15)
        threshold_cv.append(dict(subject=subject, ratio=ratio, td4_cv_accuracy=score))
    selected_ratio = sorted(THRESHOLD_RATIOS,
                            key=lambda v: (-next(r["td4_cv_accuracy"] for r in threshold_cv if r["ratio"] == v),
                                           abs(v - 0.01)))[0]
    x_train = training[selected_ratio]
    selected_cv_features = cv_features[selected_ratio]
    del training
    del cv_features
    x_test, test_meta = extract(runs, subject, "test", scale * selected_ratio)
    y_train = meta.gesture.to_numpy(int)
    y_test = test_meta.gesture.to_numpy(int)
    subset_cv = []
    for mask in range(1, 16):
        subset_cv.append(dict(subject=subject, mask=mask,
                              families="+".join(FAMILIES[i] for i in range(4) if mask & (1 << i)),
                              cv_accuracy=cv_score(selected_cv_features, meta, mask)))
    del selected_cv_features
    chosen = sorted(subset_cv, key=lambda r: (-r["cv_accuracy"], int(r["mask"]).bit_count(), r["mask"]))[0]["mask"]
    test_rows, predictions = [], {}
    for row in subset_cv:
        mask = row["mask"]
        ix = columns(mask)
        pred, prob = predict(fit_lda(x_train[:, ix], y_train), x_test[:, ix])
        predictions[mask] = (pred, prob)
        test_rows.append({**row, "test_accuracy": float(np.mean(pred == y_test)),
                          "correct": int(np.sum(pred == y_test)), "windows": len(y_test),
                          "cv_selected": mask == chosen, "td4_full": mask == 15,
                          "selected_threshold_ratio": selected_ratio})
    subject_trace = trace[trace.subject == subject].copy()
    check_trace_coverage(subject_trace, test_meta, subject)
    joined = subject_trace.merge(test_meta.assign(feature_row=np.arange(len(test_meta))),
                                 on=KEY, how="left", validate="many_to_one")
    if joined.feature_row.isna().any() or len(joined) != len(subject_trace):
        raise AssertionError(f"S{subject}: saved neural/test feature window mismatch")
    if not np.array_equal(joined.gesture.to_numpy(), y_test[joined.feature_row.to_numpy(int)]):
        raise AssertionError("Label mismatch after joining predictions")
    combination_rows = []
    indices = joined.feature_row.to_numpy(int)
    si, inertial = joined.si_pred.to_numpy(int), joined.i_pred.to_numpy(int)
    for row in subset_cv:
        mask = row["mask"]
        pred, prob = predictions[mask]
        aligned_pred = pred[indices]
        margin = prob[indices, inertial - 1] - prob[indices, si - 1]
        for seed, positions in joined.groupby("seed").indices.items():
            positions = np.asarray(positions)
            combination_rows.append(dict(subject=subject, seed=int(seed), mask=mask,
                families=row["families"], cv_selected=mask == chosen,
                **subset_complementarity(joined.iloc[positions], aligned_pred[positions],
                                         margin[positions])))
    for name, mask in (("selected", chosen), ("TD4", 15)):
        pred, prob = predictions[mask]
        indices = joined.feature_row.to_numpy(int)
        si = joined.si_pred.to_numpy(int)
        inertial = joined.i_pred.to_numpy(int)
        joined[f"{name}_pred"] = pred[indices]
        joined[f"{name}_correct"] = pred[indices] == joined.gesture.to_numpy(int)
        # Candidate support is diagnostic; do not treat raw LDA probabilities as calibrated.
        joined[f"{name}_candidate_margin"] = (prob[indices, inertial - 1] -
                                               prob[indices, si - 1]).astype(np.float32)
    joined.drop(columns="feature_row").to_csv(out / f"S{subject:02}_window_predictions.csv.gz", index=False)
    (out / f"S{subject:02}_source.json").write_text(json.dumps({**source, "subject": subject,
        "threshold_ratio": selected_ratio, "selected_subset": chosen,
        "threshold_scale_final": scale.tolist(),
        "threshold_scale_by_cv_held_repetition": {str(r): v.tolist() for r, v in fold_scales.items()},
        "train_windows_50ms": len(x_train), "test_windows_10ms": len(x_test)}, indent=2))
    print(f"S{subject:02}: train {len(x_train)}, test {len(x_test)}, "
          f"TD4 {test_rows[14]['test_accuracy']:.4f}, chosen {chosen:04b}", flush=True)
    return threshold_cv, subset_cv, test_rows, combination_rows


def summary(out, subjects):
    parts = [pd.read_csv(out / f"S{s:02}_window_predictions.csv.gz") for s in subjects]
    data = pd.concat(parts, ignore_index=True)
    if data.duplicated(KEY + ["seed"]).any():
        raise AssertionError("Duplicate test predictions")
    expected = len(data)
    target = data.gesture.to_numpy()
    correct = {"SI": data.si_pred.to_numpy() == target, "I": data.i_pred.to_numpy() == target,
               "TD4": data.TD4_pred.to_numpy() == target,
               "selected": data.selected_pred.to_numpy() == target}
    si_wrong = ~correct["SI"]
    shared = si_wrong & ~correct["I"]
    recovery = si_wrong & correct["I"]
    prevent_harm = correct["SI"] & ~correct["I"]
    records = []
    gate_rows = []
    for method in ("TD4", "selected"):
        good = correct[method]
        records.append(dict(method=method, windows=expected, correct=int(good.sum()),
            accuracy=float(good.mean()), si_errors=int(si_wrong.sum()),
            si_i_both_wrong=int(shared.sum()), shared_errors_recovered=int((shared & good).sum()),
            shared_errors_recovery_pct=float(100 * (shared & good).sum() / shared.sum()),
            si_wrong_i_correct_support=int((recovery & good).sum()),
            si_correct_i_wrong_td_correct=int((prevent_harm & good).sum()),
            si_correct_td_wrong=int((correct["SI"] & ~good).sum()),
            si_i_td_oracle_accuracy=float((correct["SI"] | correct["I"] | good).mean()),
            si_td_oracle_accuracy=float((correct["SI"] | good).mean())))
        margin = data[f"{method}_candidate_margin"].to_numpy(float)
        switch = (data.si_pred.to_numpy() != data.i_pred.to_numpy()) & (margin > 0)
        recovered = int((switch & recovery).sum())
        harmed = int((switch & prevent_harm).sum())
        directional = recovery | prevent_harm
        ranks = rankdata(margin[directional], method="average")
        positive = recovery[directional]
        negative_count = int((~positive).sum())
        auc = ((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
               / (positive.sum() * negative_count))
        gate_rows.append(dict(method=method, disagreement_windows=int((data.si_pred != data.i_pred).sum()),
            recovery_opportunities=int(recovery.sum()), harm_opportunities=int(prevent_harm.sum()),
            pair_margin_auc_recovery_vs_harm=float(auc),
            fixed_zero_margin_switches=int(switch.sum()), recovered=recovered, harmed=harmed,
            net_corrected=recovered - harmed,
            fixed_zero_margin_gate_accuracy=float((correct["SI"].sum() + recovered - harmed) / expected),
            note="Fixed threshold zero; diagnostic only, no BRB fitting or threshold tuning"))
    pd.DataFrame(records).to_csv(out / "complementarity_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(out / "candidate_preference_diagnostic.csv", index=False)
    for grouping, filename in [(["subject", "seed"], "subject_seed_complementarity.csv"),
                               (["subject", "gesture"], "subject_gesture_complementarity.csv"),
                               (["gesture"], "gesture_complementarity.csv"),
                               (["phase"], "phase_complementarity.csv")]:
        rows = []
        for key, indices in data.groupby(grouping, sort=True).indices.items():
            ix = np.asarray(indices)
            values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(grouping, values))
            row.update(windows=len(ix), SI_accuracy=float(correct["SI"][ix].mean()),
                       I_accuracy=float(correct["I"][ix].mean()))
            for method in ("TD4", "selected"):
                row[f"{method}_accuracy"] = float(correct[method][ix].mean())
                row[f"{method}_shared_recovered"] = int((shared & correct[method])[ix].sum())
                row[f"{method}_si_error_recovered"] = int((si_wrong & correct[method])[ix].sum())
                row[f"{method}_si_correct_harmed"] = int((correct["SI"] & ~correct[method])[ix].sum())
            row["SI_wrong"] = int(si_wrong[ix].sum())
            row["shared_wrong"] = int(shared[ix].sum())
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    selected = pd.read_csv(out / "subset_test_metrics.csv")
    aggregate = selected.groupby(["mask", "families"]).agg(mean_subject_test_accuracy=("test_accuracy", "mean"),
        mean_subject_cv_accuracy=("cv_accuracy", "mean"), subjects_selected=("cv_selected", "sum")).reset_index()
    aggregate.sort_values("mean_subject_cv_accuracy", ascending=False).to_csv(out / "feature_combination_summary.csv", index=False)
    combinations = pd.read_csv(out / "subset_subject_seed_complementarity.csv")
    count_fields = ["windows", "td_correct", "si_correct", "shared_wrong", "shared_recovered",
        "si_wrong_td_correct", "si_correct_td_wrong", "recovery_opportunities", "harm_opportunities",
        "td_correct_on_recovery", "td_correct_on_harm", "switches", "switch_recovered", "switch_harmed",
        "switch_net_corrected", "si_i_td_oracle_correct"]
    rows = []
    for (mask, families), group in combinations.groupby(["mask", "families"]):
        row = dict(mask=int(mask), families=families,
                   **{field: int(group[field].sum()) for field in count_fields})
        row.update(mean_subject_td_accuracy=float(group.td_accuracy.mean()),
            mean_subject_si_accuracy=float(group.si_accuracy.mean()),
            mean_subject_switch_accuracy=float(group.switch_accuracy.mean()),
            pooled_td_accuracy=row["td_correct"] / row["windows"],
            pooled_switch_accuracy=(row["si_correct"] + row["switch_net_corrected"]) / row["windows"],
            shared_recovery_pct=100 * row["shared_recovered"] / max(row["shared_wrong"], 1),
            pooled_si_i_td_oracle_accuracy=row["si_i_td_oracle_correct"] / row["windows"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "subset_complementarity_summary.csv", index=False)
    completion = dict(success=True, subjects=subjects, seeds=[42, 43, 44],
        test_window_seed_evaluations=expected, neural_fits=0,
        lda_final_fits=15 * len(subjects), lda_cv_fits=(3 + 15) * 4 * len(subjects),
        exact_seed_window_coverage_verified=True, cv_threshold_scale_fitting_repetitions_only=True,
        protocol="DB7-016 200ms/10ms test; 50ms TD4 training grid; repetition-grouped train-only fourfold selection")
    (out / "completion.json").write_text(json.dumps(completion, indent=2))
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    args = parser.parse_args()
    root = args.input or find_root()
    args.out.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(args.trace)
    needed = KEY + ["seed", "si_pred", "i_pred", "phase"]
    if any(v not in trace for v in needed):
        raise ValueError("Saved SI/I trace is missing required columns")
    if trace.duplicated(KEY + ["seed"]).any():
        raise ValueError("Saved SI/I trace has duplicate windows")
    source = dict(raw_root=str(root), trace_file=str(args.trace),
                  trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  subjects=args.subjects, train_repetitions=TRAIN_REPS,
                  test_repetitions=TEST_REPS, window_samples=WINDOW,
                  train_step_samples=TRAIN_STEP, test_step_samples=TEST_STEP,
                  feature_families=FAMILIES, threshold_ratios=THRESHOLD_RATIOS,
                  lda_covariance_shrinkage=0.1,
                  cv_threshold_scale="95th percentile fitted separately on three fitting repetitions",
                  final_threshold_scale="95th percentile fitted on repetitions 1/3/4/6",
                  testing_note="Existing inspected DB7-016 test split; exploratory, no independent confirmation")
    (args.out / "PROTOCOL.json").write_text(json.dumps(source, indent=2))
    threshold_rows, cv_rows, test_rows, combination_rows = [], [], [], []
    for s in args.subjects:
        a, b, c, d = subject_study(root, s, trace, args.out)
        threshold_rows.extend(a)
        cv_rows.extend(b)
        test_rows.extend(c)
        combination_rows.extend(d)
        pd.DataFrame(threshold_rows).to_csv(args.out / "threshold_cv.csv", index=False)
        pd.DataFrame(cv_rows).to_csv(args.out / "subset_cv.csv", index=False)
        pd.DataFrame(test_rows).to_csv(args.out / "subset_test_metrics.csv", index=False)
        pd.DataFrame(combination_rows).to_csv(args.out / "subset_subject_seed_complementarity.csv", index=False)
    print(json.dumps(summary(args.out, args.subjects), indent=2), flush=True)


if __name__ == "__main__":
    main()
