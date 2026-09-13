"""Paired C1/C2A/C2D energy study with frozen splits and auditable diagnostics."""
import gc
import gzip
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from baseline_support import (
    Config, DiagnosticTrainer,
    load_modalities, ModalityWindows, train_scaler, json_write,
    learning_report, probability_metrics, save_figure,
)
from energy_model import (
    ThreeBranchC1, ThreeBranchEnergy, energy_log_features,
    energy_model_preflight,
)
from reference_windows import REFERENCE_WINDOW_HASHES

WINDOW_KEYS = ['subject', 'gesture', 'native_repetition', 'window_start', 'window_end']
ARMS = ('C1', 'C2A', 'C2D')
STUDY_EXPECTED_PARAMETERS = {'C1': 551542, 'C2A': 555510, 'C2D': 555510}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(arm):
    if arm == 'C1':
        return ThreeBranchC1(dropout=Config.DROPOUT)
    if arm in ('C2A', 'C2D'):
        return ThreeBranchEnergy(energy_mode='amplitude' if arm == 'C2A' else 'dynamic', dropout=Config.DROPOUT)
    raise ValueError(arm)


def parameter_sha256(state, names):
    """Hash named learned tensors; exclude fitted scalers and BatchNorm buffers."""
    digest = hashlib.sha256()
    for name in sorted(names):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b'\0')
        digest.update(str(array.shape).encode() + b'\0' + str(array.dtype).encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def paired_initial_states(subject, seed_base, subject_folder):
    """Use identical base tensors in all arms and identical residual tensors in C2A/D."""
    actual_seed = seed_base + 1009 * subject
    seed_everything(actual_seed)
    base = make_model('C1')
    shared_names = tuple(name for name, _ in base.named_parameters())
    base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    seed_everything(actual_seed)
    energy = make_model('C2A')
    missing, unexpected = energy.load_state_dict(base_state, strict=False)
    assert not unexpected and missing, 'Only new energy tensors may be absent from the base state'
    assert all(key.startswith('energy_') for key in missing)
    energy_state = {key: value.detach().cpu().clone() for key, value in energy.state_dict().items()}
    seed_everything(actual_seed)
    dynamic = make_model('C2D')
    dynamic.load_state_dict(base_state, strict=False)
    dynamic.energy_projection.load_state_dict(energy.energy_projection.state_dict())
    dynamic_state = {key: value.detach().cpu().clone() for key, value in dynamic.state_dict().items()}
    assert dynamic.energy_mode == 'dynamic' and energy.energy_mode == 'amplitude'
    for key, value in base_state.items():
        assert torch.equal(value, energy_state[key]), f'Shared initial state mismatch: {key}'
    states = {'C1': base_state, 'C2A': energy_state, 'C2D': dynamic_state}
    names = {'C1': shared_names, 'C2A': tuple(name for name, _ in energy.named_parameters())}
    names['C2D'] = names['C2A']
    common_hash = parameter_sha256(base_state, shared_names)
    record = dict(subject=subject, seed_base=seed_base, seed=actual_seed,
                  initial_shared_parameter_sha256=common_hash,
                  shared_parameter_names=list(shared_names),
                  initial_parameter_sha256={arm: parameter_sha256(state, names[arm]) for arm, state in states.items()},
                  all_shared_tensors_including_buffers_equal=True,
                  energy_residual_initialization_identical=True,
                  training_rng_policy='Reset Python/NumPy/Torch/CUDA immediately before each trainer.fit; identical train DataLoader generator seeds')
    assert all(parameter_sha256(state, shared_names) == common_hash for state in states.values())
    assert record['initial_parameter_sha256']['C2A'] == record['initial_parameter_sha256']['C2D']
    json_write(subject_folder / 'paired_initialization.json', record)
    del base, energy, dynamic
    return states, shared_names, record


def window_hashes(datasets):
    return {split: hashlib.sha256(ds.meta[WINDOW_KEYS].to_csv(index=False).encode()).hexdigest()
            for split, ds in datasets.items()}


def preflight_models(root):
    energy_check = energy_model_preflight(Config.DEVICE)
    record = {'C1': energy_check['C1_preflight'], 'energy': energy_check}
    assert record['C1']['success'] and record['energy']['success']
    record['expected_parameters'] = STUDY_EXPECTED_PARAMETERS
    record['success'] = True
    json_write(root / 'preflight.json', record)
    gc.collect()
    torch.cuda.empty_cache()


def predict_arm(model, dataset, arm):
    model.eval()
    probabilities, embeddings, attention = [], [], []
    branches, energy_inputs = {}, []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False):
            result = model.forward_features(x.to(Config.DEVICE), retain_sequences=arm != 'C1')
            probabilities.append(result['logits'].softmax(1).cpu().numpy())
            embeddings.append(result['embedding'].cpu().numpy())
            attention.append(result['attention'].cpu().numpy())
            for name, value in result['branch_embeddings'].items():
                branches.setdefault(name, []).append(value.cpu().numpy())
            if arm != 'C1':
                energy_inputs.append(result['energy_inputs'].cpu().numpy())
    return dict(probabilities=np.concatenate(probabilities), embeddings=np.concatenate(embeddings),
                attention={'temporal': np.concatenate(attention)},
                branch_embeddings={k: np.concatenate(v) for k, v in branches.items()},
                energy_inputs=np.concatenate(energy_inputs) if energy_inputs else None)


def export_predictions(model, dataset, arm, subject, seed_base, split, directory):
    directory.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    result = predict_arm(model, dataset, arm)
    elapsed = time.perf_counter() - tick
    p = result['probabilities']
    assert p.shape == (len(dataset), 17) and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    frame = dataset.meta.copy()
    frame['arm'], frame['seed_base'] = arm, seed_base
    frame['prediction'], frame['confidence'] = p.argmax(1) + 1, p.max(1)
    frame['correct'] = frame['prediction'] == frame['gesture']
    frame['window_ms'], frame['stride_ms'] = 400, 100
    frame['endpoint_ms'] = (frame.window_end - frame.run_start) / 2
    frame['window_start_ms'] = (frame.window_start - frame.run_start) / 2
    frame['window_end_recording_ms'] = frame.window_end / 2
    frame['endpoint_bin_start_ms'] = (np.floor(frame.endpoint_ms / 100) * 100).astype(np.int64)
    frame['endpoint_bin_end_ms'] = frame.endpoint_bin_start_ms + 100
    frame['phase_fraction'] = ((frame.window_start + frame.window_end) / 2 - frame.run_start) / (frame.run_end - frame.run_start)
    frame.to_csv(directory / 'predictions.csv', index=False)
    extra = {} if result['energy_inputs'] is None else {'energy_inputs': result['energy_inputs']}
    np.savez_compressed(directory / 'probabilities_embeddings.npz', y_true=dataset.y,
                        probabilities=p, embeddings=result['embeddings'],
                        **{f'branch_{k}': v for k, v in result['branch_embeddings'].items()}, **extra)
    if result['attention']:
        np.savez_compressed(directory / 'attention.npz', **result['attention'])
    metrics, calibration = probability_metrics(dataset.y, p)
    metrics.update(arm=arm, subject=subject, seed_base=seed_base, split=split,
                   evaluation_seconds=elapsed)
    json_write(directory / 'metrics.json', metrics)
    pd.DataFrame(calibration).to_csv(directory / 'calibration.csv', index=False)
    errors = frame.groupby(['gesture', 'native_repetition']).agg(windows=('correct', 'size'), correct=('correct', 'sum')).reset_index()
    errors['wrong'] = errors.windows - errors.correct
    errors['error_percent'] = 100 * errors.wrong / errors.windows
    errors.to_csv(directory / 'gesture_errors.csv', index=False)
    timing = frame.groupby(['gesture', 'native_repetition', 'endpoint_bin_start_ms', 'endpoint_bin_end_ms']).agg(
        windows=('correct', 'size'), correct=('correct', 'sum')).reset_index()
    timing['wrong'] = timing.windows - timing.correct
    timing['error_percent'] = 100 * timing.wrong / timing.windows
    timing.to_csv(directory / 'endpoint_bin_errors.csv', index=False)
    if split == 'test':
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(errors.gesture, errors.correct, label='Correct')
        ax.bar(errors.gesture, errors.wrong, bottom=errors.correct, label='Wrong')
        ax.set(title=f'S{subject:02} / seed {seed_base} / {arm}', xlabel='Exercise B gesture', ylabel='Test windows', xticks=range(1, 18))
        ax.legend()
        save_figure(fig, directory / 'gesture_errors.png')
    return metrics, result


def export_input_features(dataset, subject_folder, seed_base):
    """Describe each test input and all seven frames; labels remain diagnostic only."""
    from scipy.io import loadmat
    output = dataset.meta[WINDOW_KEYS + ['phase', 'stimulus_disagreement_fraction']].copy()
    output['seed_base'] = seed_base
    recording = dataset.recording
    matches = list(Config.KAGGLE_INPUT.rglob(recording['metadata'].file_name.iloc[0]))
    assert len(matches) == 1
    restimulus = loadmat(matches[0], variable_names=['restimulus'])['restimulus'].reshape(-1)
    assert len(restimulus) == len(recording['stimulus'])
    offset = torch.as_tensor(dataset.mean[:12] / dataset.std[:12], device=Config.DEVICE)
    records = []
    trajectory_path = subject_folder / 'time_energy_trajectory.csv.gz'
    with gzip.open(trajectory_path, 'wt', encoding='utf-8', newline='') as trajectory_file:
        for start in range(0, len(dataset), 64):
            stop = min(start + 64, len(dataset))
            physical = np.stack([dataset.physical(i) for i in range(start, stop)]).astype(np.float64)
            emg = physical[:, :12]
            rms25 = np.sqrt(np.mean(emg[:, :, -50:] ** 2, axis=-1))
            rms100 = np.sqrt(np.mean(emg[:, :, -200:] ** 2, axis=-1))
            rms400 = np.sqrt(np.mean(emg ** 2, axis=-1))
            mav = np.mean(np.abs(emg), axis=-1)
            spectrum = np.abs(np.fft.rfft(emg * np.hanning(800)[None, None], axis=-1)) ** 2
            frequencies = np.fft.rfftfreq(800, 1 / 2000)
            band = (frequencies >= 20) & (frequencies <= 450)
            powers = spectrum[:, :, band]
            mean_frequency = (powers * frequencies[band]).sum(-1) / np.maximum(powers.sum(-1), 1e-30)
            values = {}
            for c in range(12):
                for name, array in [('rms25', rms25), ('rms100', rms100), ('rms400', rms400), ('mav400', mav), ('mean_frequency400', mean_frequency)]:
                    values[f'emg_{c+1:02}_{name}'] = array[:, c]
            for name, sl in [('acc', slice(12, 48)), ('gyro', slice(48, 84)), ('mag', slice(84, 120))]:
                x = physical[:, sl]
                mean, sd = x.mean(-1), x.std(-1)
                for c in range(36):
                    values[f'{name}_{c+1:02}_mean'] = mean[:, c]
                    values[f'{name}_{c+1:02}_std'] = sd[:, c]
            records.append(pd.DataFrame(values))
            frames = np.stack([emg[:, :, begin:begin + 200] for begin in range(0, 601, 100)], axis=2)
            frame_rms25 = np.sqrt(np.mean(frames[:, :, :, -50:] ** 2, axis=-1))
            frame_rms100 = np.sqrt(np.mean(frames ** 2, axis=-1))
            standardized = (physical.astype(np.float32) - dataset.mean) / dataset.std
            with torch.no_grad():
                log_features = energy_log_features(torch.as_tensor(standardized[:, :12], device=Config.DEVICE), offset, 'dynamic').cpu().numpy()
            meta = dataset.meta.iloc[start:stop]
            trajectory = meta.loc[meta.index.repeat(7), WINDOW_KEYS + ['run_start', 'run_end', 'phase']].reset_index(drop=True)
            trajectory['seed_base'] = seed_base
            trajectory['frame_index'] = np.tile(np.arange(1, 8), len(meta))
            trajectory['frame_start'] = trajectory.window_start + (trajectory.frame_index - 1) * 100
            trajectory['frame_end'] = trajectory.frame_start + 200
            trajectory['frame_end_relative_refined_ms'] = (trajectory.frame_end - trajectory.run_start) / 2
            trajectory['frame_end_recording_ms'] = trajectory.frame_end / 2
            label_rows = []
            for row in trajectory.itertuples(index=False):
                raw_labels = recording['stimulus'][row.frame_start:row.frame_end]
                refined = restimulus[row.frame_start:row.frame_end]
                label_rows.append(dict(stimulus_last_label=int(raw_labels[-1]), restimulus_last_label=int(refined[-1]),
                    stimulus_fraction_gesture=float(np.mean(raw_labels == row.gesture)),
                    restimulus_fraction_gesture=float(np.mean(refined == row.gesture)),
                    stimulus_restimulus_disagreement_fraction=float(np.mean(raw_labels != refined)),
                    stimulus_transition_count=int(np.count_nonzero(np.diff(raw_labels))),
                    restimulus_transition_count=int(np.count_nonzero(np.diff(refined)))))
            trajectory = pd.concat([trajectory, pd.DataFrame(label_rows)], axis=1)
            feature_values = {}
            for channel in range(12):
                for name, array in [('rms25_physical', frame_rms25[:, channel]),
                                    ('rms100_physical', frame_rms100[:, channel]),
                                    ('log_rms25_before_energy_scaler', log_features[:, channel]),
                                    ('log_rms25_minus_log_rms100_before_energy_scaler', log_features[:, channel + 12])]:
                    feature_values[f'emg_{channel+1:02}_{name}'] = array.reshape(-1)
            trajectory = pd.concat([trajectory, pd.DataFrame(feature_values)], axis=1)
            trajectory.to_csv(trajectory_file, index=False, header=start == 0)
    pd.concat([output.reset_index(drop=True), pd.concat(records, ignore_index=True)], axis=1).to_csv(subject_folder / 'input_features.csv', index=False)
    json_write(subject_folder / 'time_energy_trajectory_manifest.json', dict(
        split='test', windows=len(dataset), rows=len(dataset) * 7, frames_per_window=7,
        frame_ms=100, frame_hop_ms=50, frame_end_within_window_ms=list(range(100, 401, 50)),
        values='Physical filtered EMG RMS plus dimensionless log features computed after undoing training-input centering and dividing by training SD',
        normalization='Exported log features precede energy-feature standardization and sqrt(2) factor; actual model projection inputs are saved per arm/split in probabilities_embeddings.npz',
        labels='Stimulus/restimulus fields are diagnostic overlays, never model inputs or window-selection controls',
        causality='Existing per-repetition zero-phase filtering retained; this is an offline comparison, not physiological onset detection'))


def export_matched_cases(dataset, subject_folder, seed_base, predictions):
    """Retain early and confident failures with nearby-phase successes; no fitting uses cases."""
    reasons = {}
    def choose(index, reason):
        reasons.setdefault(int(index), set()).add(reason)
    y = dataset.y
    phase = ((dataset.meta.window_start + dataset.meta.window_end) / 2 - dataset.meta.run_start) / (dataset.meta.run_end - dataset.meta.run_start)
    for gesture in range(17):
        indices = np.flatnonzero(y == gesture)
        choose(indices[0], 'first_eligible_test_window')
    for arm, p in predictions.items():
        correct = p.argmax(1) == y
        for gesture in range(17):
            failed = np.flatnonzero((y == gesture) & ~correct)
            good = np.flatnonzero((y == gesture) & correct)
            if len(failed):
                index = int(failed[np.argmax(p[failed].max(1))])
                selections = [(index, f'{arm}_highest_confidence_failure')]
                early_failed = failed[phase.iloc[failed].to_numpy() < 1 / 3]
                if len(early_failed):
                    selections.append((int(early_failed[0]), f'{arm}_earliest_early_phase_failure'))
                for selected, reason in selections:
                    choose(selected, reason)
                    if len(good):
                        choose(good[np.argmin(np.abs(phase.iloc[good].to_numpy() - phase.iloc[selected]))], reason + '_matched_success')
    indices = np.asarray(sorted(reasons), dtype=np.int64)
    recording = dataset.recording
    # Load the original EMG only for selected cases; never retain whole raw IMU duplicates.
    from scipy.io import loadmat
    file_name = recording['metadata'].file_name.iloc[0]
    matches = list(Config.KAGGLE_INPUT.rglob(file_name))
    assert len(matches) == 1
    raw = loadmat(matches[0], variable_names=['emg', 'restimulus'])
    starts = dataset.starts[indices]
    context_indices = starts[:, None] + np.arange(-1000, 1800)[None, :]
    context_valid = (context_indices >= 0) & (context_indices < len(raw['emg']))
    clipped = np.clip(context_indices, 0, len(raw['emg']) - 1)
    context_emg = np.asarray(raw['emg'][clipped], dtype=np.float32).transpose(0, 2, 1)
    context_emg = np.where(context_valid[:, None, :], context_emg, np.nan)
    context_refined = np.where(context_valid, raw['restimulus'].reshape(-1)[clipped], -1)
    context_stimulus = np.where(context_valid, recording['stimulus'][clipped], -1)
    np.savez_compressed(subject_folder / 'waveform_cases.npz',
        window_indices=indices, signal_filtered=np.stack([dataset.physical(i) for i in indices]),
        raw_emg=np.stack([raw['emg'][s:s+800].T for s in starts]),
        restimulus=np.stack([raw['restimulus'][s:s+800].reshape(-1) for s in starts]),
        stimulus=np.stack([recording['stimulus'][s:s+800] for s in starts]),
        context_raw_emg=context_emg, context_restimulus=context_refined, context_stimulus=context_stimulus,
        context_recording_sample_index=context_indices, context_valid=context_valid,
        context_time_from_window_end_ms=np.arange(-1800, 1000) / 2,
        **{f'probability_{arm}': p[indices] for arm, p in predictions.items()})
    cases = dataset.meta.iloc[indices].copy()
    cases['seed_base'] = seed_base
    cases['window_index'] = indices
    cases['selection_reason'] = [';'.join(sorted(reasons[int(index)])) for index in indices]
    cases.to_csv(subject_folder / 'waveform_cases.csv', index=False)
    json_write(subject_folder / 'waveform_cases_manifest.json', dict(
        cases=len(indices), raw_context_before_window_ms=500, raw_context_after_window_ms=500,
        selection='First eligible window, highest-confidence failure and earliest early-phase failure per gesture/arm, with nearest-phase correct matches; union removes duplicates',
        interpretation='Selected diagnostic examples, not a random sample and not model-selection data; early phase is label-relative'))


def fit_embedding_probes(results, datasets, directory):
    """Training-fitted diagnostic probes of each frozen model's embeddings."""
    rows = []
    for name in ('waveform', 'spectral', 'inertial', 'fused'):
        def representation(split):
            return results[split]['embeddings'] if name == 'fused' else results[split]['branch_embeddings'][name]
        probe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        probe.fit(representation('train'), datasets['train'].y)
        saved = {}
        for split, dataset in datasets.items():
            p = probe.predict_proba(representation(split))
            assert np.array_equal(probe.classes_, np.arange(17))
            metric, _ = probability_metrics(dataset.y, p)
            rows.append(dict(branch=name, split=split, **metric))
            saved[f'{split}_probabilities'] = p.astype(np.float32)
        np.savez_compressed(directory / f'probe_{name}.npz', **saved)
    pd.DataFrame(rows).to_csv(directory / 'embedding_probe_metrics.csv', index=False)


def train_one(subject, seed_base, arm, datasets, hashes, folder, initial_state, shared_names, initialization):
    folder.mkdir(parents=True, exist_ok=True)
    actual_seed = seed_base + 1009 * subject
    seed_everything(actual_seed)
    model = make_model(arm).to(Config.DEVICE)
    model.load_state_dict(initial_state, strict=True)
    parameter_names = tuple(name for name, _ in model.named_parameters())
    initial_hash = parameter_sha256(model.state_dict(), parameter_names)
    shared_hash = parameter_sha256(model.state_dict(), shared_names)
    assert initial_hash == initialization['initial_parameter_sha256'][arm]
    assert shared_hash == initialization['initial_shared_parameter_sha256']
    assert sum(p.numel() for p in model.parameters()) == STUDY_EXPECTED_PARAMETERS[arm]
    if arm != 'C1':
        assert model.energy_mode == ('amplitude' if arm == 'C2A' else 'dynamic')
    manifest = dict(status='running', arm=arm, subject=subject, seed_base=seed_base, seed=actual_seed,
                    parameters=sum(p.numel() for p in model.parameters()), window_key_hashes=hashes,
                    initial_parameter_sha256=initial_hash,
                    initial_shared_parameter_sha256=shared_hash, window_ms=400, stride_ms=100,
                    input_modalities=['emg', 'acc', 'gyro', 'mag'],
                    energy_mode='none' if arm == 'C1' else model.energy_mode,
                    rng_reset_immediately_before_fit=True, train_batch_generator_seed=actual_seed,
                    **{f'{s}_windows': len(ds) for s, ds in datasets.items()})
    json_write(folder / 'fit_manifest.json', manifest)
    np.savez_compressed(folder / 'normalizer.npz', mean=datasets['train'].mean, std=datasets['train'].std,
                        constant_channels=datasets['train'].std[:, 0] <= 1.01e-8,
                        channel_names=datasets['train'].recording['channel_names'])
    before_bn = {name: value.detach().clone() for name, value in model.named_buffers()
                 if 'running_' in name or 'num_batches_tracked' in name}
    tick = time.perf_counter()
    stats = model.fit_spectral_scaler(DataLoader(datasets['train'], batch_size=64, shuffle=False), split='train')
    spectral_seconds = time.perf_counter() - tick
    assert stats['windows'] == len(datasets['train'])
    stats['source_window_key_sha256'] = hashes['train']
    json_write(folder / 'spectral_scaler.json', stats)
    np.savez_compressed(folder / 'spectral_scaler.npz', mean=model.spectral_mean.cpu().numpy(), std=model.spectral_std.cpu().numpy(),
                        constant=model.spectral_constant.cpu().numpy())
    energy_seconds = 0.0
    if arm != 'C1':
        tick = time.perf_counter()
        stats = model.fit_energy_scaler(DataLoader(datasets['train'], batch_size=64, shuffle=False),
            input_mean=datasets['train'].mean, input_std=datasets['train'].std, split='train')
        energy_seconds = time.perf_counter() - tick
        assert stats['windows'] == len(datasets['train']) and stats['energy_mode'] == model.energy_mode
        stats['source_window_key_sha256'] = hashes['train']
        json_write(folder / 'energy_scaler.json', stats)
        np.savez_compressed(folder / 'energy_scaler.npz', mean=model.energy_mean.cpu().numpy(), std=model.energy_std.cpu().numpy(),
            constant=model.energy_constant.cpu().numpy(), input_mean_over_std=model.energy_input_offset.cpu().numpy(),
            post_standardization_factor=np.asarray(2 ** -0.5), mode_code=model.energy_mode_code.cpu().numpy())
    for name, value in model.named_buffers():
        if name in before_bn:
            assert torch.equal(value, before_bn[name]), f'Scaler fitting changed BatchNorm: {name}'
    assert parameter_sha256(model.state_dict(), shared_names) == shared_hash
    assert parameter_sha256(model.state_dict(), parameter_names) == initial_hash
    model.eval()
    initial_x = torch.stack([datasets['train'][index][0] for index in range(min(2, len(datasets['train'])))]).to(Config.DEVICE)
    with torch.no_grad():
        initial_logits = model(initial_x).cpu().numpy()
    reference_path = folder.parent / 'initial_training_logits_reference.npz'
    if arm == 'C1':
        np.savez_compressed(reference_path, logits=initial_logits, training_window_indices=np.arange(len(initial_x)))
    else:
        with np.load(reference_path, allow_pickle=False) as reference:
            assert np.allclose(initial_logits, reference['logits'], rtol=1e-5, atol=1e-6), 'Zero energy residual must preserve initial C1 function'
    manifest['initial_training_logits_match_reference'] = True
    manifest['scaler_fit_preserved_batchnorm'] = True
    json_write(folder / 'fit_manifest.json', manifest)
    del initial_x, initial_logits, before_bn
    train = DataLoader(datasets['train'], batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(actual_seed))
    val = DataLoader(datasets['validation'], batch_size=128, shuffle=False)
    trainer = DiagnosticTrainer(model, folder / 'best_model.pt', 17)
    torch.cuda.reset_peak_memory_stats()
    seed_everything(actual_seed)
    trainer.fit(train, val)
    learning_report(trainer, folder)
    model.load_state_dict(torch.load(folder / 'best_model.pt', map_location=Config.DEVICE, weights_only=True))
    metrics, results = [], {}
    for split, dataset in datasets.items():
        metric, result = export_predictions(model, dataset, arm, subject, seed_base, split, folder / split)
        metrics.append(metric)
        results[split] = result
    pd.DataFrame(metrics).to_csv(folder / 'metrics.csv', index=False)
    json_write(folder / 'metrics.json', metrics)
    fit_embedding_probes(results, datasets, folder)
    manifest.update(status='complete', selected_epoch=trainer.best_epoch, epochs_run=len(trainer.epoch_rows),
                    optimizer_steps=len(train) * len(trainer.epoch_rows), training_seconds=trainer.train_wall,
                    spectral_fit_seconds=spectral_seconds, energy_fit_seconds=energy_seconds,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    json_write(folder / 'fit_manifest.json', manifest)
    json_write(folder / 'resource_metrics.json', {k: manifest[k] for k in ['parameters', 'training_seconds', 'spectral_fit_seconds', 'energy_fit_seconds', 'peak_cuda_memory_bytes']})
    test_probabilities = results['test']['probabilities']
    del model, trainer, train, val, results
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, test_probabilities


def run_energy_study():
    assert Config.DEVICE.type == 'cuda', 'Enable the Kaggle T4 GPU.'
    Config.SEED = 42  # Model seed changes must never change the repetition split.
    Config.MODALITIES = ('emg', 'acc', 'gyro', 'mag')
    Config.AUGMENT_TRAIN = Config.REFIT_ON_TRAIN_PLUS_VAL = False
    Config.EVALUATE_TEST = True
    seed_base = int(getattr(Config, 'SEED_BASE', 42))
    assert seed_base in (42, 43, 44)
    smoke = bool(getattr(Config, 'SMOKE', False))
    subjects = [1, 22] if smoke else list(range(1, 23))
    Config.MIN_EPOCHS, Config.MAX_EPOCHS, Config.PATIENCE = (1, 2, 2) if smoke else (20, 150, 15)
    Config.WIN_MS, Config.STEP_MS, Config.TRIM_MS = 400, 100, 100
    Config.WIN_SAMPLES, Config.STEP_SAMPLES, Config.TRIM_SAMPLES = 800, 200, 200
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Config.KAGGLE_WORKING / f'db7_energy_{seed_base}_{stamp}'
    root.mkdir(parents=True)
    manifest = dict(arms=list(ARMS), seed_base=seed_base, split_seed=42, subjects=subjects, smoke=smoke,
                    automation=getattr(Config, 'AUTOMATION', {}), labels=list(range(1, 18)), modalities=list(Config.MODALITIES),
                    window_ms=400, stride_ms=100, trim_ms=100, internal_frame_ms=100, internal_hop_ms=50,
                    protocol='within-subject 4 train / 1 validation / 1 test repetitions',
                    checkpoint='minimum validation loss; no refit', acc_centering=False, gating=False,
                    augmentation=False, test_interpretation='previously inspected exploratory test split',
                    versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version),
                    expected_parameters=STUDY_EXPECTED_PARAMETERS,
                    expected_fits=len(subjects) * len(ARMS), expected_full_study_fits=198,
                    paired_initialization='Identical C1 parameters/buffers in all arms; identical extra C2A/C2D learned parameters; mode codes remain distinct',
                    training_randomness='Reset RNG immediately before each trainer.fit; equal train DataLoader generator seeds',
                    energy_representation=dict(C1='none', C2A='duplicate log RMS25 amplitude control',
                        C2D='log RMS25 plus log RMS25 minus log RMS100',
                        physical_zero='undo input centering; preserve filtered EMG divided by training SD',
                        normalization='training-only feature moments, then divide by sqrt(2) symmetrically in C2A/C2D'),
                    diagnostics='100 ms endpoint bins relative refined labels, seven-frame energy trajectories, raw stimulus/restimulus case overlays; no physiological-onset claim')
    json_write(root / 'run_manifest.json', manifest)
    metrics, completed = [], []
    try:
        preflight_models(root)
        for subject in subjects:
            subject_folder = root / f'S{subject:02}' / f'seed_{seed_base}'
            subject_folder.mkdir(parents=True)
            recording = load_modalities(subject, subject_folder)
            datasets = {split: ModalityWindows(recording, split) for split in ('train', 'validation', 'test')}
            hashes = window_hashes(datasets)
            assert hashes == REFERENCE_WINDOW_HASHES[subject], f'S{subject}: historical window identities changed'
            mean, std = train_scaler(datasets['train'])
            assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
            for split, ds in datasets.items():
                ds.mean, ds.std = mean, std
                ds.meta.to_csv(subject_folder / f'{split}_window_manifest.csv', index=False)
            export_input_features(datasets['test'], subject_folder, seed_base)
            initial_states, shared_names, initialization = paired_initial_states(subject, seed_base, subject_folder)
            predictions = {}
            for arm in ARMS:
                print(f'\nS{subject:02} / seed {seed_base} / {arm}', flush=True)
                rows, p = train_one(subject, seed_base, arm, datasets, hashes, subject_folder / arm,
                                    initial_states[arm], shared_names, initialization)
                metrics.extend(rows)
                predictions[arm] = p
                completed.append(dict(subject=subject, seed_base=seed_base, arm=arm))
                pd.DataFrame(metrics).to_csv(root / 'all_subject_metrics.csv', index=False)
                json_write(root / 'progress.json', dict(completed_fits=completed))
            export_matched_cases(datasets['test'], subject_folder, seed_base, predictions)
            del datasets, recording, predictions, ds, initial_states, shared_names, initialization
            gc.collect()
            torch.cuda.empty_cache()
        pd.DataFrame(metrics).groupby(['arm', 'split'])[['accuracy', 'f1_macro']].mean().to_csv(root / 'mean_subject_metrics.csv')
        assert len(completed) == len(subjects) * len(ARMS)
        json_write(root / 'completion.json', dict(success=True, cnn_fits=len(completed), subjects=subjects,
                                                seed_base=seed_base, arms=list(ARMS)))
    except Exception:
        (root / 'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        archive = shutil.make_archive(str(root), 'zip', root)
        print('Outputs:', root, '\nArchive:', archive, flush=True)
    return root


if __name__ == '__main__':
    Config.SEED_BASE = 42
    Config.SMOKE = False
    run_energy_study()
