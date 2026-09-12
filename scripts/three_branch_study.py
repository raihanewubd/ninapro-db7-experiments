"""Train the frozen B0/C1 comparison and retain evidence for error analysis."""
import gc
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
    Config, OriginalMultiKernelAttention1DCNN, DiagnosticTrainer,
    load_modalities, ModalityWindows, train_scaler, json_write,
    learning_report, probability_metrics, predict_with_diagnostics, save_figure,
)
from three_branch_model import ThreeBranchC1, log_power, model_preflight as c1_preflight
from reference_windows import REFERENCE_WINDOW_HASHES

WINDOW_KEYS = ['subject', 'gesture', 'native_repetition', 'window_start', 'window_end']
ARMS = ('B0', 'C1')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(arm):
    if arm == 'B0':
        return OriginalMultiKernelAttention1DCNN(120, 17, Config.DROPOUT)
    if arm == 'C1':
        return ThreeBranchC1(dropout=Config.DROPOUT)
    raise ValueError(arm)


def window_hashes(datasets):
    return {split: hashlib.sha256(ds.meta[WINDOW_KEYS].to_csv(index=False).encode()).hexdigest()
            for split, ds in datasets.items()}


def preflight_models(root):
    record = {'C1': c1_preflight(Config.DEVICE)}
    seed_everything(123)
    model = make_model('B0').to(Config.DEVICE)
    assert sum(p.numel() for p in model.parameters()) == 552966
    x = torch.randn(2, 120, 800, device=Config.DEVICE)
    logits = model(x)
    F.cross_entropy(logits, torch.tensor([0, 16], device=Config.DEVICE)).backward()
    assert logits.shape == (2, 17) and torch.isfinite(logits).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    record['B0'] = {'success': True, 'parameters': 552966}
    record['success'] = True
    json_write(root / 'preflight.json', record)
    del model, x, logits
    gc.collect()
    torch.cuda.empty_cache()


def predict_arm(model, dataset, arm):
    if arm == 'B0':
        probability, embedding, attention = predict_with_diagnostics(model, dataset, retain_attention=True)
        return dict(probabilities=probability, embeddings=embedding, attention=attention,
                    branch_embeddings={})
    model.eval()
    probabilities, embeddings, attention = [], [], []
    branches = {'waveform': [], 'spectral': [], 'inertial': []}
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False):
            result = model.forward_features(x.to(Config.DEVICE))
            probabilities.append(result['logits'].softmax(1).cpu().numpy())
            embeddings.append(result['embedding'].cpu().numpy())
            attention.append(result['attention'].cpu().numpy())
            for name in branches:
                branches[name].append(result['branch_embeddings'][name].cpu().numpy())
    return dict(probabilities=np.concatenate(probabilities), embeddings=np.concatenate(embeddings),
                attention={'temporal': np.concatenate(attention)},
                branch_embeddings={k: np.concatenate(v) for k, v in branches.items()})


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
    frame['phase_fraction'] = ((frame.window_start + frame.window_end) / 2 - frame.run_start) / (frame.run_end - frame.run_start)
    frame.to_csv(directory / 'predictions.csv', index=False)
    np.savez_compressed(directory / 'probabilities_embeddings.npz', y_true=dataset.y,
                        probabilities=p, embeddings=result['embeddings'],
                        **{f'branch_{k}': v for k, v in result['branch_embeddings'].items()})
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
    if split == 'test':
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(errors.gesture, errors.correct, label='Correct')
        ax.bar(errors.gesture, errors.wrong, bottom=errors.correct, label='Wrong')
        ax.set(title=f'S{subject:02} / seed {seed_base} / {arm}', xlabel='Exercise B gesture', ylabel='Test windows', xticks=range(1, 18))
        ax.legend()
        save_figure(fig, directory / 'gesture_errors.png')
    return metrics, result


def export_input_features(dataset, subject_folder, seed_base):
    """Describe every test input once; predictions from either arm join by window key."""
    output = dataset.meta[WINDOW_KEYS + ['phase', 'stimulus_disagreement_fraction']].copy()
    output['seed_base'] = seed_base
    records = []
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
    pd.concat([output.reset_index(drop=True), pd.concat(records, ignore_index=True)], axis=1).to_csv(subject_folder / 'input_features.csv', index=False)


def export_matched_cases(dataset, subject_folder, seed_base, predictions):
    """Save one failure and a same-gesture, nearby-phase success for each arm/gesture."""
    chosen = set()
    y = dataset.y
    phase = ((dataset.meta.window_start + dataset.meta.window_end) / 2 - dataset.meta.run_start) / (dataset.meta.run_end - dataset.meta.run_start)
    for arm, p in predictions.items():
        correct = p.argmax(1) == y
        for gesture in range(17):
            failed = np.flatnonzero((y == gesture) & ~correct)
            good = np.flatnonzero((y == gesture) & correct)
            if len(failed):
                index = int(failed[np.argmax(p[failed].max(1))])
                chosen.add(index)
                if len(good):
                    chosen.add(int(good[np.argmin(np.abs(phase.iloc[good].to_numpy() - phase.iloc[index]))]))
    if not chosen:
        return
    indices = np.asarray(sorted(chosen), dtype=np.int64)
    recording = dataset.recording
    # Load the original EMG only for selected cases; never retain whole raw IMU duplicates.
    from scipy.io import loadmat
    file_name = recording['metadata'].file_name.iloc[0]
    matches = list(Config.KAGGLE_INPUT.rglob(file_name))
    assert len(matches) == 1
    raw = loadmat(matches[0], variable_names=['emg', 'restimulus'])
    starts = dataset.starts[indices]
    np.savez_compressed(subject_folder / 'waveform_cases.npz',
        window_indices=indices, signal_filtered=np.stack([dataset.physical(i) for i in indices]),
        raw_emg=np.stack([raw['emg'][s:s+800].T for s in starts]),
        restimulus=np.stack([raw['restimulus'][s:s+800].reshape(-1) for s in starts]),
        stimulus=np.stack([recording['stimulus'][s:s+800] for s in starts]),
        **{f'probability_{arm}': p[indices] for arm, p in predictions.items()})
    cases = dataset.meta.iloc[indices].copy()
    cases['seed_base'] = seed_base
    cases['window_index'] = indices
    cases.to_csv(subject_folder / 'waveform_cases.csv', index=False)


def fit_embedding_probes(results, datasets, directory):
    """Training-fitted diagnostic probes of frozen C1 embeddings, not new sensor CNNs."""
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


def train_one(subject, seed_base, arm, datasets, hashes, folder):
    folder.mkdir(parents=True, exist_ok=True)
    actual_seed = seed_base + 1009 * subject
    seed_everything(actual_seed)
    model = make_model(arm).to(Config.DEVICE)
    initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    manifest = dict(status='running', arm=arm, subject=subject, seed_base=seed_base, seed=actual_seed,
                    parameters=sum(p.numel() for p in model.parameters()), window_key_hashes=hashes,
                    initial_parameter_sha256=initial_hash,
                    **{f'{s}_windows': len(ds) for s, ds in datasets.items()})
    json_write(folder / 'fit_manifest.json', manifest)
    np.savez_compressed(folder / 'normalizer.npz', mean=datasets['train'].mean, std=datasets['train'].std,
                        constant_channels=datasets['train'].std[:, 0] <= 1.01e-8,
                        channel_names=datasets['train'].recording['channel_names'])
    spectral_seconds = 0.0
    if arm == 'C1':
        tick = time.perf_counter()
        stats = model.fit_spectral_scaler(DataLoader(datasets['train'], batch_size=64, shuffle=False), split='train')
        spectral_seconds = time.perf_counter() - tick
        json_write(folder / 'spectral_scaler.json', stats)
        np.savez_compressed(folder / 'spectral_scaler.npz', mean=model.spectral_mean.cpu().numpy(), std=model.spectral_std.cpu().numpy(),
                            constant=model.spectral_constant.cpu().numpy())
    train = DataLoader(datasets['train'], batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(actual_seed))
    val = DataLoader(datasets['validation'], batch_size=128, shuffle=False)
    trainer = DiagnosticTrainer(model, folder / 'best_model.pt', 17)
    torch.cuda.reset_peak_memory_stats()
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
    if arm == 'C1':
        fit_embedding_probes(results, datasets, folder)
    manifest.update(status='complete', selected_epoch=trainer.best_epoch, epochs_run=len(trainer.epoch_rows),
                    optimizer_steps=len(train) * len(trainer.epoch_rows), training_seconds=trainer.train_wall,
                    spectral_fit_seconds=spectral_seconds, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    json_write(folder / 'fit_manifest.json', manifest)
    json_write(folder / 'resource_metrics.json', {k: manifest[k] for k in ['parameters', 'training_seconds', 'spectral_fit_seconds', 'peak_cuda_memory_bytes']})
    test_probabilities = results['test']['probabilities']
    del model, trainer, train, val, results
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, test_probabilities


def run_three_branch_study():
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
    root = Config.KAGGLE_WORKING / f'db7_three_branch_{seed_base}_{stamp}'
    root.mkdir(parents=True)
    manifest = dict(arms=list(ARMS), seed_base=seed_base, split_seed=42, subjects=subjects, smoke=smoke,
                    automation=getattr(Config, 'AUTOMATION', {}), labels=list(range(1, 18)), modalities=list(Config.MODALITIES),
                    window_ms=400, stride_ms=100, trim_ms=100, internal_frame_ms=100, internal_hop_ms=50,
                    protocol='within-subject 4 train / 1 validation / 1 test repetitions',
                    checkpoint='minimum validation loss; no refit', acc_centering=False, gating=False,
                    augmentation=False, test_interpretation='previously inspected exploratory test split',
                    versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version),
                    expected_parameters={'B0': 552966, 'C1': 551542})
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
            predictions = {}
            for arm in ARMS:
                print(f'\nS{subject:02} / seed {seed_base} / {arm}', flush=True)
                rows, p = train_one(subject, seed_base, arm, datasets, hashes, subject_folder / arm)
                metrics.extend(rows)
                predictions[arm] = p
                completed.append(dict(subject=subject, seed_base=seed_base, arm=arm))
                pd.DataFrame(metrics).to_csv(root / 'all_subject_metrics.csv', index=False)
                json_write(root / 'progress.json', dict(completed_fits=completed))
            export_matched_cases(datasets['test'], subject_folder, seed_base, predictions)
            del datasets, recording, predictions, ds
            gc.collect()
            torch.cuda.empty_cache()
        pd.DataFrame(metrics).groupby(['arm', 'split'])[['accuracy', 'f1_macro']].mean().to_csv(root / 'mean_subject_metrics.csv')
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
    run_three_branch_study()
