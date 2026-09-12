"""Unmodified model/data/trainer definitions from the modality study."""
import os, gc, time, json, warnings, random, re, shutil

from pathlib import Path

from copy import deepcopy

from math import gcd

import numpy as np

import pandas as pd

from scipy import io

from scipy.signal import resample_poly, butter, sosfiltfilt, iirnotch, filtfilt

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from tqdm.auto import tqdm

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

import matplotlib

import hashlib

def _find_kaggle_input() -> Path:
    base = Path('/kaggle/input')
    if not base.exists():
        return Path('/kaggle/input/ninapro-db7/Dataset')

    def _has_subjects(p):
        return p.is_dir() and any((c.is_dir() and c.name.lower().startswith('subject_') for c in p.iterdir()))

    def _search(root, depth=0):
        if depth > 5:
            return None
        if _has_subjects(root):
            return root
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    result = _search(child, depth + 1)
                    if result is not None:
                        return result
        except PermissionError:
            pass
        return None
    result = _search(base)
    return result if result else Path('/kaggle/input/ninapro-db7/Dataset')

class Config:
    KAGGLE_INPUT = _find_kaggle_input()
    KAGGLE_WORKING = Path('/kaggle/working') if Path('/kaggle').exists() else Path.cwd() / 'eda_working'
    EXERCISE_IDS = (1,)
    GESTURE_MIN, GESTURE_MAX, N_CLASSES = (1, 17, 17)
    SUBJECTS = list(range(1, 23))
    RUN_SUBJECTS = SUBJECTS.copy()
    INTACT_SUBJECTS = list(range(1, 21))
    AMPUTEE_SUBJECTS = [21, 22]
    SEED = 42
    MODEL_SEED = 42
    EMG_FS, ACC_FS, TARGET_FS = (2000, 128, 2000)
    EMG_KEY, ACC_KEY, LBL_KEY = ('emg', 'acc', 'restimulus')
    N_EMG_CH = 12
    USE_ACC = True
    REPS_PER_GESTURE, TRAIN_REPS, VAL_REPS, TEST_REPS = (6, 4, 1, 1)
    BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, FILTER_ORDER = (20.0, 450.0, 4)
    NOTCH_HZ, NOTCH_Q = (50.0, 30.0)
    WIN_MS, STEP_MS, TRIM_MS = (400, 100, 100)
    WIN_SAMPLES, STEP_SAMPLES, TRIM_SAMPLES = (800, 200, 200)
    DROPOUT, LR, WEIGHT_DECAY, GRAD_CLIP = (0.15, 0.0003, 0.0001, 5.0)
    MIN_EPOCHS, MAX_EPOCHS, PATIENCE, BATCH_SIZE = (20, 150, 15, 128)
    NUM_WORKERS = 0
    USE_ADAMW, AUGMENT_TRAIN, EVALUATE_TEST, REFIT_ON_TRAIN_PLUS_VAL = (False, False, False, False)
    LABEL_SMOOTHING = 0.0
    EMG_GAIN_STD, EMG_NOISE_STD = (0.1, 0.01)
    RUN_CNN = True
    RAW_CASES_PER_SUBJECT = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AUTOMATION = {}

class RepetitionEMGFilter:
    """Zero-phase EMG filtering applied to one already-assigned repetition only."""

    def __init__(self):
        nyquist = Config.EMG_FS / 2.0
        self.sos = butter(Config.FILTER_ORDER, [Config.BANDPASS_LOW_HZ / nyquist, Config.BANDPASS_HIGH_HZ / nyquist], btype='bandpass', output='sos')
        self.b_notch, self.a_notch = iirnotch(Config.NOTCH_HZ / nyquist, Config.NOTCH_Q)

    def apply(self, repetition_emg):
        repetition_emg = np.asarray(repetition_emg, dtype=np.float32)
        if repetition_emg.ndim != 2:
            raise ValueError(f'Expected (time,channels), got {repetition_emg.shape}.')
        if len(repetition_emg) < 64:
            raise RuntimeError(f'EMG repetition unexpectedly short: {len(repetition_emg)} samples.')
        filtered = sosfiltfilt(self.sos, repetition_emg, axis=0)
        filtered = filtfilt(self.b_notch, self.a_notch, filtered, axis=0)
        return filtered.astype(np.float32)

class SubjectLoader:

    @staticmethod
    def _split_repetition_indices(sid, gesture):
        rng = np.random.default_rng(Config.SEED + 1009 * int(sid) + 9176 * int(gesture))
        perm = rng.permutation(Config.REPS_PER_GESTURE).tolist()
        test_idx = sorted(perm[:Config.TEST_REPS])
        val_idx = sorted(perm[Config.TEST_REPS:Config.TEST_REPS + Config.VAL_REPS])
        train_idx = sorted(perm[Config.TEST_REPS + Config.VAL_REPS:])
        if len(train_idx) != Config.TRAIN_REPS or len(val_idx) != Config.VAL_REPS or len(test_idx) != Config.TEST_REPS:
            raise RuntimeError('Unexpected repetition split size.')
        return (train_idx, val_idx, test_idx)

import torch

import torch.nn as nn

class ParallelMultiKernelBlock(nn.Module):
    """
    True multi-kernel block: parallel branches with different kernel sizes
    (default 3, 5, 7) processed at the SAME depth and concatenated along the
    channel dimension, then merged with a 1x1 conv. This captures multi-scale
    temporal patterns simultaneously (unlike a sequential 7->5->3 design,
    which only changes kernel size across depth, not within one stage).
    """

    def __init__(self, in_ch, out_ch, kernels=(3, 5, 7), pool=True, dropout=0.1):
        super().__init__()
        for k in kernels:
            assert k % 2 == 1, f"kernel size {k} must be odd so that padding=kernel//2 gives symmetric 'same' padding"
        branch_sizes = self._split_channels(out_ch, len(kernels))
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(in_ch, b_ch, k, padding=k // 2, bias=False), nn.BatchNorm1d(b_ch), nn.ReLU(inplace=True), nn.Conv1d(b_ch, b_ch, k, padding=k // 2, bias=False), nn.BatchNorm1d(b_ch), nn.ReLU(inplace=True)) for k, b_ch in zip(kernels, branch_sizes)])
        self.merge = nn.Sequential(nn.Conv1d(out_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    @staticmethod
    def _split_channels(total, n):
        base, rem = divmod(total, n)
        return [base + 1 if i < rem else base for i in range(n)]

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.merge(x)
        x = self.drop(x)
        return self.pool(x)

class ChannelAttentionBlock(nn.Module):
    """
    Squeeze-and-Excitation style CHANNEL attention. Learns which learned feature channels
    matter most; it does NOT attend across time
    steps. Named explicitly so it isn't confused with temporal attention.
    """

    def __init__(self, in_ch, reduction=4):
        super().__init__()
        reduced = max(in_ch // reduction, 1)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(in_ch, reduced, 1), nn.ReLU(inplace=True), nn.Conv1d(reduced, in_ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)

class TemporalAttentionPool(nn.Module):
    """
    Learned attention-weighted pooling over the time axis, replacing plain
    global average pooling. A 1x1 conv scores every time step, softmax turns
    scores into weights, and the weighted sum replaces a uniform mean — so
    the model can down-weight uninformative (e.g. resting) time steps instead
    of averaging them in blindly.
    """

    def __init__(self, in_ch):
        super().__init__()
        self.score = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)

class OriginalMultiKernelAttention1DCNN(nn.Module):

    def __init__(self, n_channels, n_classes, dropout=0.15):
        super().__init__()
        self.n_channels = int(n_channels)
        self.stage1 = ParallelMultiKernelBlock(n_channels, 64, kernels=(3, 5, 7), pool=True, dropout=dropout)
        self.attn1 = ChannelAttentionBlock(64)
        self.stage2 = ParallelMultiKernelBlock(64, 128, kernels=(3, 5, 7), pool=True, dropout=dropout)
        self.attn2 = ChannelAttentionBlock(128)
        self.stage3 = ParallelMultiKernelBlock(128, 256, kernels=(3, 5, 7), pool=False, dropout=dropout)
        self.attn3 = ChannelAttentionBlock(256)
        self.temporal_pool = TemporalAttentionPool(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.attn1(x)
        x = self.stage2(x)
        x = self.attn2(x)
        x = self.stage3(x)
        x = self.attn3(x)
        x = self.temporal_pool(x)
        return self.head(x)

    def count_params(self):
        return sum((parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

import torch.nn.functional as F

class Trainer:

    def __init__(self, model, save_path: Path, n_classes: int):
        self.model = model.to(Config.DEVICE)
        self.save_path = save_path
        self.n_classes = n_classes
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.train_wall = 0.0

    def _run_epoch(self, loader, optimizer=None, criterion=None):
        training = optimizer is not None
        self.model.train(training)
        total_loss, correct, total = (0.0, 0, 0)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for X, y in loader:
                X = X.to(Config.DEVICE, non_blocking=True)
                y = y.to(Config.DEVICE, non_blocking=True)
                if training and Config.AUGMENT_TRAIN:
                    X = X.clone()
                    emg = X[:, :Config.N_EMG_CH]
                    gain = torch.exp(Config.EMG_GAIN_STD * torch.randn(emg.shape[0], emg.shape[1], 1, device=emg.device))
                    X[:, :Config.N_EMG_CH] = emg * gain + Config.EMG_NOISE_STD * torch.randn_like(emg)
                logits = self.model(X)
                loss = criterion(logits, y)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                    optimizer.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        return (total_loss / max(total, 1), correct / max(total, 1))

    def fit(self, train_loader, val_loader):
        optimizer = (torch.optim.AdamW if Config.USE_ADAMW else Adam)(self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)
        patience_count = 0
        start_wall = time.perf_counter()
        for epoch in range(1, Config.MAX_EPOCHS + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, optimizer, criterion)
            vl_loss, vl_acc = self._run_epoch(val_loader, criterion=criterion)
            scheduler.step()
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)
            if vl_loss < self.best_val_loss:
                self.best_val_loss = float(vl_loss)
                self.best_epoch = int(epoch)
                patience_count = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_count += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f'  Epoch {epoch:3d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}')
            if epoch >= Config.MIN_EPOCHS and patience_count >= Config.PATIENCE:
                print(f'  Early stop at epoch {epoch} (patience={Config.PATIENCE})')
                break
        self.train_wall = time.perf_counter() - start_wall
        print(f'  Selection training: {self.train_wall:.1f} s | best epoch={self.best_epoch} | best val_loss={self.best_val_loss:.4f}')
        return self

    def fit_fixed_epochs(self, train_loader, epochs: int):
        """Fresh final model training on train+validation after epoch selection."""
        optimizer = (torch.optim.AdamW if Config.USE_ADAMW else Adam)(self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)
        start_wall = time.perf_counter()
        for epoch in range(1, int(epochs) + 1):
            loss, acc = self._run_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
                print(f'  Refit epoch {epoch:3d}/{int(epochs)} | loss={loss:.4f} | acc={acc:.4f}')
        self.train_wall = time.perf_counter() - start_wall
        torch.save(self.model.state_dict(), self.save_path)
        print(f'  Refit training: {self.train_wall:.1f} s')
        return self

import sys, platform, traceback, zipfile

from datetime import datetime, timezone

from sklearn.metrics import classification_report, balanced_accuracy_score

from scipy.signal import welch

def json_write(path, obj):

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return str(x)
    Path(path).write_text(json.dumps(obj, indent=2, default=convert), encoding='utf-8')

def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)

def probability_metrics(y, p):
    """Uncalibrated confidence diagnostics; multiclass Brier is a sum over classes."""
    pred = p.argmax(1)
    confidence = p.max(1)
    correct = pred == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    calibration = []
    ece = 0.0
    for b in range(10):
        mask = bins == b
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        calibration.append(dict(bin=b, lower=b / 10, upper=(b + 1) / 10, count=n, accuracy=acc, confidence=conf))
        if n:
            ece += n / len(y) * abs(acc - conf)
    targets = np.eye(p.shape[1])[y]
    result = dict(accuracy=float(correct.mean()), balanced_accuracy=float(balanced_accuracy_score(y, pred)), f1_macro=float(f1_score(y, pred, labels=np.arange(p.shape[1]), average='macro', zero_division=0)), f1_weighted=float(f1_score(y, pred, average='weighted', zero_division=0)), nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()), brier_multiclass=float(((p - targets) ** 2).sum(1).mean()), ece_10_bins=float(ece), high_confidence_error_fraction=float(((confidence >= 0.9) & ~correct).mean()), n_windows=int(len(y)))
    return (result, calibration)

class DiagnosticTrainer(Trainer):
    """Preserves Trainer.fit selection logic; records each epoch without extra forwards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_rows = []
        self.last_training = {}

    def _run_epoch(self, loader, optimizer=None, criterion=None):
        training = optimizer is not None
        self.model.train(training)
        total_loss, correct, total = (0.0, 0, 0)
        norms, pred_all, y_all = ([], [], [])
        tick = time.perf_counter()
        with torch.set_grad_enabled(training):
            for X, y in loader:
                X, y = (X.to(Config.DEVICE), y.to(Config.DEVICE))
                if training and Config.AUGMENT_TRAIN:
                    X = X.clone()
                    emg = X[:, :Config.N_EMG_CH]
                    gain = torch.exp(Config.EMG_GAIN_STD * torch.randn(emg.shape[0], emg.shape[1], 1, device=emg.device))
                    X[:, :Config.N_EMG_CH] = emg * gain + Config.EMG_NOISE_STD * torch.randn_like(emg)
                logits = self.model(X)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss; inspect signal_health.csv')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Non-finite gradient norm')
                    norms.append(float(norm.item()))
                    optimizer.step()
                pred = logits.argmax(1)
                total_loss += loss.item() * len(y)
                correct += (pred == y).sum().item()
                total += len(y)
                pred_all.extend(pred.detach().cpu().tolist())
                y_all.extend(y.cpu().tolist())
        values = dict(loss=total_loss / total, accuracy=correct / total, f1_macro=float(f1_score(y_all, pred_all, labels=range(self.n_classes), average='macro', zero_division=0)), seconds=time.perf_counter() - tick)
        labels = np.asarray(y_all)
        guesses = np.asarray(pred_all)
        if not hasattr(self, 'class_epoch_rows'):
            self.class_epoch_rows = []
        for c in range(self.n_classes):
            selected = labels == c
            self.class_epoch_rows.append(dict(epoch=len(self.epoch_rows) + 1, split='train_online_dropout' if training else 'validation', gesture=c + Config.GESTURE_MIN, windows=int(selected.sum()), recall=float((guesses[selected] == c).mean())))
        pd.DataFrame(self.class_epoch_rows).to_csv(self.save_path.parent / 'class_learning_history.csv', index=False)
        if training:
            self.last_training = {'train_' + k: v for k, v in values.items()}
            self.last_training.update(lr=optimizer.param_groups[0]['lr'], gradient_norm_mean=float(np.mean(norms)), gradient_norm_max=float(np.max(norms)), gradient_clipped_fraction=float(np.mean(np.array(norms) > Config.GRAD_CLIP)))
        else:
            row = dict(epoch=len(self.epoch_rows) + 1, **self.last_training, **{'val_' + k: v for k, v in values.items()})
            self.epoch_rows.append(row)
            pd.DataFrame(self.epoch_rows).to_csv(self.save_path.parent / 'history.csv', index=False)
        return (values['loss'], values['accuracy'])

def learning_report(trainer, directory):
    h = pd.DataFrame(trainer.epoch_rows)
    h.to_csv(directory / 'history.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for split in ('train', 'val'):
        axes[0, 0].plot(h.epoch, h[split + '_loss'], label=split)
        axes[0, 1].plot(h.epoch, h[split + '_accuracy'], label=split)
    axes[0, 0].set_ylabel('Cross entropy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[1, 0].plot(h.epoch, h.lr)
    axes[1, 0].set_ylabel('Learning rate')
    axes[1, 1].plot(h.epoch, h.gradient_norm_mean, label='Mean before clipping')
    axes[1, 1].plot(h.epoch, h.gradient_norm_max, label='Max before clipping')
    axes[1, 1].axhline(Config.GRAD_CLIP, color='gray', linestyle='--')
    axes[1, 1].set_ylabel('Gradient norm')
    for ax in axes.flat:
        ax.axvline(trainer.best_epoch, color='red', linestyle=':', label='Selected epoch')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
    fig.suptitle('Online training uses dropout; checkpoint gap is measured separately in eval mode')
    save_figure(fig, directory / 'learning_curves.png')

def predict_with_diagnostics(model, dataset, retain_attention=False):
    """Evaluate the checkpoint; collect embeddings and optionally attention on every window."""
    model.eval()
    captured, probabilities, embeddings, attention = ({}, [], [], {})

    def capture(name):

        def hook(module, args, result):
            captured[name] = result.detach()
        return hook
    handles = [model.temporal_pool.register_forward_hook(capture('embedding'))]
    if retain_attention:
        handles += [model.temporal_pool.score.register_forward_hook(capture('time_scores'))]
        for stage in [1, 2, 3]:
            handles.append(getattr(model, f'attn{stage}').attention.register_forward_hook(capture(f'channel_stage{stage}')))
    try:
        with torch.no_grad():
            for signal, _ in DataLoader(dataset, batch_size=128, shuffle=False):
                logits = model(signal.to(Config.DEVICE))
                probabilities.append(logits.softmax(1).cpu().numpy())
                embeddings.append(captured['embedding'].cpu().numpy())
                if retain_attention:
                    for name in ['time_scores', 'channel_stage1', 'channel_stage2', 'channel_stage3']:
                        values = captured[name].softmax(-1) if name == 'time_scores' else captured[name]
                        attention.setdefault(name, []).append(values.squeeze(1 if name == 'time_scores' else -1).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return (np.concatenate(probabilities), np.concatenate(embeddings), {k: np.concatenate(v) for k, v in attention.items()})

CHANNEL_COUNTS = {'emg': 12, 'acc': 36, 'gyro': 36, 'mag': 36}

def load_modalities(subject, directory):
    """Read only selected sensor arrays; use refined labels for every arm."""
    matches = sorted(Config.KAGGLE_INPUT.rglob(f'S{subject}_E1_A1.mat'))
    if len(matches) != 1:
        raise ValueError(f'Expected one S{subject}_E1_A1.mat under {Config.KAGGLE_INPUT}; found {len(matches)}')
    path = matches[0]
    keys = list(Config.MODALITIES) + ['restimulus', 'rerepetition', 'stimulus', 'subject', 'exercise']
    data = io.loadmat(path, variable_names=keys)
    labels = np.asarray(data['restimulus']).reshape(-1)
    repetitions = np.asarray(data['rerepetition']).reshape(-1)
    stimulus = np.asarray(data['stimulus']).reshape(-1)
    assert len(labels) == len(repetitions) == len(stimulus)
    assert set(np.unique(labels)) == set(range(18))
    assert int(data['exercise'].item()) == 1
    arrays, names, hashes = [], [], {}
    for modality in Config.MODALITIES:
        values = np.asarray(data[modality], dtype=np.float32)
        assert values.shape == (len(labels), CHANNEL_COUNTS[modality]), (modality, values.shape)
        assert np.isfinite(values).all(), f'Nonfinite {modality}: subject {subject}'
        hashes[modality] = hashlib.sha256(values.tobytes()).hexdigest()
        arrays.append(values)
        names.extend(f'{modality}_{i+1:02}' for i in range(values.shape[1]))
    signal = np.concatenate(arrays, axis=1)
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    rows = []
    emg_filter = RepetitionEMGFilter() if 'emg' in Config.MODALITIES else None
    for gesture in range(1,18):
        runs = [(int(a),int(b)) for a,b in zip(boundaries[:-1],boundaries[1:]) if labels[a] == gesture]
        assert len(runs) == 6, (subject, gesture, len(runs))
        train_ids, val_ids, test_ids = SubjectLoader._split_repetition_indices(subject, gesture)
        seen = set()
        for index,(start,end) in enumerate(runs):
            native = np.unique(repetitions[start:end])
            assert len(native) == 1 and 1 <= native[0] <= 6 and int(native[0]) not in seen
            seen.add(int(native[0]))
            split = 'train' if index in train_ids else 'validation' if index in val_ids else 'test'
            if emg_filter is not None:
                # EMG is first in every combination containing it.
                signal[start:end,:12] = emg_filter.apply(signal[start:end,:12])
            rows.append(dict(subject=subject, gesture=gesture, native_repetition=int(native[0]),
                split=split,file_name=path.name,run_start=start,run_end=end))
    metadata = pd.DataFrame(rows)
    metadata.to_csv(directory/'repetition_inventory.csv',index=False)
    identity = dict(folder_subject=subject, internal_subject=int(data['subject'].item()),
        exercise=1,modalities=Config.MODALITIES,channel_names=names,signal_shape=list(signal.shape),
        source_hashes=hashes,identity_matches=int(data['subject'].item())==subject,
        sampling_note='Supplied synchronized row grid, 2000 rows/s; inertial acquisition rate is not inferred from row count.')
    json_write(directory/'identity.json',identity)
    if not identity['identity_matches']:
        print(f'Identity discrepancy for folder S{subject}: internal subject {identity["internal_subject"]}; retained and recorded.')
    return dict(signal=signal,metadata=metadata,stimulus=stimulus,channel_names=names)

class ModalityWindows(Dataset):
    """All split boundaries and windows are independent of selected modalities."""
    def __init__(self, recording, split):
        self.recording=recording
        self.samples=Config.WIN_SAMPLES
        rows=[]
        for rep in recording['metadata'].to_dict('records'):
            if rep['split'] != split: continue
            for end in range(rep['run_start']+Config.TRIM_SAMPLES+self.samples,
                             rep['run_end']-Config.TRIM_SAMPLES+1,Config.STEP_SAMPLES):
                start=end-self.samples
                phase=((start+end)/2-rep['run_start'])/(rep['run_end']-rep['run_start'])
                rows.append(dict(**rep,window_start=start,window_end=end,
                    phase=['early','middle','late'][min(2,int(phase*3))],
                    stimulus_disagreement_fraction=float(np.mean(recording['stimulus'][start:end]!=rep['gesture']))))
        self.meta=pd.DataFrame(rows)
        assert len(self.meta) and self.meta.gesture.nunique()==17
        self.starts=self.meta.window_start.to_numpy(dtype=np.int64)
        self.y=self.meta.gesture.to_numpy(dtype=np.int64)-1
        channels=recording['signal'].shape[1]
        self.mean=np.zeros((channels,1),np.float32)
        self.std=np.ones((channels,1),np.float32)
    def __len__(self): return len(self.y)
    def physical(self,index):
        start=self.starts[index]
        return self.recording['signal'][start:start+self.samples].T.copy()
    def __getitem__(self,index):
        return torch.from_numpy((self.physical(index)-self.mean)/self.std), int(self.y[index])

def train_scaler(dataset):
    total=np.zeros(dataset.mean.shape[0],np.float64); squares=total.copy(); count=0
    for begin in range(0,len(dataset),64):
        x=np.stack([dataset.physical(i) for i in range(begin,min(begin+64,len(dataset)))]).astype(np.float64)
        total+=x.sum(axis=(0,2)); squares+=(x*x).sum(axis=(0,2));count+=x.shape[0]*x.shape[2]
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0))+1e-8
    return mean.astype(np.float32)[:,None],std.astype(np.float32)[:,None]
