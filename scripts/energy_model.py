"""C1: frame-aligned EMG waveform, log-power and inertial encoders for DB7.

The caller supplies filtered, training-channel-standardized windows in this order:
EMG (12), ACC (36), gyroscope (36), magnetometer (36). This module does not read
recordings, fit the raw input scaler, choose data splits, or filter signals.

Before training, call ``model.fit_spectral_scaler(training_loader, split='train')``.
That loader must contain only already-standardized training windows. Spectral
statistics are registered buffers and travel with every model checkpoint.
"""

from __future__ import annotations

import io as checkpoint_io
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EXPECTED_PARAMETERS = 551_542
ENERGY_BRANCH_PARAMETERS = 3_968
ENERGY_EXPECTED_PARAMETERS = {'C1': 551_542, 'C2A': 555_510, 'C2D': 555_510}
EXPECTED_PARAMETER_BREAKDOWN = {
    'waveform': 81_492,
    'spectral': 67_936,
    'inertial': 123_760,
    'fusion': 41_216,
    'temporal': 197_888,
    'attention': 4_161,
    'classifier': 35_089,
}
FRAME_SAMPLES = 200
FRAME_HOP = 100
WINDOW_SAMPLES = 800
N_FRAMES = 7


def unfold_frames(signal: Tensor) -> Tensor:
    """Return [batch, channels, 7, 200], using only the supplied 800 rows."""
    if signal.ndim != 3 or signal.shape[-1] != WINDOW_SAMPLES:
        raise ValueError(f'Expected [batch, channels, 800], received {tuple(signal.shape)}')
    return signal.unfold(-1, FRAME_SAMPLES, FRAME_HOP)


def log_power(z_emg: Tensor, hann: Tensor | None = None) -> Tensor:
    """Return unscaled log-power [batch, 12, 44, 7] for 20:10:450 Hz.

    ``z_emg`` has already received the fixed training input scaler. Explicit
    200-sample frames avoid the FFT-length-dependent framing of torch.stft.
    The FFT is at least float32, including inside an autocast context: CUDA
    half-precision FFTs cannot implement this non-power-of-two length.
    """
    if z_emg.ndim != 3 or z_emg.shape[1:] != (12, WINDOW_SAMPLES):
        raise ValueError(f'Expected [batch, 12, 800], received {tuple(z_emg.shape)}')
    if not z_emg.is_floating_point():
        raise TypeError('EMG input must be a floating-point tensor')
    fft_dtype = torch.float64 if z_emg.dtype == torch.float64 else torch.float32
    if hann is None:
        hann = torch.hann_window(FRAME_SAMPLES, periodic=True,
                                 device=z_emg.device, dtype=fft_dtype)
    else:
        hann = hann.to(device=z_emg.device, dtype=fft_dtype)
        if hann.shape != (FRAME_SAMPLES,):
            raise ValueError('Hann window must contain exactly 200 samples')
    with torch.autocast(device_type=z_emg.device.type, enabled=False):
        frames = unfold_frames(z_emg.to(dtype=fft_dtype))
        spectrum = torch.fft.rfft(frames * hann, n=FRAME_SAMPLES, dim=-1)
        power = spectrum.abs().square() / hann.square().sum()
        # [B, 12, frame, frequency] -> [B, 12, frequency, frame].
        return torch.log(power[..., 2:46] + 1e-8).permute(0, 1, 3, 2).contiguous()


def _conv_bn_relu(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(),
    )


class ChannelSqueezeExcitation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // 8), nn.ReLU(),
            nn.Linear(channels // 8, channels), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x.mean(-1)).unsqueeze(-1)


class FrameMultiKernelBlock(nn.Module):
    """Three single-convolution paths, plus projected residual and frame SE."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.paths = nn.ModuleList(_conv_bn_relu(in_channels, 32, k) for k in (3, 5, 7))
        self.merge = nn.Sequential(
            nn.Conv1d(96, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.se = ChannelSqueezeExcitation(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        merged = self.merge(torch.cat([path(x) for path in self.paths], dim=1))
        return self.se(F.relu(merged + self.skip(x)))


class WaveformEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.Sequential(
            FrameMultiKernelBlock(12, 64), nn.MaxPool1d(2),
            FrameMultiKernelBlock(64, 96), nn.MaxPool1d(2),
        )
        self.projection = nn.Linear(192, 96)

    def forward(self, frames: Tensor) -> Tensor:
        # The same encoder processes every frame; batch and frame remain distinct.
        batch = frames.shape[0]
        x = frames.permute(0, 2, 1, 3).reshape(batch * N_FRAMES, 12, FRAME_SAMPLES)
        x = self.stages(x)
        x = F.relu(self.projection(torch.cat([x.mean(-1), x.amax(-1)], dim=1)))
        return x.reshape(batch, N_FRAMES, 96).transpose(1, 2).contiguous()


class SpectralEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for incoming, outgoing, kernel, stride in ((12, 32, 5, 2),
                                                   (32, 64, 5, 2),
                                                   (64, 96, 3, 1)):
            layers.extend([
                nn.Conv2d(incoming, outgoing, (kernel, 1), stride=(stride, 1),
                          padding=(kernel // 2, 0), bias=False),
                nn.BatchNorm2d(outgoing, eps=1e-5, momentum=0.1), nn.ReLU(),
            ])
        self.stages = nn.Sequential(*layers)
        self.projection = nn.Conv1d(384, 96, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stages(x)
        # Preserve four ordered feature-frequency regions, rather than collapsing
        # absolute frequency location into a single global mean/max vector.
        regions = torch.stack([x[:, :, begin:end, :].mean(2)
                               for begin, end in ((0, 3), (3, 6), (6, 9), (9, 11))], dim=2)
        # Channel-major: each channel retains regions low -> high in adjacent slots.
        return F.relu(self.projection(regions.flatten(1, 2)))


class InertialModalityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.Sequential(
            _conv_bn_relu(36, 48, 5), nn.AvgPool1d(4),
            _conv_bn_relu(48, 64, 5),
        )
        self.projection = nn.Linear(128, 48)

    def forward(self, frames: Tensor) -> Tensor:
        x = self.stages(frames)
        return F.relu(self.projection(torch.cat([x.mean(-1), x.amax(-1)], dim=1)))


class InertialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modalities = nn.ModuleList(InertialModalityEncoder() for _ in range(3))
        self.dynamic_projection = nn.Linear(144, 128)
        self.mean_projection = nn.Linear(108, 128)

    def forward(self, frames: Tensor) -> Tensor:
        batch = frames.shape[0]
        x = frames.permute(0, 2, 1, 3).reshape(batch * N_FRAMES, 108, FRAME_SAMPLES)
        encoded = [encoder(x[:, i * 36:(i + 1) * 36, :])
                   for i, encoder in enumerate(self.modalities)]
        dynamic = self.dynamic_projection(torch.cat(encoded, dim=1))
        static = self.mean_projection(x.mean(-1))
        # The mean is an additional feature; it is never subtracted from frames.
        output = F.relu(dynamic + static)
        return output.reshape(batch, N_FRAMES, 128).transpose(1, 2).contiguous()


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(128, 192, 3, dilation=dilation, padding=dilation, bias=False),
            nn.BatchNorm1d(192, eps=1e-5, momentum=0.1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(192, 128, 1, bias=False),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.layers(x))


class ThreeBranchC1(nn.Module):
    """Exact 551,542-parameter C1, compatible with Trainer's model(x) call."""
    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.waveform = WaveformEncoder()
        self.spectral = SpectralEncoder()
        self.inertial = InertialEncoder()
        self.fusion = nn.Sequential(
            nn.Conv1d(320, 128, 1, bias=False),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1), nn.ReLU(),
        )
        self.temporal = nn.Sequential(ResidualTemporalBlock(1, dropout),
                                      ResidualTemporalBlock(2, dropout))
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 17))
        self.register_buffer('hann', torch.hann_window(FRAME_SAMPLES, periodic=True))
        self.register_buffer('spectral_mean', torch.zeros(12, 44))
        self.register_buffer('spectral_std', torch.ones(12, 44))
        self.register_buffer('spectral_constant', torch.zeros(12, 44, dtype=torch.bool))
        self.register_buffer('spectral_fitted', torch.tensor(False))
        self.register_buffer('spectral_fit_frames', torch.tensor(0, dtype=torch.long))
        if self.count_params() != EXPECTED_PARAMETERS:
            raise AssertionError(f'C1 parameter mismatch: {self.count_params()}')

    def count_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        return {name: sum(parameter.numel() for parameter in getattr(self, name).parameters())
                for name in EXPECTED_PARAMETER_BREAKDOWN}

    @torch.no_grad()
    def fit_spectral_scaler(self, training_batches: Iterable[Any], *, split: str = 'train',
                            std_floor: float = 1e-6) -> dict[str, Any]:
        """Fit frozen moments without running a CNN or updating BatchNorm.

        The iterable yields standardized x or (x,y) batches. One observation for
        each channel/frequency bin is one frame; all seven frames of every
        training window receive equal weight. Overlap is intentional. Population
        variance uses a stable float64 batched merge; SD below ``std_floor`` is
        flagged and clamped. A second fit is rejected to avoid accidental reuse
        on validation/test; create a fresh model for a new subject or fold.

        The explicit split guard cannot establish arbitrary generator provenance.
        The caller must save source/window hashes. A DataLoader exposing a dataset
        with a ``meta['split']`` column receives an additional provenance check.
        """
        if split != 'train':
            raise ValueError('Spectral scaler may be fitted only on the train split')
        if bool(self.spectral_fitted.item()):
            raise RuntimeError('Spectral scaler is already fitted; use a fresh model for another fold')
        if std_floor <= 0:
            raise ValueError('std_floor must be positive')
        dataset = getattr(training_batches, 'dataset', None)
        metadata = getattr(dataset, 'meta', None)
        if metadata is not None and 'split' in metadata:
            if set(metadata['split'].unique()) != {'train'}:
                raise ValueError('Spectral fitting DataLoader contains non-training metadata')
        device = self.spectral_mean.device
        count = 0
        mean = torch.zeros(12, 44, dtype=torch.float64, device=device)
        m2 = torch.zeros_like(mean)
        for batch in training_batches:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = torch.as_tensor(x, device=device, dtype=self.spectral_mean.dtype)
            if x.ndim != 3 or x.shape[1:] != (120, WINDOW_SAMPLES) or x.shape[0] == 0:
                raise ValueError('Spectral scaler requires nonempty [B,120,800] training batches')
            if not bool(torch.isfinite(x).all().item()):
                raise ValueError('Nonfinite training input in spectral scaler fit')
            values = log_power(x[:, :12], self.hann).to(torch.float64)
            batch_count = values.shape[0] * N_FRAMES
            batch_mean = values.mean(dim=(0, 3))
            batch_m2 = (values - batch_mean[None, :, :, None]).square().sum(dim=(0, 3))
            combined_count = count + batch_count
            delta = batch_mean - mean
            m2 += batch_m2 + delta.square() * (count * batch_count / combined_count)
            mean += delta * (batch_count / combined_count)
            count = combined_count
        if not count:
            raise ValueError('Cannot fit the spectral scaler on an empty iterable')
        std = (m2 / count).clamp_min(0).sqrt()
        self.spectral_mean.copy_(mean)
        self.spectral_constant.copy_(std < std_floor)
        self.spectral_std.copy_(std.clamp_min(std_floor))
        self.spectral_fit_frames.fill_(count)
        self.spectral_fitted.fill_(True)
        return {
            'split': 'train', 'windows': count // N_FRAMES, 'frames': count,
            'statistic': 'float64_population_moments_over_training_windows_and_seven_frames',
            'std_floor': float(std_floor),
            'constant_bins': int(self.spectral_constant.sum().item()),
            'mean': self.spectral_mean.detach().cpu().tolist(),
            'std': self.spectral_std.detach().cpu().tolist(),
            'constant': self.spectral_constant.detach().cpu().tolist(),
        }

    def _waveform_tokens(self, x: Tensor, frames: Tensor) -> Tensor:
        """Extension point: C1's waveform calculation remains exactly unchanged."""
        return self.waveform(frames[:, :12])

    def forward_features(self, x: Tensor, *, retain_sequences: bool = False) -> dict[str, Any]:
        if x.ndim != 3 or x.shape[1:] != (120, WINDOW_SAMPLES):
            raise ValueError(f'C1 expects [B,120,800], received {tuple(x.shape)}')
        if not bool(self.spectral_fitted.item()):
            raise RuntimeError('Fit the spectral scaler using training windows before model(x)')
        frames = unfold_frames(x)
        waveform = self._waveform_tokens(x, frames)
        power = log_power(x[:, :12], self.hann)
        scaled_power = ((power - self.spectral_mean[None, :, :, None]) /
                        self.spectral_std[None, :, :, None])
        spectral = self.spectral(scaled_power)
        inertial = self.inertial(frames[:, 12:])
        fused = self.temporal(self.fusion(torch.cat([waveform, spectral, inertial], dim=1)))
        scores = self.attention(fused.transpose(1, 2)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        weighted = (fused * attention[:, None, :]).sum(-1)
        embedding = torch.cat([weighted, fused.mean(-1)], dim=1)
        result = {
            'logits': self.classifier(embedding), 'embedding': embedding,
            'branch_embeddings': {'waveform': waveform.mean(-1),
                                  'spectral': spectral.mean(-1),
                                  'inertial': inertial.mean(-1)},
            'attention': attention,
        }
        if retain_sequences:
            result['sequences'] = {'waveform': waveform, 'spectral': spectral,
                                   'inertial': inertial, 'fused': fused}
        return result

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)['logits']


def energy_log_features(z_emg: Tensor, input_offset: Tensor,
                        energy_mode: str = 'dynamic') -> Tensor:
    """Unscaled energy features [B,24,7], using only each frame's own samples.

    ``z_emg`` is (filtered physical EMG - train mean) / train SD. Adding the
    saved ``input_offset = train mean / train SD`` recovers physical EMG / SD,
    retaining the physical zero without undoing amplitude scaling. For each
    200-sample frame, RMS25 uses its last 50 samples; RMS100 uses all 200.

    Amplitude mode duplicates log(RMS25 + 1e-6), giving the same 24-input
    projection as dynamic mode without revealing RMS100 or its ratio.
    Dynamic mode concatenates log(RMS25 + 1e-6) and its difference from
    log(RMS100 + 1e-6). A 1e-24 power floor keeps zero-input derivatives
    finite; the corresponding RMS floor is negligible relative to 1e-6.
    The common 1/sqrt(2) factor is applied AFTER training standardization.
    """
    if energy_mode not in ('amplitude', 'dynamic'):
        raise ValueError('energy_mode must be amplitude or dynamic')
    if z_emg.ndim != 3 or z_emg.shape[1:] != (12, WINDOW_SAMPLES):
        raise ValueError('Energy features require [B,12,800] standardized EMG')
    if not z_emg.is_floating_point():
        raise TypeError('Energy input must be floating point')
    dtype = torch.float64 if z_emg.dtype == torch.float64 else torch.float32
    offset = torch.as_tensor(input_offset, device=z_emg.device, dtype=dtype)
    if offset.numel() != 12:
        raise ValueError('Energy input offset must contain 12 training EMG mean/SD ratios')
    with torch.autocast(device_type=z_emg.device.type, enabled=False):
        physical_scaled = z_emg.to(dtype=dtype) + offset.reshape(1, 12, 1)
        frames = unfold_frames(physical_scaled)
        r25 = frames[..., -50:].square().mean(-1).clamp_min(1e-24).sqrt()
        log25 = torch.log(r25 + 1e-6)
        if energy_mode == 'amplitude':
            second = log25
        else:
            r100 = frames.square().mean(-1).clamp_min(1e-24).sqrt()
            second = log25 - torch.log(r100 + 1e-6)
        return torch.cat([log25, second], dim=1)


class ThreeBranchEnergy(ThreeBranchC1):
    """C2A/C2D: identical 3,968-parameter residual paths with different cues.

    Load a C1 initialization using ``load_state_dict(c1.state_dict(), strict=False)``;
    every missing key must start with ``energy_`` and no key should be unexpected.
    To pair C2A/C2D initialization, copy ``energy_projection.state_dict()`` as well.
    Do not copy the entire C2A state into C2D: the checkpointed energy mode and
    its fitted feature moments intentionally differ between these controls.
    """
    def __init__(self, energy_mode: str = 'dynamic', dropout: float = 0.15) -> None:
        if energy_mode not in ('amplitude', 'dynamic'):
            raise ValueError('energy_mode must be amplitude or dynamic')
        super().__init__(dropout=dropout)
        self.energy_projection = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 96))
        nn.init.zeros_(self.energy_projection[-1].weight)
        nn.init.zeros_(self.energy_projection[-1].bias)
        self.register_buffer('energy_mean', torch.zeros(24))
        self.register_buffer('energy_std', torch.ones(24))
        self.register_buffer('energy_constant', torch.zeros(24, dtype=torch.bool))
        self.register_buffer('energy_input_offset', torch.zeros(12, 1))
        self.register_buffer('energy_fitted', torch.tensor(False))
        self.register_buffer('energy_fit_frames', torch.tensor(0, dtype=torch.long))
        self.register_buffer('energy_mode_code', torch.tensor(0 if energy_mode == 'amplitude' else 1))
        if self.count_params() != ENERGY_EXPECTED_PARAMETERS['C2A']:
            raise AssertionError(f'Energy model parameter mismatch: {self.count_params()}')

    @property
    def energy_mode(self) -> str:
        code = int(self.energy_mode_code.item())
        if code not in (0, 1):
            raise RuntimeError('Invalid checkpointed energy mode')
        return 'amplitude' if code == 0 else 'dynamic'

    def parameter_breakdown(self) -> dict[str, int]:
        result = super().parameter_breakdown()
        result['energy_projection'] = sum(p.numel() for p in self.energy_projection.parameters())
        return result

    @torch.no_grad()
    def fit_energy_scaler(self, training_batches: Iterable[Any], *, input_mean: Any,
                          input_std: Any, split: str = 'train',
                          std_floor: float = 1e-6) -> dict[str, Any]:
        """Fit on already input-standardized TRAIN windows; no CNN forward.

        Input mean/SD come from the SAME training-only channel scaler used by
        the loader. They may contain all 120 channels or just the 12 EMG channels.
        Feature moments use every training frame, population variance and a
        stable float64 batched merge. The positive SD floor is checkpointed
        through the resulting SDs; constant channels remain present and flagged.
        """
        if split != 'train':
            raise ValueError('Energy scaler may be fitted only on the train split')
        if bool(self.energy_fitted.item()):
            raise RuntimeError('Energy scaler is already fitted; create a fresh model for another fold')
        if not (0 < std_floor < float('inf')):
            raise ValueError('std_floor must be finite and positive')
        dataset = getattr(training_batches, 'dataset', None)
        metadata = getattr(dataset, 'meta', None)
        if metadata is not None and 'split' in metadata:
            if set(metadata['split'].unique()) != {'train'}:
                raise ValueError('Energy fitting DataLoader contains non-training metadata')
        device = self.energy_mean.device
        mean_input = torch.as_tensor(input_mean, device=device, dtype=torch.float64).reshape(-1)
        std_input = torch.as_tensor(input_std, device=device, dtype=torch.float64).reshape(-1)
        if mean_input.numel() not in (12, 120) or mean_input.shape != std_input.shape:
            raise ValueError('Provide matching training-channel mean/SD arrays of 12 or 120 values')
        if not bool(torch.isfinite(mean_input).all().item() and torch.isfinite(std_input).all().item()):
            raise ValueError('Training input mean/SD must be finite')
        if not bool((std_input > 0).all().item()):
            raise ValueError('Training input SD must be strictly positive')
        self.energy_input_offset.copy_((mean_input[:12] / std_input[:12]).reshape(12, 1))
        count = 0
        mean = torch.zeros(24, dtype=torch.float64, device=device)
        m2 = torch.zeros_like(mean)
        for batch in training_batches:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = torch.as_tensor(x, device=device, dtype=self.energy_mean.dtype)
            if x.ndim != 3 or x.shape[1:] != (120, WINDOW_SAMPLES) or x.shape[0] == 0:
                raise ValueError('Energy scaler requires nonempty [B,120,800] training batches')
            if not bool(torch.isfinite(x).all().item()):
                raise ValueError('Nonfinite training input in energy scaler fit')
            values = energy_log_features(x[:, :12], self.energy_input_offset, self.energy_mode).double()
            batch_count = values.shape[0] * N_FRAMES
            batch_mean = values.mean(dim=(0, 2))
            batch_m2 = (values - batch_mean[None, :, None]).square().sum(dim=(0, 2))
            combined_count = count + batch_count
            delta = batch_mean - mean
            m2 += batch_m2 + delta.square() * (count * batch_count / combined_count)
            mean += delta * (batch_count / combined_count)
            count = combined_count
        if not count:
            raise ValueError('Cannot fit the energy scaler on an empty iterable')
        std = (m2 / count).clamp_min(0).sqrt()
        self.energy_mean.copy_(mean)
        self.energy_constant.copy_(std < std_floor)
        self.energy_std.copy_(std.clamp_min(std_floor))
        self.energy_fit_frames.fill_(count)
        self.energy_fitted.fill_(True)
        return {
            'split': 'train', 'energy_mode': self.energy_mode, 'windows': count // N_FRAMES,
            'frames': count, 'statistic': 'float64_population_moments_over_training_windows_and_seven_frames',
            'std_floor': float(std_floor), 'post_standardization_factor': 2 ** -0.5,
            'feature_order': 'EMG1..12 logR25; EMG1..12 ' +
                ('duplicate logR25' if self.energy_mode == 'amplitude' else 'logR25-minus-logR100'),
            'log_epsilon': 1e-6, 'power_floor': 1e-24,
            'constant_features': int(self.energy_constant.sum().item()),
            'mean': self.energy_mean.detach().cpu().tolist(),
            'std': self.energy_std.detach().cpu().tolist(),
            'constant': self.energy_constant.detach().cpu().tolist(),
            'input_mean_over_std': self.energy_input_offset.detach().cpu().tolist(),
        }

    def energy_features(self, x: Tensor, *, standardized: bool = True) -> Tensor:
        """Return [B,24,7] cues; standardized=True returns actual projection inputs."""
        if not bool(self.energy_fitted.item()):
            raise RuntimeError('Fit the energy scaler on training windows before calculating energy inputs')
        if x.ndim != 3 or x.shape[1:] != (120, WINDOW_SAMPLES):
            raise ValueError('Energy model expects [B,120,800] input')
        values = energy_log_features(x[:, :12], self.energy_input_offset, self.energy_mode)
        if standardized:
            values = ((values - self.energy_mean[None, :, None]) /
                      self.energy_std[None, :, None]) * (2 ** -0.5)
        return values

    def _waveform_tokens(self, x: Tensor, frames: Tensor) -> Tensor:
        baseline = super()._waveform_tokens(x, frames)
        energy = self.energy_features(x).transpose(1, 2)
        return baseline + self.energy_projection(energy).transpose(1, 2)

    def forward_features(self, x: Tensor, *, retain_sequences: bool = False) -> dict[str, Any]:
        if not bool(self.energy_fitted.item()):
            raise RuntimeError('Fit the energy scaler using training windows before model(x)')
        result = super().forward_features(x, retain_sequences=retain_sequences)
        if retain_sequences:
            result['energy_inputs'] = self.energy_features(x)
        return result


def model_preflight(device: str | torch.device = 'cpu') -> dict[str, Any]:
    """Meaningful synthetic checks for the installed Kaggle PyTorch runtime.

    Does not touch recordings or disk, install packages, or start an experiment.
    CPU and applicable CUDA RNG state are restored before return.
    """
    target = torch.device(device)
    # torch.manual_seed also seeds CUDA generators. Preserve every initialized
    # device's stream instead of changing an unused second Kaggle GPU's RNG.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(941)
        model = ThreeBranchC1().to(target)
        assert model.parameter_breakdown() == EXPECTED_PARAMETER_BREAKDOWN
        x = torch.randn(3, 120, WINDOW_SAMPLES, device=target)
        try:
            model(x)
        except RuntimeError as error:
            assert 'spectral scaler' in str(error)
        else:
            raise AssertionError('Unfitted spectral scaling must block model inference')
        try:
            model.fit_spectral_scaler([x], split='test')
        except ValueError as error:
            assert 'train split' in str(error)
        else:
            raise AssertionError('The spectral scaler must reject non-training splits')
        frames = unfold_frames(x)
        assert frames.shape == (3, 120, 7, 200)
        for frame in range(N_FRAMES):
            torch.testing.assert_close(frames[:, :, frame], x[:, :, frame * 100:frame * 100 + 200])
        # Independent explicit-frame transform verifies frequency/frame ordering.
        hann = torch.hann_window(200, periodic=True, device=target)
        reference = torch.stack([
            torch.log(torch.fft.rfft(x[:, :12, begin:begin + 200] * hann, dim=-1)
                      .abs().square()[..., 2:46] / hann.square().sum() + 1e-8)
            for begin in range(0, 601, 100)], dim=-1)
        torch.testing.assert_close(log_power(x[:, :12]), reference)
        before_bn = {name: value.clone() for name, value in model.named_buffers()
                     if 'running_' in name or 'num_batches_tracked' in name}
        statistics = model.fit_spectral_scaler([(x[:2], torch.zeros(2)),
                                                (x[2:], torch.zeros(1))], split='train')
        assert statistics['windows'] == 3 and statistics['frames'] == 21
        try:
            model.fit_spectral_scaler([x], split='train')
        except RuntimeError as error:
            assert 'already fitted' in str(error)
        else:
            raise AssertionError('A fitted spectral scaler must reject accidental refitting')
        reference_double = reference.double()
        expected_mean = reference_double.mean(dim=(0, 3))
        expected_std = reference_double.permute(1, 2, 0, 3).reshape(12, 44, -1).std(-1, correction=0)
        torch.testing.assert_close(model.spectral_mean, expected_mean.float(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(model.spectral_std, expected_std.float(), rtol=2e-5, atol=2e-6)
        for name, value in model.named_buffers():
            if name in before_bn:
                torch.testing.assert_close(value, before_bn[name], rtol=0, atol=0)
        frozen_scaler = {name: value.clone() for name, value in model.named_buffers()
                         if name.startswith('spectral_')}
        model.train()
        outputs = model.forward_features(x, retain_sequences=True)
        assert outputs['logits'].shape == (3, 17)
        assert outputs['embedding'].shape == (3, 256)
        assert outputs['attention'].shape == (3, 7)
        for branch, width in (('waveform', 96), ('spectral', 96), ('inertial', 128), ('fused', 128)):
            assert outputs['sequences'][branch].shape == (3, width, 7)
        torch.testing.assert_close(outputs['attention'].sum(-1), torch.ones(3, device=target))
        loss = F.cross_entropy(outputs['logits'], torch.tensor([0, 8, 16], device=target))
        loss.backward()
        for name, module in (('waveform', model.waveform), ('spectral', model.spectral),
                             ('acc', model.inertial.modalities[0]),
                             ('gyro', model.inertial.modalities[1]),
                             ('mag', model.inertial.modalities[2]),
                             ('inertial_mean', model.inertial.mean_projection),
                             ('fusion', model.fusion)):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(bool(torch.isfinite(g).all().item()) for g in gradients), name
            assert any(bool(g.abs().sum().item() > 0) for g in gradients), name
        for name, value in model.named_buffers():
            if name in frozen_scaler:
                torch.testing.assert_close(value, frozen_scaler[name], rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            # An input to another branch cannot change waveform/spectral tokens.
            changed = x.clone()
            changed[:, 12:] += 1.75
            original_features = model.forward_features(x, retain_sequences=True)
            altered_features = model.forward_features(changed, retain_sequences=True)
            for name in ('waveform', 'spectral'):
                torch.testing.assert_close(original_features['sequences'][name],
                                           altered_features['sequences'][name], rtol=0, atol=0)
            checkpoint = checkpoint_io.BytesIO()
            torch.save(model.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = ThreeBranchC1().to(target)
            restored.load_state_dict(torch.load(checkpoint, map_location=target, weights_only=True))
            restored.eval()
            torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        return {'success': True, 'parameters': model.count_params(),
                'parameter_breakdown': model.parameter_breakdown(), 'device': str(target),
                'torch_version': torch.__version__, 'frames': 7, 'frequency_bins': 44,
                'output_shape': list(expected.shape),
                'checks': ['exact_parameter_count', 'frame_alignment', 'explicit_fft_reference',
                           'unfitted_nontraining_and_refit_guards',
                           'training_spectral_moments', 'scaler_fit_does_not_update_batchnorm',
                           'frozen_scaler', 'forward_backward_all_branches',
                           'branch_input_isolation', 'checkpoint_round_trip']}


def energy_model_preflight(device: str | torch.device = 'cpu') -> dict[str, Any]:
    """Check C1/C2A/C2D initialization, transforms, learning and serialization.

    Synthetic checks only: no recordings, installations, filesystem writes or
    experimental fitting. Run this on Kaggle before the real-data smoke fits.
    """
    baseline_checks = model_preflight(device)
    target = torch.device(device)
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(1907)
        x = torch.randn(4, 120, WINDOW_SAMPLES, device=target)
        input_mean = torch.linspace(-0.12, 0.12, 120, device=target).unsqueeze(-1)
        input_std = torch.linspace(0.7, 1.8, 120, device=target).unsqueeze(-1)
        offset = input_mean[:12] / input_std[:12]
        baseline = ThreeBranchC1().to(target)
        baseline.fit_spectral_scaler([x[:2], x[2:]], split='train')
        common_state = {key: value.clone() for key, value in baseline.state_dict().items()}
        models = {'C2A': ThreeBranchEnergy('amplitude').to(target),
                  'C2D': ThreeBranchEnergy('dynamic').to(target)}
        for name, model in models.items():
            loaded = model.load_state_dict(common_state, strict=False)
            assert loaded.missing_keys and all(key.startswith('energy_') for key in loaded.missing_keys)
            assert not loaded.unexpected_keys
            for key, value in common_state.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            assert model.count_params() == ENERGY_EXPECTED_PARAMETERS[name]
            assert model.parameter_breakdown()['energy_projection'] == ENERGY_BRANCH_PARAMETERS
            try:
                model.fit_energy_scaler([x], input_mean=input_mean, input_std=input_std, split='test')
            except ValueError as error:
                assert 'train split' in str(error)
            else:
                raise AssertionError('Energy scaler accepted the test split')
            statistics = model.fit_energy_scaler([x[:3], x[3:]], input_mean=input_mean,
                                                 input_std=input_std, split='train')
            assert statistics['frames'] == 28 and statistics['windows'] == 4
            raw = energy_log_features(x[:, :12], model.energy_input_offset, model.energy_mode).double()
            expected_mean = raw.mean(dim=(0, 2))
            expected_std = raw.permute(1, 0, 2).reshape(24, -1).std(-1, correction=0).clamp_min(1e-6)
            torch.testing.assert_close(model.energy_mean, expected_mean.float(), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(model.energy_std, expected_std.float(), rtol=2e-5, atol=2e-6)
            try:
                model.fit_energy_scaler([x], input_mean=input_mean, input_std=input_std, split='train')
            except RuntimeError as error:
                assert 'already fitted' in str(error)
            else:
                raise AssertionError('Energy scaler accepted accidental refitting')
        # Both added paths have identical initial weights, including the zero last layer.
        models['C2D'].energy_projection.load_state_dict(models['C2A'].energy_projection.state_dict())
        amplitude_input = models['C2A'].energy_features(x)
        torch.testing.assert_close(amplitude_input[:, :12], amplitude_input[:, 12:], rtol=0, atol=0)
        dynamic_input = models['C2D'].energy_features(x)
        torch.testing.assert_close(amplitude_input[:, :12], dynamic_input[:, :12], rtol=0, atol=0)
        assert not torch.allclose(dynamic_input[:, :12], dynamic_input[:, 12:])

        # Zero/constant physical EMG must remain finite, including input derivatives.
        zero_z = (-offset).expand(2, 12, WINDOW_SAMPLES).clone().requires_grad_(True)
        zero_features = energy_log_features(zero_z, offset, 'dynamic')
        assert bool(torch.isfinite(zero_features).all().item())
        torch.testing.assert_close(zero_features[:, 12:], torch.zeros_like(zero_features[:, 12:]))
        zero_features.sum().backward()
        assert bool(torch.isfinite(zero_z.grad).all().item())
        constant_z = (torch.full((2, 12, WINDOW_SAMPLES), 0.5, device=target) - offset)
        assert bool(torch.isfinite(energy_log_features(constant_z, offset, 'dynamic')).all().item())
        constant_model = ThreeBranchEnergy('dynamic').to(target)
        zero_all = -input_mean.unsqueeze(0) / input_std.unsqueeze(0)
        zero_all = zero_all.expand(2, 120, WINDOW_SAMPLES).clone()
        constant_stats = constant_model.fit_energy_scaler([zero_all], input_mean=input_mean,
                                                           input_std=input_std, split='train')
        assert constant_stats['constant_features'] == 24
        assert bool(torch.isfinite(constant_model.energy_features(zero_all)).all().item())

        # Verify physical-zero recovery against direct filtered-EMG/train-SD RMS.
        physical_scaled = x[:, :12] + models['C2D'].energy_input_offset
        first_frame = physical_scaled[:, :, :200]
        direct25 = first_frame[:, :, -50:].square().mean(-1).sqrt()
        direct100 = first_frame.square().mean(-1).sqrt()
        direct = torch.cat([torch.log(direct25 + 1e-6),
                            torch.log(direct25 + 1e-6) - torch.log(direct100 + 1e-6)], dim=1)
        torch.testing.assert_close(models['C2D'].energy_features(x, standardized=False)[:, :, 0], direct)
        # Altering samples after t=150ms cannot affect the first two frame inputs.
        future_changed = x.clone()
        future_changed[:, :12, 300:] += 7.0
        for model in models.values():
            original = model.energy_features(x, standardized=False)
            changed = model.energy_features(future_changed, standardized=False)
            torch.testing.assert_close(original[:, :, :2], changed[:, :, :2], rtol=0, atol=0)
        # Alter the first frame's history while preserving its last25ms: the
        # amplitude control cannot detect that alteration, but the ratio can.
        history_changed = x.clone()
        history_changed[:, :12, :150] += 5.0
        original_a = models['C2A'].energy_features(x, standardized=False)[:, :, 0]
        altered_a = models['C2A'].energy_features(history_changed, standardized=False)[:, :, 0]
        torch.testing.assert_close(original_a, altered_a, rtol=0, atol=0)
        original_d = models['C2D'].energy_features(x, standardized=False)[:, :, 0]
        altered_d = models['C2D'].energy_features(history_changed, standardized=False)[:, :, 0]
        torch.testing.assert_close(original_d[:, :12], altered_d[:, :12], rtol=0, atol=0)
        assert not torch.allclose(original_d[:, 12:], altered_d[:, 12:])

        baseline.eval()
        with torch.no_grad():
            expected_logits = baseline(x)
            for name, model in models.items():
                model.eval()
                outputs = model.forward_features(x, retain_sequences=True)
                torch.testing.assert_close(outputs['logits'], expected_logits, rtol=0, atol=0)
                assert outputs['energy_inputs'].shape == (4, 24, 7)

        # The zero last layer should learn immediately; the first layer becomes
        # active after the first optimizer step rather than being a dead add-on.
        labels = torch.tensor([0, 5, 10, 16], device=target)
        for name, model in models.items():
            frozen = {key: value.clone() for key, value in model.named_buffers()
                      if key.startswith(('energy_', 'spectral_'))}
            optimizer = torch.optim.SGD(model.energy_projection.parameters(), lr=0.01)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), labels).backward()
            first, last = model.energy_projection[0], model.energy_projection[-1]
            assert bool((first.weight.grad == 0).all().item()), name
            assert bool(torch.isfinite(last.weight.grad).all().item()), name
            assert bool(last.weight.grad.abs().sum().item() > 0), name
            optimizer.step()
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), labels).backward()
            assert bool(torch.isfinite(first.weight.grad).all().item()), name
            assert bool(first.weight.grad.abs().sum().item() > 0), name
            for key, value in model.named_buffers():
                if key in frozen:
                    torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
            with torch.no_grad():
                expected = model(x)
                checkpoint = checkpoint_io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                # Intentionally choose the opposite constructor mode: checkpoint
                # restoration must recover the saved transform and its buffers.
                opposite = 'dynamic' if model.energy_mode == 'amplitude' else 'amplitude'
                restored = ThreeBranchEnergy(opposite).to(target)
                restored.load_state_dict(torch.load(checkpoint, map_location=target, weights_only=True))
                restored.eval()
                assert restored.energy_mode == model.energy_mode
                torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
        return {'success': True, 'parameters': ENERGY_EXPECTED_PARAMETERS.copy(),
                'energy_branch_parameters': ENERGY_BRANCH_PARAMETERS, 'device': str(target),
                'torch_version': torch.__version__, 'C1_preflight': baseline_checks,
                'checks': ['shared_C1_state_exact', 'equal_energy_parameter_counts',
                           'paired_energy_projection_initialization', 'training_only_energy_scaler',
                           'direct_population_moments', 'amplitude_control_has_no_ratio_information',
                           'finite_zero_and_constant_inputs_and_zero_input_derivatives',
                           'physical_zero_restoration', 'no_future_frame_dependency',
                           'zero_residual_matches_C1_logits', 'residual_gradients_activate_after_one_step',
                           'frozen_scaler_buffers', 'checkpoint_restores_energy_mode_and_statistics']}


energy_preflight = energy_model_preflight


if __name__ == '__main__':
    import json
    print(json.dumps(energy_model_preflight('cuda' if torch.cuda.is_available() else 'cpu'), indent=2))
