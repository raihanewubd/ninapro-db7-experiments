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

    def forward_features(self, x: Tensor, *, retain_sequences: bool = False) -> dict[str, Any]:
        if x.ndim != 3 or x.shape[1:] != (120, WINDOW_SAMPLES):
            raise ValueError(f'C1 expects [B,120,800], received {tuple(x.shape)}')
        if not bool(self.spectral_fitted.item()):
            raise RuntimeError('Fit the spectral scaler using training windows before model(x)')
        frames = unfold_frames(x)
        waveform = self.waveform(frames[:, :12])
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


if __name__ == '__main__':
    import json
    print(json.dumps(model_preflight('cuda' if torch.cuda.is_available() else 'cpu'), indent=2))
