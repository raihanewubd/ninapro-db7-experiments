"""GPU population Differential Evolution for the DB7-032 JOPS adaptation.

This is a classification adaptation, not a reproduction of Yang et al.'s scalar
regression experiments. Structure search belongs to the caller. Each of twelve
channel modules retains four Hudgins antecedents and 17 consequent beliefs.
Every fitted quantity uses the x/y passed to train_de; test data must never be
passed there. No autograd or Adam is used by this engine.

DE jointly optimizes ordered internal references, rule weights, normalized
attribute weights, consequent beliefs, and the existing global ER fusion
weights. To make the large search practical, its initial population jitters a
training-data empirical rule-belief initialization. This initialization is an
explicit adaptation rather than a method attributed to the paper.
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

FAMILIES = ("MAV", "WL", "ZC", "SSC")
N_CHANNELS, N_CLASSES = 12, 17
ACTIVE_BITS = np.asarray(list(itertools.product(range(2), repeat=4)), dtype=np.int64)
GROUPS = ("reference_logits", "rule_logits", "attribute_logits", "belief_logits", "fusion_logits")


def make_layout(counts):
    """Create a compact parameter vector. No parameters are spent on padding."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape == (4,):
        counts = np.tile(counts, (12, 1))
    if counts.shape != (12, 4) or not np.all((counts >= 2) & (counts <= 4)):
        raise ValueError("counts must be four values, or 12x4 values, each in [2,4]")
    rules = np.prod(counts, axis=1)
    offset, sections = 0, {}
    sizes = [int(np.where(counts > 2, counts - 1, 0).sum()), int(rules.sum()),
             48, int(rules.sum()) * 17, 12]
    for name, size in zip(GROUPS, sizes):
        sections[name] = [offset, offset + size]
        offset += size
    return {"counts": counts.tolist(), "rules_per_channel": rules.tolist(),
            "total_rules": int(rules.sum()), "dimension": offset, "sections": sections,
            "feature_order": "family-major: MAV_CH01..12, WL_CH01..12, ZC_CH01..12, SSC_CH01..12"}


def _slice(layout, name):
    return slice(*layout["sections"][name])


def parameter_bounds(layout):
    lower = np.full(layout["dimension"], -3., np.float32)
    upper = np.full(layout["dimension"], 3., np.float32)
    lower[_slice(layout, "reference_logits")] = -4.
    upper[_slice(layout, "reference_logits")] = 4.
    lower[_slice(layout, "belief_logits")] = -12.
    upper[_slice(layout, "belief_logits")] = 3.
    return lower, upper


def _validate_xy(x, y=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 48 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a finite, nonempty Nx48 array in family-major order")
    if y is None:
        return x
    y = np.asarray(y, dtype=np.int64)
    if y.shape != (len(x),) or not np.all((y >= 0) & (y < 17)):
        raise ValueError("y must contain zero-based class IDs 0..16, one per window")
    return x, y


class _Context:
    """Reusable device constants and padded population inference, without grads."""
    def __init__(self, layout, endpoints, device, dtype=torch.float32):
        self.layout, self.device, self.dtype = layout, torch.device(device), dtype
        self.counts = np.asarray(layout["counts"], int)
        self.rules = np.asarray(layout["rules_per_channel"], int)
        self.max_rules = int(max(self.rules))
        self.endpoints = torch.as_tensor(endpoints, device=device, dtype=dtype)
        self.bits = torch.as_tensor(ACTIVE_BITS, device=device)
        radix = np.stack([np.prod(self.counts[:, i + 1:], axis=1) for i in range(4)], -1)
        self.radix = torch.as_tensor(radix, device=device)
        self.n = torch.as_tensor(self.counts, device=device)
        self.channels = torch.arange(12, device=device)[None, None, :, None]
        self.reference_offsets = np.r_[0, np.cumsum(np.where(self.counts > 2, self.counts - 1, 0).sum(1))]
        self.rule_offsets = np.r_[0, np.cumsum(self.rules)]

    def decode_channel(self, vector, c):
        """Decode only one active block; avoid GPU launches for frozen modules."""
        p = len(vector)
        refs = torch.full((p, 4, 4), torch.inf, device=self.device, dtype=self.dtype)
        gap = vector[:, _slice(self.layout, "reference_logits")]
        offset = int(self.reference_offsets[c])
        for f in range(4):
            n = int(self.counts[c, f])
            lo, hi = self.endpoints[c, f]
            refs[:, f, 0] = lo
            refs[:, f, n - 1] = hi
            if n > 2:
                fractions = gap[:, offset:offset + n - 1].softmax(-1)
                fractions = 1e-4 + (1 - (n - 1) * 1e-4) * fractions
                refs[:, f, 1:n - 1] = lo + (hi - lo) * fractions.cumsum(-1)[:, :-1]
                offset += n - 1
        first, last = self.rule_offsets[c:c + 2]
        beta = vector[:, _slice(self.layout, "belief_logits")].reshape(p, -1, 17)[:, first:last].softmax(-1)
        rules = vector[:, _slice(self.layout, "rule_logits")][:, first:last]
        attrs = vector[:, _slice(self.layout, "attribute_logits")].reshape(p, 12, 4)[:, c]
        attrs = (attrs - attrs.amax(-1, keepdim=True)).exp()
        fusion = vector[:, _slice(self.layout, "fusion_logits")].softmax(-1)
        return refs, beta, rules, attrs, fusion

    def decode(self, vector):
        p = len(vector)
        refs = torch.full((p, 12, 4, 4), torch.inf, device=self.device, dtype=self.dtype)
        gap = vector[:, _slice(self.layout, "reference_logits")]
        offset = 0
        for c in range(12):
            for f in range(4):
                n = int(self.counts[c, f])
                lo, hi = self.endpoints[c, f]
                refs[:, c, f, 0] = lo
                refs[:, c, f, n - 1] = hi
                if n > 2:
                    # Positive, normalized spacings guarantee strict ordering.
                    fractions = gap[:, offset:offset + n - 1].softmax(-1)
                    fractions = 1e-4 + (1 - (n - 1) * 1e-4) * fractions
                    refs[:, c, f, 1:n - 1] = lo + (hi - lo) * fractions.cumsum(-1)[:, :-1]
                    offset += n - 1
        beta = torch.zeros((p, 12, self.max_rules, 17), device=self.device, dtype=self.dtype)
        rule_logits = torch.full((p, 12, self.max_rules), -torch.inf, device=self.device, dtype=self.dtype)
        flat_beta = vector[:, _slice(self.layout, "belief_logits")].reshape(p, -1, 17)
        flat_rule = vector[:, _slice(self.layout, "rule_logits")]
        offset = 0
        for c, r in enumerate(self.rules):
            beta[:, c, :r] = flat_beta[:, offset:offset + r].softmax(-1)
            rule_logits[:, c, :r] = flat_rule[:, offset:offset + r]
            offset += r
        attr = vector[:, _slice(self.layout, "attribute_logits")].reshape(p, 12, 4)
        attr = (attr - attr.amax(-1, keepdim=True)).exp()
        fusion = vector[:, _slice(self.layout, "fusion_logits")].softmax(-1)
        return refs, beta, rule_logits, attr, fusion

    def encode(self, x, refs):
        # x: B,48 -> P,B,12,4. Count each internal reference at/below x.
        xx = x.reshape(-1, 4, 12).transpose(1, 2)[None]
        low = (xx[..., None] >= refs[:, None, :, :, 1:]).sum(-1)
        low = torch.minimum(low, self.n[None, None] - 2)
        expanded = refs[:, None].expand(-1, len(x), -1, -1, -1)
        a = expanded.gather(-1, low[..., None]).squeeze(-1)
        b = expanded.gather(-1, (low + 1)[..., None]).squeeze(-1)
        frac = ((xx - a) / (b - a)).clamp(0, 1)
        match = torch.where(self.bits[None, None, None] == 0,
                            1 - frac[..., None, :], frac[..., None, :])
        index = ((low[..., None, :] + self.bits[None, None, None]) *
                 self.radix[None, None, :, None, :]).sum(-1)
        valid = (match > 0).all(-1)
        return index, match, valid

    @staticmethod
    def er(activation, beliefs, dim):
        singleton = ((1 + activation[..., None] * (beliefs - 1)).prod(dim) -
                     (1 - activation).prod(dim)[..., None]).clamp_min(1e-12)
        return singleton / singleton.sum(-1, keepdim=True)

    def forward(self, x, decoded, channel_output=False):
        refs, beta, rules, attrs, fusion = decoded
        index, match, valid = self.encode(x, refs)
        population = torch.arange(len(refs), device=self.device)[:, None, None, None]
        scores = (match.clamp_min(1e-30).log() * attrs[:, None, :, None, :]).sum(-1)
        scores += rules[population, self.channels, index]
        activation = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        channel = self.er(activation, beta[population, self.channels, index], dim=3)
        p = self.er(fusion[:, None].expand(-1, len(x), -1), channel, dim=2)
        return (p, channel) if channel_output else p


def initialize_candidate(x, y, counts, seed=42):
    """Fit endpoints/initial references and empirical beliefs using fitting data only."""
    x, y = _validate_xy(x, y)
    layout = make_layout(counts)
    counts = np.asarray(layout["counts"])
    xx = x.reshape(-1, 4, 12).transpose(0, 2, 1)
    endpoints = np.percentile(xx, [5, 95], axis=0).transpose(1, 2, 0)
    span = np.maximum(endpoints[..., 1] - endpoints[..., 0],
                      np.maximum(np.max(np.abs(xx), axis=0) * 1e-6, 1e-12))
    equal = endpoints[..., 1] <= endpoints[..., 0]
    endpoints[..., 0] -= np.where(equal, span / 2, 0)
    endpoints[..., 1] += np.where(equal, span / 2, 0)
    vector = np.zeros(layout["dimension"], np.float32)
    gaps = []
    for c in range(12):
        for f in range(4):
            n = int(counts[c, f])
            if n > 2:
                target = np.percentile(xx[:, c, f], np.linspace(5, 95, n))
                fractions = np.maximum(np.diff(target), span[c, f] * 1e-3)
                fractions /= fractions.sum()
                raw = np.log(fractions)
                gaps.extend(raw - raw.mean())
    vector[_slice(layout, "reference_logits")] = np.clip(gaps, -4, 4)
    # Empirical initial consequents: soft counts from each window's rule activation.
    # Every class receives a half-prior pseudo-count per rule; unsupported rules
    # therefore retain a smoothed fitting-data class prior instead of test knowledge.
    prior = np.bincount(y, minlength=17).astype(float) + .5
    prior /= prior.sum()
    context = _Context(layout, endpoints, "cpu", torch.float64)
    with torch.no_grad():
        decoded = context.decode(torch.as_tensor(vector[None], dtype=torch.float64))
        refs = decoded[0]
        empirical = [np.tile(.5 * prior, (r, 1)) for r in layout["rules_per_channel"]]
        for begin in range(0, len(x), 512):
            index, match, _ = context.encode(torch.as_tensor(x[begin:begin + 512]), refs)
            ix = index[0].numpy()
            weight = match[0].prod(-1).numpy()
            for c in range(12):
                np.add.at(empirical[c], (ix[:, c].ravel(),
                          np.repeat(y[begin:begin + 512], 16)), weight[:, c].ravel())
    belief = np.concatenate([v / v.sum(-1, keepdims=True) for v in empirical])
    vector[_slice(layout, "belief_logits")] = np.log(np.maximum(belief, 1e-12)).ravel()
    lo, hi = parameter_bounds(layout)
    vector = np.clip(vector, lo, hi)
    return {"vector": vector, "initial_vector": vector.copy(), "layout": layout,
            "endpoints": endpoints, "seed": int(seed), "history": [],
            "initialization": "training empirical fuzzy-rule class counts, 0.5 prior pseudo-count per rule",
            "objective": "mean multiclass cross entropy on fitting windows", "loss": None}


def _score_vectors(vectors, x, y, context, sample_batch=256, population_chunk=4):
    result = []
    with torch.no_grad():
        for first in range(0, len(vectors), population_chunk):
            decoded = context.decode(vectors[first:first + population_chunk])
            total = torch.zeros(len(decoded[0]), device=context.device, dtype=torch.float64)
            for begin in range(0, len(x), sample_batch):
                p = context.forward(x[begin:begin + sample_batch], decoded)
                yy = y[begin:begin + sample_batch]
                ce = -p.gather(-1, yy[None, :, None].expand(len(p), -1, -1)).squeeze(-1).clamp_min(1e-12).log()
                total += ce.double().sum(-1)
            result.append(total / len(x))
    return torch.cat(result)


def train_de_global(x, y, counts, seed=42, device="cpu", population=12, generations=40,
             initial=None, sample_batch=256, population_chunk=4, mutation=.5,
             crossover=.9, jitter=.15, progress=None):
    """Optimize full fitting-set CE with bounded DE/rand/1/bin and elitist selection.

    Population and sample chunking only control memory, not which observations
    contribute to fitness. All individuals face the identical complete objective.
    The first population member is the unperturbed empirical initialization.
    Other members get independent Gaussian parameter perturbations. The seed
    controls genuine parameter initialization and subsequent DE proposals.
    """
    x, y = _validate_xy(x, y)
    if population < 4 or generations < 0 or sample_batch < 1 or population_chunk < 1:
        raise ValueError("population>=4, generations>=0 and positive chunk sizes required")
    if not 0 < mutation <= 2 or not 0 <= crossover <= 1 or jitter <= 0:
        raise ValueError("invalid DE controls")
    candidate = initialize_candidate(x, y, counts, seed) if initial is None else dict(initial)
    layout = make_layout(counts)
    if candidate["layout"]["counts"] != layout["counts"]:
        raise ValueError("A warm start must have the same structure; reinitialize changed counts")
    candidate["layout"] = layout
    context = _Context(layout, candidate["endpoints"], device)
    xx = torch.as_tensor(x, device=device, dtype=torch.float32)
    yy = torch.as_tensor(y, device=device, dtype=torch.long)
    lo, hi = [torch.as_tensor(v, device=device) for v in parameter_bounds(layout)]
    rng = np.random.default_rng(seed)
    base = np.asarray(candidate["vector"], np.float32)
    init = np.tile(base, (population, 1)) + rng.normal(0, jitter, (population, len(base))).astype(np.float32)
    init[0] = base
    vectors = torch.as_tensor(init, device=device).clamp(lo, hi)
    scores = _score_vectors(vectors, xx, yy, context, sample_batch, population_chunk)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite initial population objective")
    initial_vector = np.asarray(base).copy()
    start = time.monotonic()
    history = [{"generation": 0, "best_training_ce": float(scores.min()),
                "mean_training_ce": float(scores.mean()), "accepted": 0, "seconds": 0.}]
    for generation in range(1, generations + 1):
        donors = np.stack([rng.choice(np.delete(np.arange(population), i), 3, replace=False)
                           for i in range(population)])
        donor = torch.as_tensor(donors, device=device)
        mutant = vectors[donor[:, 0]] + mutation * (vectors[donor[:, 1]] - vectors[donor[:, 2]])
        # Reflection repairs bounded parameters without discarding diversity.
        span = hi - lo
        repaired = lo + span - torch.abs(torch.remainder(mutant - lo, 2 * span) - span)
        mask = rng.random((population, len(base))) < crossover
        mask[np.arange(population), rng.integers(0, len(base), population)] = True
        trial = torch.where(torch.as_tensor(mask, device=device), repaired, vectors)
        trial_scores = _score_vectors(trial, xx, yy, context, sample_batch, population_chunk)
        if not torch.isfinite(trial_scores).all():
            raise FloatingPointError("Nonfinite DE trial objective")
        accepted = trial_scores <= scores
        vectors = torch.where(accepted[:, None], trial, vectors)
        scores = torch.where(accepted, trial_scores, scores)
        row = {"generation": generation, "best_training_ce": float(scores.min()),
               "mean_training_ce": float(scores.mean()), "accepted": int(accepted.sum()),
               "seconds": time.monotonic() - start}
        history.append(row)
        if progress is not None:
            progress(row)
    best = int(scores.argmin())
    vector = vectors[best].cpu().numpy()
    candidate.update(vector=vector, initial_vector=initial_vector, loss=float(scores[best]),
                     seed=int(seed), history=history, population=int(population), generations=int(generations),
                     mutation=float(mutation), crossover=float(crossover), jitter=float(jitter),
                     fitting_windows=len(x), device=str(device), optimizer="DE/rand/1/bin",
                     elapsed_seconds=time.monotonic() - start)
    candidate["parameter_changes"] = {
        name: float(np.max(np.abs(vector[_slice(layout, name)] - initial_vector[_slice(layout, name)]), initial=0))
        for name in GROUPS}
    return candidate


def _block_indices(layout):
    """One compact, jointly optimized block per channel plus a fusion block."""
    counts = np.asarray(layout["counts"])
    rules = np.asarray(layout["rules_per_channel"])
    ref_offset, rule_offset = 0, 0
    blocks = []
    for c in range(12):
        ref_size = int(np.where(counts[c] > 2, counts[c] - 1, 0).sum())
        parts = []
        for name, start, size in (("reference_logits", ref_offset, ref_size),
                                  ("rule_logits", rule_offset, rules[c]),
                                  ("attribute_logits", c * 4, 4),
                                  ("belief_logits", rule_offset * 17, rules[c] * 17)):
            first = layout["sections"][name][0] + start
            parts.extend(range(first, first + size))
        blocks.append(np.asarray(parts, dtype=np.int64))
        ref_offset += ref_size
        rule_offset += rules[c]
    blocks.append(np.arange(*layout["sections"]["fusion_logits"], dtype=np.int64))
    return blocks


def _channel_forward(context, x, decoded, c):
    """Recompute only the channel under optimization; all others are cached."""
    refs, beta, rules, attrs, _ = decoded
    if refs.ndim == 4:
        refs, beta, rules, attrs = refs[:, c], beta[:, c], rules[:, c], attrs[:, c]
    xx = x.reshape(-1, 4, 12)[:, :, c][None]
    low = (xx[..., None] >= refs[:, None, :, 1:]).sum(-1)
    low = torch.minimum(low, context.n[None, None, c] - 2)
    expanded = refs[:, None].expand(-1, len(x), -1, -1)
    a = expanded.gather(-1, low[..., None]).squeeze(-1)
    b = expanded.gather(-1, (low + 1)[..., None]).squeeze(-1)
    frac = ((xx - a) / (b - a)).clamp(0, 1)
    match = torch.where(context.bits[None, None] == 0, 1 - frac[..., None, :], frac[..., None, :])
    index = ((low[..., None, :] + context.bits[None, None]) * context.radix[None, None, c, None]).sum(-1)
    valid = (match > 0).all(-1)
    pop = torch.arange(len(refs), device=context.device)[:, None, None]
    score = (match.clamp_min(1e-30).log() * attrs[:, None, None]).sum(-1)
    score += rules[pop, index]
    activation = score.masked_fill(~valid, -torch.inf).softmax(-1)
    return context.er(activation, beta[pop, index], dim=2)


def _score_block(vectors, x, y, context, cached_channels, channel,
                 sample_batch=256, population_chunk=4):
    result = []
    with torch.no_grad():
        for first in range(0, len(vectors), population_chunk):
            v = vectors[first:first + population_chunk]
            decoded = context.decode_channel(v, channel) if channel < 12 else None
            fusion = v[:, _slice(context.layout, "fusion_logits")].softmax(-1)
            total = torch.zeros(len(v), device=context.device, dtype=torch.float64)
            for begin in range(0, len(x), sample_batch):
                cache = cached_channels[begin:begin + sample_batch][None].expand(len(v), -1, -1, -1)
                if channel < 12:
                    cache = cache.clone()
                    cache[:, :, channel] = _channel_forward(context, x[begin:begin + sample_batch], decoded, channel)
                p = context.er(fusion[:, None].expand(-1, cache.shape[1], -1), cache, dim=2)
                yy = y[begin:begin + sample_batch]
                ce = -p.gather(-1, yy[None, :, None].expand(len(p), -1, -1)).squeeze(-1).clamp_min(1e-12).log()
                total += ce.double().sum(-1)
            result.append(total / len(x))
    return torch.cat(result)


def _report_changes(candidate):
    before, after = decoded_parameters(candidate, "initial"), decoded_parameters(candidate, "trained")
    counts = np.asarray(candidate["layout"]["counts"])
    ref_change = [abs(after["references"][c, f, level] - before["references"][c, f, level])
                  for c in range(12) for f in range(4) for level in range(1, counts[c, f] - 1)]
    changes = {"reference_positions": float(max(ref_change, default=0.)),
               "rule_weights": float(np.max(np.abs(after["rule_weights"] - before["rule_weights"]))),
               "attribute_weights": float(np.max(np.abs(after["attribute_weights"] - before["attribute_weights"]))),
               "consequent_beliefs": float(np.max(np.abs(after["beliefs"] - before["beliefs"]))),
               "fusion_weights": float(np.max(np.abs(after["fusion_weights"] - before["fusion_weights"])))}
    return changes


def train_de(x, y, counts, seed=42, device="cpu", population=12, generations=4,
             cycles=2, initial=None, sample_batch=256, population_chunk=4,
             mutation=.5, crossover=.9, jitter=.15, progress=None):
    """Budgeted cooperative DE: twelve channel blocks and one fusion block.

    Each block's vector is optimized by actual DE/rand/1/bin; its fitness is the
    full classifier's CE on every fitting window. Cache the other eleven channel
    beliefs and refresh the changed channel after accepting its best candidate.
    This computational adaptation is not a claim of reproducing paper JOPS's
    global-vector optimizer. No Adam, gradients, validation or test fitness.
    The incumbent is included in every block population, preserving elitism.
    """
    x, y = _validate_xy(x, y)
    if population < 4 or generations < 0 or cycles < 1 or sample_batch < 1 or population_chunk < 1:
        raise ValueError("population>=4, generations>=0, cycles>=1, positive chunks required")
    if not 0 < mutation <= 2 or not 0 <= crossover <= 1 or jitter <= 0:
        raise ValueError("invalid DE controls")
    candidate = initialize_candidate(x, y, counts, seed) if initial is None else dict(initial)
    layout = make_layout(counts)
    if candidate["layout"]["counts"] != layout["counts"]:
        raise ValueError("Warm starts require the same structure")
    candidate["layout"] = layout
    context = _Context(layout, candidate["endpoints"], device)
    xx = torch.as_tensor(x, device=device, dtype=torch.float32)
    yy = torch.as_tensor(y, device=device, dtype=torch.long)
    bounds = [torch.as_tensor(v, device=device) for v in parameter_bounds(layout)]
    rng = np.random.default_rng(seed)
    current = torch.as_tensor(np.asarray(candidate["vector"]), device=device, dtype=torch.float32).clone()
    initial_vector = current.cpu().numpy().copy()
    blocks = _block_indices(layout)
    start = time.monotonic()
    evaluations = 0
    with torch.no_grad():
        decoded = context.decode(current[None])
        cached = torch.cat([context.forward(xx[i:i + sample_batch], decoded, channel_output=True)[1][0]
                            for i in range(0, len(xx), sample_batch)])
        current_loss = float(_score_block(current[None], xx, yy, context, cached, 12, sample_batch, 1)[0])
        initial_loss = current_loss
        initial_p = context.er(decoded[4].expand(len(xx), -1), cached, dim=1)
        initial_accuracy = float((initial_p.argmax(-1) == yy).float().mean())
        evaluations += 1
        history = [{"cycle": 0, "block": "initial", "generation": 0,
                    "best_training_ce": current_loss, "mean_training_ce": current_loss,
                    "accepted": 0, "seconds": time.monotonic() - start}]
        for cycle in range(1, cycles + 1):
            # Fixed channel order is fully reproducible and recorded explicitly.
            for channel, index_np in enumerate(blocks):
                ix = torch.as_tensor(index_np, device=device)
                base = current[ix]
                lo, hi = (v[ix] for v in bounds)
                vectors = base[None].repeat(population, 1)
                perturbation = torch.as_tensor(rng.normal(0, jitter, vectors.shape).astype(np.float32), device=device)
                vectors[1:] += perturbation[1:]
                vectors = vectors.clamp(lo, hi)

                def score(v):
                    full = current[None].repeat(len(v), 1)
                    full[:, ix] = v
                    return _score_block(full, xx, yy, context, cached, channel, sample_batch, population_chunk)

                scores = score(vectors)
                evaluations += population
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite block initial population fitness")
                for generation in range(1, generations + 1):
                    donors = np.stack([rng.choice(np.delete(np.arange(population), i), 3, replace=False)
                                       for i in range(population)])
                    donor = torch.as_tensor(donors, device=device)
                    mutant = vectors[donor[:, 0]] + mutation * (vectors[donor[:, 1]] - vectors[donor[:, 2]])
                    span = hi - lo
                    repaired = lo + span - torch.abs(torch.remainder(mutant - lo, 2 * span) - span)
                    mask = rng.random(vectors.shape) < crossover
                    mask[np.arange(population), rng.integers(0, len(base), population)] = True
                    trial = torch.where(torch.as_tensor(mask, device=device), repaired, vectors)
                    trial_scores = score(trial)
                    evaluations += population
                    if not torch.isfinite(trial_scores).all():
                        raise FloatingPointError("Nonfinite block DE trial fitness")
                    accepted = trial_scores <= scores
                    vectors = torch.where(accepted[:, None], trial, vectors)
                    scores = torch.where(accepted, trial_scores, scores)
                    row = {"cycle": cycle, "block": f"channel_{channel+1:02}" if channel < 12 else "fusion",
                           "generation": generation, "best_training_ce": float(scores.min()),
                           "mean_training_ce": float(scores.mean()), "accepted": int(accepted.sum()),
                           "seconds": time.monotonic() - start}
                    history.append(row)
                    if progress is not None:
                        progress(row)
                winner = int(scores.argmin())
                current[ix] = vectors[winner]
                current_loss = float(scores[winner])
                if channel < 12:
                    decoded = context.decode_channel(current[None], channel)
                    for i in range(0, len(xx), sample_batch):
                        cached[i:i + sample_batch, channel] = _channel_forward(context, xx[i:i + sample_batch], decoded, channel)[0]
        # An independent full forward pass must agree with the cached objective.
        full_loss = float(_score_vectors(current[None], xx, yy, context, sample_batch, 1)[0])
        final_fusion = current[_slice(layout, "fusion_logits")].softmax(-1)
        final_p = context.er(final_fusion[None].expand(len(xx), -1), cached, dim=1)
        final_accuracy = float((final_p.argmax(-1) == yy).float().mean())
        if abs(full_loss - current_loss) > 3e-6:
            raise AssertionError(("Cached/full objective mismatch", current_loss, full_loss))
    candidate.update(vector=current.cpu().numpy(), initial_vector=initial_vector, loss=full_loss,
                     seed=int(seed), history=history, population=int(population), generations=int(generations),
                     cycles=int(cycles), mutation=float(mutation), crossover=float(crossover), jitter=float(jitter),
                     fitting_windows=len(x), device=str(device), optimizer="cooperative block DE/rand/1/bin",
                     block_order=[f"channel_{c:02}" for c in range(1, 13)] + ["fusion"],
                     searched_parameter_groups=["reference_positions", "rule_weights", "attribute_weights", "consequent_beliefs", "fusion_weights"],
                     internal_reference_degrees=int(np.maximum(np.asarray(layout["counts"]) - 2, 0).sum()),
                     fitness_evaluations=int(evaluations + 1), elapsed_seconds=time.monotonic() - start,
                     initial_training_ce=initial_loss, initial_training_accuracy=initial_accuracy,
                     final_training_ce=full_loss, final_training_accuracy=final_accuracy,
                     training_ce_reduction=initial_loss - full_loss,
                     budget_exhausted=True, convergence_claimed=False)
    candidate["raw_parameter_changes"] = {
        name: float(np.max(np.abs(candidate["vector"][_slice(layout, name)] - initial_vector[_slice(layout, name)]), initial=0))
        for name in GROUPS}
    candidate["parameter_changes"] = _report_changes(candidate)
    candidate["unchanged_parameter_groups"] = [k for k, v in candidate["parameter_changes"].items() if v == 0]
    return candidate


def predict(x, candidate, device="cpu", batch_size=512, return_channels=False):
    x = _validate_xy(x)
    context = _Context(candidate["layout"], candidate["endpoints"], device)
    outputs, channels = [], []
    with torch.no_grad():
        vector = torch.as_tensor(np.asarray(candidate["vector"])[None], device=device, dtype=torch.float32)
        decoded = context.decode(vector)
        for begin in range(0, len(x), batch_size):
            xx = torch.as_tensor(x[begin:begin + batch_size], device=device, dtype=torch.float32)
            value = context.forward(xx, decoded, channel_output=return_channels)
            if return_channels:
                value, channel = value
                channels.append(channel[0].cpu().numpy())
            outputs.append(value[0].cpu().numpy())
    p = np.concatenate(outputs)
    if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1, atol=2e-6):
        raise FloatingPointError("Invalid class probabilities")
    return (p, np.concatenate(channels)) if return_channels else p


def decoded_parameters(candidate, state="trained"):
    key = "initial_vector" if state == "initial" else "vector"
    context = _Context(candidate["layout"], candidate["endpoints"], "cpu", torch.float64)
    with torch.no_grad():
        refs, beliefs, rules, attributes, fusion = context.decode(
            torch.as_tensor(np.asarray(candidate[key])[None], dtype=torch.float64))
    return {"references": refs[0].numpy(), "beliefs": beliefs[0].numpy(),
            "rule_weights": rules[0].exp().numpy(), "attribute_weights": attributes[0].numpy(),
            "fusion_weights": fusion[0].numpy()}


def export_rules(folder, candidate):
    """Persist interpretable initial/final rules, references, weights and state."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(candidate["layout"]["counts"])
    rules, references, weights = [], [], []
    for state in ("initial", "trained"):
        d = decoded_parameters(candidate, state)
        for c in range(12):
            weights.append({"state": state, "channel": c + 1, "fusion_weight": d["fusion_weights"][c]})
            for f, name in enumerate(FAMILIES):
                for level in range(counts[c, f]):
                    references.append({"state": state, "channel": c + 1, "attribute": name,
                                       "reference_count": counts[c, f], "level": level + 1,
                                       "reference_value": d["references"][c, f, level],
                                       "attribute_weight": d["attribute_weights"][c, f],
                                       "fitted_endpoint": level in (0, counts[c, f] - 1)})
            for r, bits in enumerate(itertools.product(*(range(n) for n in counts[c]))):
                row = {"state": state, "channel": c + 1, "rule": r + 1,
                       "rule_weight": d["rule_weights"][c, r]}
                for f, name in enumerate(FAMILIES):
                    row.update({name + "_level": bits[f] + 1,
                                name + "_reference": d["references"][c, f, bits[f]],
                                name + "_attribute_weight": d["attribute_weights"][c, f]})
                row.update({f"belief_G{k+1}": d["beliefs"][c, r, k] for k in range(17)})
                rules.append(row)
    for name, rows in [("initial_trained_rules.csv.gz", rules),
                       ("initial_trained_references.csv", references),
                       ("initial_trained_fusion_weights.csv", weights),
                       ("training_history.csv", candidate["history"])]:
        opener = gzip.open if name.endswith(".gz") else open
        with opener(folder / name, "wt", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(folder / "model.npz", vector=candidate["vector"],
                        initial_vector=candidate["initial_vector"], endpoints=candidate["endpoints"])
    metadata = {k: v for k, v in candidate.items() if k not in ("vector", "initial_vector", "endpoints")}
    (folder / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return candidate.get("parameter_changes", {})


def load_candidate(folder):
    folder = Path(folder)
    result = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    with np.load(folder / "model.npz", allow_pickle=False) as data:
        result.update({name: data[name] for name in ("vector", "initial_vector", "endpoints")})
    return result


save_candidate = export_rules


def _dense_numpy_prediction(x, candidate):
    """Independent Cartesian dense activation and complete-belief ER reference."""
    d = decoded_parameters(candidate)
    xx = np.asarray(x).reshape(-1, 4, 12).transpose(0, 2, 1)
    channels = []
    for c, counts in enumerate(candidate["layout"]["counts"]):
        memberships = []
        for f, n in enumerate(counts):
            refs = d["references"][c, f, :n]
            v = xx[:, c, f]
            low = np.clip(np.searchsorted(refs, v, side="right") - 1, 0, n - 2)
            frac = np.clip((v - refs[low]) / (refs[low + 1] - refs[low]), 0, 1)
            m = np.zeros((len(v), n))
            m[np.arange(len(v)), low] = 1 - frac
            m[np.arange(len(v)), low + 1] = frac
            memberships.append(m)
        bits = np.asarray(list(itertools.product(*(range(n) for n in counts))))
        a = np.ones((len(x), len(bits)))
        for f in range(4):
            a *= memberships[f][:, bits[:, f]] ** d["attribute_weights"][c, f]
        a *= d["rule_weights"][c, :len(bits)]
        a /= a.sum(-1, keepdims=True)
        beta = d["beliefs"][c, :len(bits)]
        value = np.prod(1 + a[..., None] * (beta[None] - 1), axis=1) - np.prod(1 - a, axis=1)[..., None]
        value = np.maximum(value, 1e-12)
        channels.append(value / value.sum(-1, keepdims=True))
    channel = np.stack(channels, axis=1)
    fusion = d["fusion_weights"][None, :, None]
    value = np.prod(1 + fusion * (channel - 1), axis=1) - np.prod(1 - fusion, axis=1)
    value = np.maximum(value, 1e-12)
    return value / value.sum(-1, keepdims=True)


def self_test():
    """CPU numerical parity, chunk equivalence, determinism and actual DE updates."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(32032)
    x = rng.normal(size=(51, 48))
    y = np.arange(len(x)) % 17
    x += y[:, None] / 8
    maximum_error = 0.
    for counts in ([3, 3, 3, 3], [2, 3, 4, 2]):
        model = initialize_candidate(x, y, counts)
        lo, hi = parameter_bounds(model["layout"])
        model["vector"] = np.clip(model["vector"] + rng.normal(0, .2, len(lo)), lo, hi)
        context = _Context(model["layout"], model["endpoints"], "cpu", torch.float64)
        with torch.no_grad():
            decoded = context.decode(torch.as_tensor(model["vector"][None], dtype=torch.float64))
            actual = context.forward(torch.as_tensor(x), decoded)[0].numpy()
        expected = _dense_numpy_prediction(x, model)
        error = float(np.max(np.abs(actual - expected)))
        maximum_error = max(maximum_error, error)
        assert error < 1e-9, (counts, error)
        refs = decoded_parameters(model)["references"]
        for c in range(12):
            for f in range(4):
                assert np.all(np.diff(refs[c, f, :counts[f]]) > 0)
        # Compare the whole-data objective with sample and population chunking.
        vectors = torch.as_tensor(np.stack([model["vector"], model["vector"] + .02]), dtype=torch.float64)
        one = _score_vectors(vectors, torch.as_tensor(x), torch.as_tensor(y), context, 51, 2)
        chunked = _score_vectors(vectors, torch.as_tensor(x), torch.as_tensor(y), context, 7, 1)
        assert torch.allclose(one, chunked, atol=1e-12, rtol=1e-12)
    # Deliberately weak initialization tests DE learning, not an already optimal
    # empirical initializer. It is only used in this synthetic test.
    model = initialize_candidate(x, y, [3] * 4)
    model["vector"][_slice(model["layout"], "belief_logits")] = np.log(1 / 17)
    fitted = train_de(x, y, [3] * 4, seed=321, population=6, generations=2, cycles=1, initial=model,
                      sample_batch=26, population_chunk=3, jitter=.25)
    assert fitted["loss"] < np.log(17), fitted["loss"]
    assert all(v > 0 for v in fitted["parameter_changes"].values()), fitted["parameter_changes"]
    best = np.array([v["best_training_ce"] for v in fitted["history"]])
    assert np.all(np.diff(best) <= 1e-10)
    repeated = train_de(x, y, [3] * 4, seed=321, population=6, generations=2, cycles=1, initial=model,
                        sample_batch=26, population_chunk=3, jitter=.25)
    assert np.array_equal(fitted["vector"], repeated["vector"])
    result = {"passed": True, "dense_sparse_max_abs_error": maximum_error,
              "chunked_full_objective_equal": True, "deterministic": True,
              "synthetic_loss": fitted["loss"], "uniform_loss": float(np.log(17)),
              "all_five_parameter_groups_updated": fitted["parameter_changes"]}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
