"""DB7-032: nested repetition validation for the hierarchical JOPS adaptation.

Only fitting repetitions reach train_de. Candidate counts are selected from four
held-training-repetition folds. The test split never enters structure search.
This is an exploratory classification adaptation, not a paper reproduction.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import db7_030_hudgins as features
from db7_031_brb import select_ratio, metrics, write
import db7_032_jops_engine as engine

SEEDS = (42, 43, 44)
POPULATION, GENERATIONS, CYCLES = 12, 4, 2
CANDIDATES = 5
PARAMETER_GROUPS = ('reference_positions', 'rule_weights', 'attribute_weights',
                    'consequent_beliefs', 'fusion_weights')


def rank_key(record):
    """Prespecified validation ranking; no test quantities are accepted here."""
    return (-record['validation_accuracy'], record['validation_nll'],
            record['total_rules'], record['candidate_id'])


def pareto_archive(records):
    """SOHS-style archive: no worse accuracy and no more refs in any attribute."""
    archive = []
    for a in records:
        ac = np.asarray(a['counts'])
        dominated = False
        for b in records:
            if a['candidate_id'] == b['candidate_id']:
                continue
            bc = np.asarray(b['counts'])
            no_worse = b['validation_accuracy'] >= a['validation_accuracy']
            no_larger = np.all(bc <= ac)
            strict = b['validation_accuracy'] > a['validation_accuracy'] or np.any(bc < ac)
            if no_worse and no_larger and strict:
                dominated = True
                break
        if not dominated:
            archive.append(a)
    return sorted(archive, key=rank_key)


def propose_structure(records, rng):
    """Move counts toward a better archived structure, with bounded exploration.

    A 0.25 probability applies per attribute. Matching counts can mutate by one
    so the archive does not prevent exploration. Candidates must be distinct.
    This is a disclosed bounded SOHS-inspired proposal, not exhaustive search.
    """
    archive = pareto_archive(records)
    seen = {tuple(np.asarray(r['counts']).ravel()) for r in records}
    for attempt in range(200):
        a = archive[int(rng.integers(len(archive)))]
        b = records[int(rng.integers(len(records)))]
        better, worse = sorted([a, b], key=rank_key)
        source = np.asarray(worse['counts'], dtype=int)
        target = np.asarray(better['counts'], dtype=int)
        candidate = source.copy()
        moving = rng.random((12, 4)) < .25
        direction = np.sign(target - source)
        tied = direction == 0
        direction[tied] = rng.choice([-1, 1], int(tied.sum()))
        candidate = np.clip(candidate + moving * direction, 2, 4)
        if tuple(candidate.ravel()) not in seen:
            return candidate, {'kind': 'bounded_add_prune', 'better_candidate': better['candidate_id'],
                               'source_candidate': worse['candidate_id'], 'attempt': attempt + 1,
                               'changed_attributes': int((candidate != source).sum()), 'phi': .25}
    # Deterministic fallback avoids a duplicate even when clipping removes moves.
    base = np.asarray(archive[0]['counts'], dtype=int)
    for attribute in rng.permutation(48):
        for step in (-1, 1):
            candidate = base.copy()
            candidate.flat[attribute] = np.clip(candidate.flat[attribute] + step, 2, 4)
            if tuple(candidate.ravel()) not in seen:
                return candidate, {'kind': 'single_attribute_fallback', 'source_candidate': archive[0]['candidate_id']}
    raise RuntimeError('Unable to produce a distinct structure')


def nll(y, p):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def fit_model(x, y, counts, seed, device):
    return engine.train_de(x, y, counts, seed=seed, device=device,
                           population=POPULATION, generations=GENERATIONS, cycles=CYCLES,
                           sample_batch=2048, population_chunk=4)


def initial_model(candidate):
    return {**candidate, 'vector': candidate['initial_vector']}


def save_fit_diagnostics(folder, candidate):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate['history']).to_csv(folder / 'optimization_history.csv', index=False)
    # Initial/final rules are large, so the complete exports are limited to final
    # refits. Every validation fit retains its parameter-change and runtime audit.
    write(folder / 'fit_audit.json', {
        'layout': candidate['layout'], 'seed': candidate['seed'],
        'parameter_changes': candidate['parameter_changes'],
        'fitting_endpoints': np.asarray(candidate['endpoints']).tolist(),
        'population': POPULATION, 'generations_per_block': GENERATIONS, 'cycles': CYCLES,
        'optimizer': candidate['optimizer'], 'fitness_evaluations': candidate['fitness_evaluations'],
        'elapsed_seconds': candidate['elapsed_seconds'], 'budget_exhausted': candidate['budget_exhausted'],
        'convergence_claimed': False, 'unchanged_parameter_groups': candidate['unchanged_parameter_groups'],
        'raw_parameter_changes': candidate['raw_parameter_changes'],
        'test_used_for_fitting': False})


def run_subject(raw, subject, out, parent, device):
    import torch
    started = time.monotonic()
    folder = out / f'S{subject:02}'
    folder.mkdir(parents=True, exist_ok=True)
    runs, raw_source = features.load_subject(raw, subject)
    old = parent[str(subject)]
    assert raw_source['raw_emg_sha256'] == old['raw_emg_sha256'], 'Raw input differs from DB7-030'
    prepared, audit = [], []
    for held in features.TRAIN_REPS:
        allowed = tuple(r for r in features.TRAIN_REPS if r != held)
        ratio, screen = select_ratio(runs, allowed)
        scale = features.threshold_scale(runs, allowed)
        x, meta = features.extract(runs, subject, 'train', scale * ratio)
        rep = meta.native_repetition.to_numpy()
        y = meta.gesture.to_numpy(int) - 1
        ti, vi = np.isin(rep, allowed), rep == held
        assert not np.any(ti & vi) and np.all(ti | vi)
        prepared.append((held, x[ti], y[ti], x[vi], y[vi]))
        audit.append({'held_repetition': held, 'fitting_repetitions': allowed,
                      'threshold_ratio': ratio, 'nested_threshold_scores': screen,
                      'threshold_scale': scale.tolist(), 'train_windows': int(ti.sum()),
                      'validation_windows': int(vi.sum())})
        print(f'S{subject:02} {device}: prepared held repetition {held}', flush=True)
    scale = features.threshold_scale(runs)
    assert np.allclose(scale, old['threshold_scale_final'], rtol=1e-10, atol=1e-15)
    xtrain, train_meta = features.extract(runs, subject, 'train', scale * old['threshold_ratio'])
    ytrain = train_meta.gesture.to_numpy(int) - 1
    write(folder / 'fold_audit.json', {'raw_source': raw_source, 'folds': audit,
          'final_threshold_ratio': old['threshold_ratio'], 'final_fitting_repetitions': features.TRAIN_REPS,
          'test_repetitions': features.TEST_REPS, 'test_used_for_selection': False})

    # Freeze all structures and final parameters for this subject before reading
    # any test feature/label arrays. The raw recording naturally contains all reps.
    final_models = []
    for seed in SEEDS:
        seed_dir = folder / f'seed{seed}'
        seed_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(seed + subject * 10000)
        records = []
        for candidate_id in range(CANDIDATES):
            if candidate_id < 2:
                count = 3 if candidate_id == 0 else 2
                counts = np.full((12, 4), count, dtype=int)
                proposal = {'kind': f'prespecified_all_{count}'}
            else:
                counts, proposal = propose_structure(records, rng)
            candidate_dir = seed_dir / f'candidate{candidate_id:02}'
            candidate_dir.mkdir(exist_ok=True)
            fold_rows = []
            for held, x, y, v, vy in prepared:
                fit_start = time.monotonic()
                # Common optimization seed across structures for the same fold;
                # dimensions differ, so this is not identical initialization.
                fit_seed = seed + subject * 10000 + held * 100
                fitted = fit_model(x, y, counts, fit_seed, device)
                p = engine.predict(v, fitted, device=device)
                initial_p = engine.predict(v, initial_model(fitted), device=device)
                metric, _ = metrics(vy, p)
                save_fit_diagnostics(candidate_dir / f'held{held}', fitted)
                row = {'held_repetition': held, 'accuracy': metric['accuracy'],
                       'macro_f1': metric['macro_f1'], 'nll': nll(vy, p),
                       'initial_accuracy': float(np.mean(initial_p.argmax(1) == vy)),
                       'initial_nll': nll(vy, initial_p),
                       'fit_seconds': time.monotonic() - fit_start,
                       'fit_seed': fit_seed, 'train_windows': len(y), 'validation_windows': len(vy)}
                fold_rows.append(row)
                if candidate_id == 0 and held == features.TRAIN_REPS[0]:
                    estimate = row['fit_seconds'] * (CANDIDATES * 4 + 1) * len(SEEDS) * 10 / 3600
                    print(f'{device} timing: first full fit {row["fit_seconds"]:.1f}s; '
                          f'rough 10-subject worker estimate {estimate:.2f}h (structure sizes vary)', flush=True)
                    write(candidate_dir / 'timing_forecast.json', {'first_fit_seconds': row['fit_seconds'],
                          'rough_worker_hours': estimate, 'estimate_only': True,
                          'budget_not_changed_after_timing': True})
                    if estimate > 9:
                        raise RuntimeError('Projected worker runtime exceeds9h; stopping before the full grid. '
                                           'Inspect timing_forecast.json and shard the same protocol; do not shrink the budget silently.')
                print(f'S{subject:02} seed{seed} candidate{candidate_id} held{held}: '
                      f'val={row["accuracy"]:.5f}; seconds={row["fit_seconds"]:.1f}', flush=True)
                del fitted
            record = {'candidate_id': candidate_id, 'counts': counts.tolist(), 'proposal': proposal,
                      'total_rules': int(np.prod(counts, axis=1).sum()),
                      'validation_accuracy': float(np.mean([r['accuracy'] for r in fold_rows])),
                      'validation_nll': float(np.mean([r['nll'] for r in fold_rows])), 'folds': fold_rows}
            records.append(record)
            write(candidate_dir / 'validation.json', record)
            write(seed_dir / 'structure_search.json', {'candidates': records,
                  'pareto_archive_ids': [r['candidate_id'] for r in pareto_archive(records)],
                  'test_used_for_selection': False})
        chosen = min(records, key=rank_key)
        write(seed_dir / 'selection.json', {'selected_candidate_id': chosen['candidate_id'],
              'counts': chosen['counts'], 'total_rules': chosen['total_rules'],
              'validation_accuracy': chosen['validation_accuracy'], 'validation_nll': chosen['validation_nll'],
              'ranking': 'mean held-repetition accuracy, then NLL, then total rules, then candidate ID',
              'test_used_for_selection': False})
        final = fit_model(xtrain, ytrain, np.asarray(chosen['counts']), seed, device)
        save_fit_diagnostics(seed_dir, final)
        engine.export_rules(seed_dir, final)
        train_p = engine.predict(xtrain, final, device=device)
        initial_train_p = engine.predict(xtrain, initial_model(final), device=device)
        final_models.append((seed, final, chosen, float(np.mean(train_p.argmax(1) == ytrain)),
                             float(np.mean(initial_train_p.argmax(1) == ytrain))))

    xtest, test_meta = features.extract(runs, subject, 'test', scale * old['threshold_ratio'])
    ytest = test_meta.gesture.to_numpy(int) - 1
    assert len(xtest) == old['test_windows_10ms']
    _, lda_p = features.predict(features.fit_lda(xtrain, ytrain + 1), xtest)
    lda_metric, _ = metrics(ytest, lda_p)
    assert abs(lda_metric['accuracy'] - old['lda_test_accuracy']) < 1e-12, 'LDA input parity failed'
    for seed, final, chosen, train_accuracy, initial_train_accuracy in final_models:
        seed_dir = folder / f'seed{seed}'
        p = engine.predict(xtest, final, device=device)
        initial_test_p = engine.predict(xtest, initial_model(final), device=device)
        result, cm = metrics(ytest, p)
        changes = final['parameter_changes']
        assert all(k in changes for k in PARAMETER_GROUPS), changes
        predictions = test_meta.copy()
        predictions['seed'] = seed
        predictions['BRB_pred'] = p.argmax(1) + 1
        predictions['LDA_pred'] = lda_p.argmax(1) + 1
        predictions['BRB_correct'] = p.argmax(1) == ytest
        predictions['LDA_correct'] = lda_p.argmax(1) == ytest
        predictions.to_csv(seed_dir / 'predictions.csv.gz', index=False)
        np.savez_compressed(seed_dir / 'probabilities.npz', brb=p, initial_brb=initial_test_p,
                            lda=lda_p.astype(np.float32))
        pd.DataFrame(cm, index=np.arange(1, 18), columns=np.arange(1, 18)).to_csv(
            seed_dir / 'confusion.csv', index_label='true_gesture')
        pd.DataFrame({'gesture': np.arange(1, 18), 'windows': cm.sum(1), 'correct': np.diag(cm),
                      'wrong': cm.sum(1) - np.diag(cm), 'recall': np.diag(cm) / cm.sum(1)}).to_csv(
            seed_dir / 'gesture_metrics.csv', index=False)
        # A reference position has no free parameter when its count is two.
        searchable = {k: True for k in PARAMETER_GROUPS}
        searchable['reference_positions'] = bool(np.any(np.asarray(chosen['counts']) > 2))
        updated = all(float(changes[k]) > 0 for k in PARAMETER_GROUPS if searchable[k])
        result.update(subject=subject, seed=seed, selected_candidate=chosen['candidate_id'],
                      total_rules=chosen['total_rules'], validation_accuracy=chosen['validation_accuracy'],
                      validation_nll=chosen['validation_nll'], train_accuracy=train_accuracy,
                      initial_train_accuracy=initial_train_accuracy,
                      initial_test_accuracy=float(np.mean(initial_test_p.argmax(1) == ytest)),
                      initial_test_nll=nll(ytest, initial_test_p),
                      test_nll=nll(ytest, p), lda_accuracy=lda_metric['accuracy'],
                      device=device, gpu_name=torch.cuda.get_device_name(device),
                      parameter_changes=changes, searchable_parameter_groups=searchable,
                      all_searchable_parameters_updated=bool(updated), all_parameter_groups_searched=True,
                      validation_brb_fits=CANDIDATES * 4, structure_candidates=CANDIDATES,
                      success=True)
        write(seed_dir / 'completion.json', result)
        print(f'S{subject:02} seed{seed} {device}: JOPS={result["accuracy"]:.5f}; '
              f'LDA={lda_metric["accuracy"]:.5f}; rules={chosen["total_rules"]}', flush=True)
    write(folder / 'runtime.json', {'seconds': time.monotonic() - started, 'device': device})


def paired_subject_stats(delta):
    """Seeds averaged within subjects; windows are not independent replicates."""
    delta = np.asarray(delta, dtype=float) * 100
    rng = np.random.default_rng(42)
    boot = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(1)
    signs = rng.choice([-1, 1], (100000, len(delta)))
    null = np.abs((signs * delta).mean(1))
    p = float((1 + (null >= abs(delta.mean()) - 1e-12).sum()) / (len(null) + 1))
    return {'mean_gain_pp': float(delta.mean()), 'subject_bootstrap95_pp': np.quantile(boot, [.025, .975]).tolist(),
            'subject_signflip_p_unadjusted': p, 'subjects_improved': int((delta > 0).sum()),
            'subjects_harmed': int((delta < 0).sum())}


def summarize(out, baseline_path):
    rows, gestures, devices, changes_rows = [], [], set(), []
    for subject in range(1, 21):
        for seed in SEEDS:
            folder = out / f'S{subject:02}' / f'seed{seed}'
            r = json.loads((folder / 'completion.json').read_text())
            assert r['success'] and r['all_parameter_groups_searched']
            devices.add(r['device'])
            rows.append({k: v for k, v in r.items() if k not in ('parameter_changes', 'searchable_parameter_groups')})
            changes_rows.append({'subject': subject, 'seed': seed, **r['parameter_changes']})
            gestures.append(pd.read_csv(folder / 'gesture_metrics.csv').assign(subject=subject, seed=seed))
    df = pd.DataFrame(rows)
    baseline = pd.read_csv(baseline_path)
    assert len(baseline) == 60 and not baseline.duplicated(['subject', 'seed']).any()
    df = df.merge(baseline[['subject', 'seed', 'accuracy', 'windows']].rename(
        columns={'accuracy': 'adam_accuracy', 'windows': 'adam_windows'}), on=['subject', 'seed'],
        validate='one_to_one')
    assert len(df) == 60 and (df.windows == df.adam_windows).all()
    df.to_csv(out / 'subject_seed_metrics.csv', index=False)
    pd.DataFrame(changes_rows).to_csv(out / 'parameter_changes.csv', index=False)
    pd.concat(gestures).to_csv(out / 'subject_gesture_metrics.csv', index=False)
    means = df.groupby('subject')[['accuracy', 'lda_accuracy', 'adam_accuracy']].mean()
    means.to_csv(out / 'subject_mean_metrics.csv')
    comparisons = {name: paired_subject_stats(means.accuracy - means[column])
                   for name, column in [('JOPS_vs_Adam', 'adam_accuracy'), ('JOPS_vs_LDA', 'lda_accuracy')]}
    # Holm adjustment across these two prespecified comparator tests.
    ordered = sorted(comparisons, key=lambda k: comparisons[k]['subject_signflip_p_unadjusted'])
    running = 0.
    for index, key in enumerate(ordered):
        running = max(running, min(1., (2 - index) * comparisons[key]['subject_signflip_p_unadjusted']))
        comparisons[key]['subject_signflip_p_holm'] = running
    summary = {
        'JOPS_mean_subject_accuracy_pct': float(df.accuracy.mean() * 100),
        'JOPS_pooled_accuracy_pct': float(np.average(df.accuracy, weights=df.windows) * 100),
        'JOPS_mean_macro_f1_pct': float(df.macro_f1.mean() * 100),
        'initial_empirical_BRB_mean_subject_accuracy_pct': float(df.initial_test_accuracy.mean() * 100),
        'DE_gain_over_own_initialization_pp': float((df.accuracy - df.initial_test_accuracy).mean() * 100),
        'Adam_mean_subject_accuracy_pct': float(df.adam_accuracy.mean() * 100),
        'LDA_mean_subject_accuracy_pct': float(df.lda_accuracy.mean() * 100),
        'comparisons': comparisons, 'selected_rules_mean': float(df.total_rules.mean()),
        'non_independent_confirmation': True,
        'interpretation': 'Changes combine empirical initialization, adaptive structure, references and block DE; not an isolated optimizer ablation.'}
    write(out / 'summary.json', summary)
    completion = {'success': True, 'experiment_id': 'DB7-032', 'subjects': list(range(1, 21)),
        'seeds': list(SEEDS), 'final_brb_fits': len(df),
        'validation_brb_fits': int(df.validation_brb_fits.sum()),
        'structure_candidates': int(df.structure_candidates.sum()),
        'test_window_seed_evaluations': int(df.windows.sum()), 'test_used_for_selection': False,
        'all_parameter_groups_searched': bool(df.all_parameter_groups_searched.all()),
        'all_parameter_groups_updated': bool(df.all_searchable_parameters_updated.all()),
        'devices_used': sorted(devices), 'features': 48, 'channel_modules': 12, 'reference_count_bounds': [2, 4]}
    assert completion['test_window_seed_evaluations'] == 695163 and devices == {'cuda:0', 'cuda:1'}
    assert completion['validation_brb_fits'] == 1200 and completion['structure_candidates'] == 300
    write(out / 'completion.json', completion)
    report = ['# DB7-032 results', '',
        'Budgeted hierarchical JOPS-inspired classification adaptation on the current DB7 EMG data.',
        'Same filtered 48 Hudgins features, training1/3/4/6 and test2/5; 200ms windows, training50ms and test10ms strides.',
        'S1–S20; seeds42/43/44; 1200 validation DE fits plus60 final fits on two GPUs.',
        'Reference counts2–4, ordered internal positions, rule weights, relative attribute weights, consequent beliefs and global ER fusion weights are searched.',
        'Zero movement is reported as optimizer stagnation; two-reference attributes have no free internal position.',
        'Raw input, train-derived thresholds and test-window counts must reproduce the parent experiment.', '',
        '```json', json.dumps(summary, indent=2), '```', '',
        'Uncertainty resamples subjects after averaging seeds. Window overlap prevents treating windows as independent significance-test observations.',
        'The historical Adam comparison changes initialization, reference search, structure and optimization together; it cannot isolate one cause of improvement.',
        'Previously inspected DB7 data: exploratory evidence, not independent confirmation. Budget exhaustion is not convergence.',
        'See each subject/seed for initial/trained rules, weights, reference grids, DE history, validation selection, predictions and gesture errors.']
    (out / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')


def self_test():
    rng = np.random.default_rng(42)
    records = []
    for i, count in enumerate((3, 2)):
        counts = np.full((12, 4), count)
        records.append({'candidate_id': i, 'counts': counts.tolist(), 'total_rules': int(np.prod(counts, axis=1).sum()),
                        'validation_accuracy': .6 + i * .1, 'validation_nll': 1. - i * .1})
    assert [r['candidate_id'] for r in pareto_archive(records)] == [1]
    for i in range(2, 5):
        counts, proposal = propose_structure(records, rng)
        assert counts.shape == (12, 4) and np.all((counts >= 2) & (counts <= 4))
        assert not any(np.array_equal(counts, r['counts']) for r in records)
        records.append({'candidate_id': i, 'counts': counts.tolist(), 'total_rules': int(np.prod(counts, axis=1).sum()),
                        'validation_accuracy': .7, 'validation_nll': .9, 'proposal': proposal})
    assert min(records, key=rank_key)['candidate_id'] == 1
    assert paired_subject_stats(np.zeros(20))['subject_signflip_p_unadjusted'] == 1.
    print('ORCHESTRATOR SELF TEST PASSED: distinct bounded proposals, Pareto dominance, ranking and paired statistics', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/db7_032_results'))
    parser.add_argument('--parent', type=Path, default=Path('/kaggle/working/db7_031_parent.json'))
    parser.add_argument('--baseline', type=Path, default=Path('/kaggle/working/db7_032_adam_baseline.csv'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--subjects', type=int, nargs='+')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test:
        self_test()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    parent = json.loads(args.parent.read_text())
    raw = args.input or features.find_root()
    if args.worker:
        torch.cuda.set_device(args.device)
        write(args.out / f'worker_{args.device[-1]}_runtime.json', {'device': args.device,
              'gpu_name': torch.cuda.get_device_name(args.device), 'pid': os.getpid(),
              'subjects': args.subjects, 'torch_version': torch.__version__})
        for subject in args.subjects:
            run_subject(raw, subject, args.out, parent, args.device)
        return
    assert torch.cuda.device_count() >= 2, 'Two GPUs required; verify T4 x2 before fitting'
    self_test()
    write(args.out / 'PROTOCOL.json', {'experiment_id': 'DB7-032', 'population': POPULATION,
        'generations_per_block': GENERATIONS, 'cycles': CYCLES, 'candidates_per_subject_seed': CANDIDATES,
        'sample_batch': 2048, 'population_chunk': 4,
        'runtime_guard': 'stop if first-fit extrapolated worker compute exceeds9h; shard rather than silently changing search budget',
        'optimizer': 'cooperative DE/rand/1/bin; 12 channel blocks plus fusion block',
        'objective': 'full fitting-data classification cross-entropy',
        'initialization': 'training-only empirical rule/class evidence plus jittered population',
        'structure_selection': 'mean fourfold held-training-repetition accuracy; NLL, rule count tie breakers',
        'structure_proposal': 'SOHS-inspired bounded add/prune; componentwise reference-count Pareto archive; phi0.25',
        'reference_counts': [2, 4], 'train_repetitions': features.TRAIN_REPS, 'test_repetitions': features.TEST_REPS,
        'window_ms': 200, 'training_stride_ms': 50, 'test_stride_ms': 10, 'test_used_for_selection': False,
        'learned': PARAMETER_GROUPS, 'seeds': SEEDS, 'subjects': list(range(1, 21)), 'neural_fits': 0,
        'paper_adaptation': True, 'budget_limited_not_convergence_claim': True})
    # Copy provenance into the downloadable result package.
    (args.out / 'adam_baseline.csv').write_bytes(args.baseline.read_bytes())
    provenance = args.baseline.with_suffix('.json')
    if provenance.exists():
        (args.out / 'adam_baseline_provenance.json').write_bytes(provenance.read_bytes())
    write(args.out / 'source_hashes.json', {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
          for name in ('db7_030_hudgins.py', 'db7_031_brb.py', 'db7_032_jops_engine.py', 'db7_032_jops.py')})
    workers = []
    for device in range(2):
        subjects = list(range(device + 1, 21, 2))
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--device', f'cuda:{device}',
               '--input', str(raw), '--out', str(args.out), '--parent', str(args.parent), '--subjects', *map(str, subjects)]
        workers.append(subprocess.Popen(cmd))
    while any(p.poll() is None for p in workers):
        if any(p.poll() not in (None, 0) for p in workers):
            for p in workers:
                if p.poll() is None:
                    p.terminate()
            break
        time.sleep(5)
    codes = [p.wait() for p in workers]
    if codes == [0, 0]:
        summarize(args.out, args.baseline)
    else:
        write(args.out / 'failure.json', {'worker_exit_codes': codes, 'success': False})
    archive = args.out.parent / 'db7_032_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(args.out.rglob('*')):
            if file.is_file():
                z.write(file, file.relative_to(args.out))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    assert codes == [0, 0], f'Worker failures {codes}; partial results archived'
    print('DB7-032 COMPLETE', archive, flush=True)


if __name__ == '__main__':
    main()
