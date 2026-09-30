"""Build the self-contained, documented DB7-032 Kaggle notebook and manifest."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "scripts/db7_030_hudgins.py": "db7_030_hudgins.py",
    "scripts/db7_031_brb.py": "db7_031_brb.py",
    "scripts/db7_032_jops_engine.py": "db7_032_jops_engine.py",
    "scripts/db7_032_jops.py": "db7_032_jops.py",
    "data/db7-031-parent.json": "db7_031_parent.json",
    "data/db7-032-adam-baseline.csv": "db7_032_adam_baseline.csv",
    "data/db7-032-adam-baseline.json": "db7_032_adam_baseline.json",
}


def cell(kind: str, text: str) -> dict:
    result = {
        "cell_type": kind,
        "id": hashlib.sha256((kind + text).encode()).hexdigest()[:12],
        "metadata": {}, "source": text.splitlines(True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


INTRO = """# DB7-032 — JOPS-inspired learning for the existing Hudgins BRB

## Research question
Can joint structure and parameter optimization improve our existing hierarchical
BRB on NinaPro DB7 **EMG**? The user confirmed this dataset; this is not an EEG study.

This is a **budgeted hierarchical JOPS-inspired classification adaptation**, not
an exact replication of Yang et al.'s regression experiments. It preserves the
12-channel hierarchical BRB and ER fusion, and jointly searches reference counts,
reference locations, rule weights, attribute weights, consequent beliefs and
fusion weights. It does not add neural branches or train an SI-to-I correction gate.

## Fixed data and inputs
- Subjects S1–S20; E1 restimulus labels 1–17; rest excluded.
- Training repetitions 1, 3, 4 and 6; reserved test repetitions 2 and 5.
- Each EMG window is 200 ms (400 samples at 2 kHz).
- Training stride is 50 ms; test stride is 10 ms, matching DB7-031.
- Per labeled repetition, apply a fourth-order 20–450 Hz Butterworth filter and
  a 50 Hz, Q=30 notch, both zero-phase. Windows do not cross repetitions.
- Each of the 12 EMG channels contributes MAV, WL, ZC and SSC: all 48 inputs remain.
- The unchanged DB7-030 preprocessing and DB7-031 data helpers are embedded below.
  Raw recording hashes and final training-derived threshold choices are checked
  against the saved parent evidence. Test data do not fit preprocessing.

## Model structure
Each of 12 channel modules receives its four Hudgins attributes. The AND-style
matching and complete-belief ER within channels and between channels are retained.
A channel uses the Cartesian product of its feature references as its rule base.
Each rule predicts 17 gesture beliefs; the final output is one 17-class distribution.

The allowed reference count for each of the 48 attributes is 2, 3 or 4. Thus the
number of rules in a channel is the product of its four reference counts. A
three-reference channel has 81 rules; two references each give 16 rules. These
are compact channel modules, not a single infeasible 48-attribute Cartesian grid.

## Five candidate structures per subject and seed
The fixed search budget is:
1. The existing all-three-reference structure.
2. An all-two-reference structure.
3. Three further SOHS-inspired add/prune proposals, keeping every attribute within
   the range 2–4 references.

A Pareto archive removes a candidate when another has no worse validation
accuracy and no more references in **every one of the 48 attributes**, with at
least one strict improvement. This componentwise reference-count comparison is
different from dominance based only on total rule count. Candidate comparisons
use only the four training repetitions. Structure
selection maximizes mean held-repetition accuracy; ties use lower classification
negative log likelihood, then fewer rules. There is no test-based rule-count choice.

This small structure search is a budget limit, not proof that its chosen structure
is globally optimal. It tests whether a practical JOPS adaptation improves this
specific BRB; it does not prove the full search space has been exhausted.

## Parameters learned by Differential Evolution
For every fitting split:
- The low and high feature endpoints are the fitting-data 5th and 95th percentiles.
  Internal reference positions are optimized while maintaining strict order.
  A two-reference attribute has no internal reference to optimize.
- Rule consequent beliefs are initialized from fitting-data rule/class evidence.
- Rule weights, attribute weights, consequent beliefs and channel-fusion weights
  are trained, with constraints enforced by the parameterization.
- The fitting objective is classification cross-entropy. Gesture numbers are
  categorical labels; numerical distance between gesture IDs is not a loss.

Optimization uses cooperative/block Differential Evolution: **population 12,
four generations per block, two cycles over the 12 channel blocks plus the fusion
block**. Fitness uses the full fitting dataset. These budgets are prespecified,
not selected by checking test accuracy. Learning histories record accepted
updates and stagnation, including a block where no proposal improved the loss.
The mutation factor is 0.5 and crossover probability is 0.9. GPU fitness is
evaluated in chunks of 2,048 windows and four population members; chunking does
not subsample or change the full-dataset objective.

This differs from the paper's full-vector DE and regression complexity criterion.
The budgeted block search makes the hierarchical classification experiment
practical and keeps its actual changes inspectable.

**Initialization also changes from DB7-031:** that model initialized every rule
with the fitting-class prior; this experiment initializes each rule using its
own fitting-data activation/class evidence. Therefore the historical comparison
changes initialization, reference positions/counts and optimization together.
An improvement cannot be attributed to DE or structure search alone. The exported
initial/final rule tables and optimization histories show how much the subsequent
search changes its empirical initializer.

## Repetition validation and final fitting
For each candidate, hold out one of training repetitions 1/3/4/6 and fit on the
other three. Repeat four times. Feature scales, references, rule initialization
and nested ZC/SSC threshold screening exclude the held-out repetition.

After selecting a structure using mean validation results, fit it from scratch
on all four training repetitions. Only then predict repetitions 2/5. The test
labels are used to calculate metrics, never to select a structure or parameters.

## Compute and reproducibility
There are 20 subjects × three seeds (42, 43, 44) × five candidates × four folds =
**1,200 validation BRB fits**, followed by **60 final fits**. The 300 candidate
evaluations include the prespecified initial structures and the add/prune proposals.

Two worker processes divide subjects between cuda:0 and cuda:1. The notebook
asserts that **both Kaggle T4 GPUs** are available, and records each worker's GPU
name and assigned subjects. An independent CPU self-test checks memberships, ER
and constrained optimization before the full experiment.

The first full fitting run on each worker provides a rough runtime forecast.
If its extrapolated worker compute exceeds **nine hours**, the run stops and
preserves partial diagnostics; it does not silently reduce the search budget.
That outcome calls for splitting the same protocol into smaller runs. The
GitHub workflow monitors for five hours within its six-hour job; a longer Kaggle
run can be monitored again with `monitor_ref` without submitting duplicate work.
The forecast is approximate because later candidate rule counts can differ.

## What the results can establish
The matched comparator is the same filtered 48-feature LDA. DB7-031 supplies the
previous hierarchical BRB result for comparison. SI also uses inertial inputs,
so SI accuracy is contextual rather than a modality-matched baseline.

Report mean subject accuracy, pooled window accuracy, macro-F1, per-gesture errors,
confusions and paired subject-level uncertainty. Overlapping windows and the
three seeds are not independent experimental subjects. These previously inspected
recordings provide exploratory evidence; an apparent improvement is not an
independent confirmation of generalization.

Exports include candidate structures, fitting/validation histories, selection
evidence, initial and trained rules/references/weights, final predictions,
probabilities and completion checks. No accuracy is promised before execution.
"""


def main() -> None:
    cells = [cell("markdown", INTRO)]
    descriptions = {
        "scripts/db7_030_hudgins.py": (
            "1. Preserve feature extraction", "Unchanged filtering, windowing and Hudgins feature definitions."),
        "scripts/db7_031_brb.py": (
            "2. Preserve parent data and validation helpers", "The previous BRB code is embedded for shared helpers; the new orchestrator runs the JOPS engine."),
        "scripts/db7_032_jops_engine.py": (
            "3. Define JOPS structure and parameter learning", "Variable reference memberships, constrained parameters, complete-belief ER, block Differential Evolution and structure proposals are implemented below."),
        "scripts/db7_032_jops.py": (
            "4. Define the experiment and diagnostics", "Read the explicit configuration, fit/validation separation, dual-GPU scheduling, rule exports and completion assertions here."),
        "data/db7-031-parent.json": (
            "5. Pin parent provenance", "These saved raw-data hashes and training-only threshold choices reproduce the parent input protocol. Parent test metrics are reproduction checks, not a selection objective."),
        "data/db7-032-adam-baseline.csv": (
            "6. Retain the historical DB7-031 result", "These 60 subject/seed metrics are used only for the final report. They are never used to select a structure or parameter setting."),
        "data/db7-032-adam-baseline.json": (
            "7. Record historical-result provenance", "The provenance record identifies the saved DB7-031 metrics and their integrity hash."),
    }
    hashes = {}
    for relative, destination in SOURCE_FILES.items():
        source_path = ROOT / relative
        source = source_path.read_text(encoding="utf-8")
        hashes[relative] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        title, explanation = descriptions[relative]
        cells.append(cell("markdown", f"## {title}\n\n{explanation}\n"))
        cells.append(cell("code", f"%%writefile /kaggle/working/{destination}\n" + source))

    cells.extend([
        cell("markdown", """## 8. Run checks, then fit using both GPUs

The engine and orchestration self-tests run before any full model training.
A missing GPU, data mismatch,
failed fit or incomplete output aborts execution. Parameter-change evidence is
recorded across the run; every parameter group must be searched, but DE is allowed
to record stagnation rather than fabricating an update. Initial and final exports
make this inspectable. A two-reference feature has no internal reference position.
"""),
        cell("code", """import os
import sys
import json
import subprocess
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
subprocess.run([sys.executable, '/kaggle/working/db7_032_jops_engine.py', '--self-test'], check=True)
subprocess.run([sys.executable, '/kaggle/working/db7_032_jops.py', '--self-test'], check=True)
subprocess.run([sys.executable, '/kaggle/working/db7_032_jops.py'], check=True)
out = Path('/kaggle/working/db7_032_results')
done = json.loads((out / 'completion.json').read_text())
assert done['success']
assert done['subjects'] == list(range(1, 21)) and done['seeds'] == [42, 43, 44]
assert done['final_brb_fits'] == 60 and done['validation_brb_fits'] == 1200
assert done['structure_candidates'] == 300
assert done['devices_used'] == ['cuda:0', 'cuda:1']
assert done['test_window_seed_evaluations'] == 695163
assert done['all_parameter_groups_searched'] and not done['test_used_for_selection']
assert isinstance(done['all_parameter_groups_updated'], bool)
print(json.dumps(done, indent=2))
"""),
        cell("markdown", """## 9. Inspect measured results

The table reports each final subject/seed model. The plot averages seeds within
each subject. Gesture recall counts actual predictions; the test grid is dense
and overlapping. Read the accompanying report for paired uncertainty and limits.
"""),
        cell("code", """import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

display(Markdown((out / 'REPORT.md').read_text()))
metrics = pd.read_csv(out / 'subject_seed_metrics.csv')
display(metrics)
by_subject = metrics.groupby('subject')[['accuracy', 'lda_accuracy']].mean() * 100
by_subject.columns = ['JOPS BRB', 'Matched Hudgins LDA']
ax = by_subject.plot.bar(figsize=(14, 5), ylim=(0, 100), ylabel='Test accuracy (%)',
                         title='DB7-032: JOPS BRB and matched LDA')
plt.tight_layout()
plt.savefig(out / 'subject_accuracy.png', dpi=160)
plt.show()

gestures = pd.read_csv(out / 'subject_gesture_metrics.csv')
grid = gestures.groupby(['subject', 'gesture']).recall.mean().unstack() * 100
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(grid, vmin=0, vmax=100, aspect='auto', cmap='viridis')
ax.set(xticks=range(17), xticklabels=range(1, 18), yticks=range(20),
       yticklabels=range(1, 21), xlabel='Gesture', ylabel='Subject',
       title='JOPS BRB mean test recall by subject and gesture')
fig.colorbar(im, ax=ax, label='Recall (%)')
fig.tight_layout()
fig.savefig(out / 'gesture_recall.png', dpi=160)
plt.show()

# Include presentation plots in the result ZIP downloaded by GitHub Actions.
import zipfile
with zipfile.ZipFile('/kaggle/working/db7_032_results.zip', 'w',
                     zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for file in sorted(out.rglob('*')):
        if file.is_file():
            archive.write(file, file.relative_to(out))
print('Download db7_032_results.zip for all rules, weights, histories and predictions.')
"""),
    ])
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    destination = ROOT / "db7-032-jops-hierarchical-brb.ipynb"
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(notebook, indent=1, ensure_ascii=False))
    manifest = {
        "experiment_id": "DB7-032",
        "method": "budgeted hierarchical JOPS-inspired classification adaptation",
        "notebook": destination.name,
        "notebook_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_files_sha256": hashes,
        "raw_dataset": "rayaanraza1/ninapro-db7",
        "subjects": list(range(1, 21)), "seeds": [42, 43, 44],
        "features": 48, "modules": 12,
        "reference_count_range": [2, 4],
        "structure_candidates_per_subject_seed": 5,
        "structure_candidates": 300,
        "de_population": 12, "de_generations_per_block": 4, "de_cycles": 2,
        "de_mutation": 0.5, "de_crossover": 0.9,
        "fitness_sample_batch": 2048, "fitness_population_chunk": 4,
        "runtime_guard_forecast_hours": 9,
        "trained_parameters": ["internal reference positions", "rule weights", "attribute weights", "consequent beliefs", "fusion weights"],
        "validation_fits": 1200, "final_fits": 60,
        "expected_test_window_seed_evaluations": 695163, "gpus_required": 2,
        "test_used_for_selection": False,
    }
    (ROOT / "db7-032-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination, manifest["notebook_sha256"])


if __name__ == "__main__":
    main()
