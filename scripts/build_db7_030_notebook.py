"""Package the reviewed DB7-030 source as a self-contained Kaggle notebook."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
source = (root / "scripts/db7_030_hudgins.py").read_text(encoding="utf-8")
trace = root / "data/db7-030-si-i-trace.csv.gz"


def cell(kind, value):
    item = {"cell_type": kind, "metadata": {}, "source": value.splitlines(True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


cells = [
    cell("markdown", """# DB7-030 — Hudgins TD4 as a complement to SI and I

**Question.** Do four time-domain EMG features recover DB7-016 errors, especially windows where both the spectral + inertial (SI) and inertial-only (I) networks fail? This study trains only lightweight regularized LDA classifiers. The neural outputs are frozen.

**Inputs.** NinaPro DB7 E1, gestures 1–17, 12 EMG channels sampled at 2 kHz; saved DB7-016 SI/I predictions for S1–S20 and seeds 42/43/44. Rest is excluded. Training repetitions 1/3/4/6, test repetitions 2/5. EMG is filtered separately within each labeled repetition with the DB7-016 20–450 Hz fourth-order Butterworth bandpass and 50 Hz Q30 notch.

**Features.** For every 200 ms window: mean absolute value (MAV), waveform length (WL), thresholded zero crossings (ZC), and thresholded slope sign changes (SSC), each on 12 channels. The three candidate ZC/SSC thresholds are 0.5%, 1%, and 2% of each channel's 95th percentile absolute EMG on **training repetitions only**. The training window grid is 50 ms to limit duplicated evidence; all test predictions use the DB7-016 10 ms grid.

**Selection.** Four leave-one-training-repetition-out folds select a threshold using full TD4. Under that threshold, the same four folds score all 15 nonempty feature-family subsets. Select the best validation subset per subject, breaking ties toward fewer families. Fit each subset on all training repetitions. Test labels are never used for selecting the threshold or subset.

**Outputs.** Per-subject and per-gesture metrics; all 15 subset scores; aligned TD4/selected predictions; SI/I shared-error recovery, avoidable harms, and diagnostic top-one oracle. Oracle scores use the true label and are **not** achieved fusion accuracy. This is an exploratory evaluation on previously inspected DB7-016 test repetitions, not independent confirmation.

**Predictions are already measured.** The GitHub Action creates a private Kaggle dataset from a compact, verified copy of the DB7-016 SI/I trace. No neural network is trained here.
"""),
    cell("code", "import os\nos.environ['OPENBLAS_NUM_THREADS']='1'\nos.environ['OMP_NUM_THREADS']='1'\nfrom pathlib import Path\nimport json,subprocess,sys,shutil,zipfile\nimport pandas as pd\nfrom IPython.display import display,Markdown\n"),
    cell("markdown", "## Executable analysis source\nThe cell below writes the versioned implementation. It has no external code dependency beyond Kaggle's standard NumPy, pandas and SciPy."),
    cell("code", "%%writefile /kaggle/working/db7_030_hudgins.py\n" + source),
    cell("markdown", "## Run S1–S20 and verify exact saved-window alignment"),
    cell("code", """trace_files=sorted(Path('/kaggle/input').rglob('db7-030-si-i-trace.csv.gz'))
assert len(trace_files)==1, f'Expected one DB7-016 trace, found {trace_files}'
out=Path('/kaggle/working/db7_030_results')
subprocess.run([sys.executable,'/kaggle/working/db7_030_hudgins.py',
                '--trace',str(trace_files[0]),'--out',str(out)],check=True)
done=json.loads((out/'completion.json').read_text())
assert done['success'] and done['subjects']==list(range(1,21))
assert done['neural_fits']==0 and done['test_window_seed_evaluations']==695163
display(pd.read_csv(out/'complementarity_summary.csv'))
display(pd.read_csv(out/'candidate_preference_diagnostic.csv'))
display(pd.read_csv(out/'feature_combination_summary.csv'))
"""),
    cell("markdown", "## Feature subsets and shared-error recovery"),
    cell("code", """import matplotlib.pyplot as plt
sub=pd.read_csv(out/'feature_combination_summary.csv')
fig,ax=plt.subplots(figsize=(11,6))
sub=sub.sort_values('mean_subject_cv_accuracy')
ax.barh(sub.families,100*sub.mean_subject_cv_accuracy,color='#3071a9')
ax.set(xlabel='Mean training-fold validation accuracy (%)',title='Hudgins feature-family combinations')
fig.tight_layout();fig.savefig(out/'feature_combinations.png',dpi=150);plt.show()
comp=pd.read_csv(out/'complementarity_summary.csv')
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(comp.method,comp.shared_errors_recovered,color=['#167f82','#9a5a9a'])
ax.set(ylabel='SI/I shared-error window–seed evaluations recovered',title='Hudgins standalone recovery opportunity')
fig.tight_layout();fig.savefig(out/'shared_error_recovery.png',dpi=150);plt.show()
"""),
    cell("markdown", "## Save the complete diagnostic bundle"),
    cell("code", """report=['# DB7-030 Hudgins TD4 results','',
        'This is exploratory: SI/I test repetitions were inspected in earlier experiments.','',
        '## Complementarity','',
        '```csv',comp.to_csv(index=False).strip(),'```','',
        '## Candidate preference (fixed zero threshold)','',
        '```csv',pd.read_csv(out/'candidate_preference_diagnostic.csv').to_csv(index=False).strip(),'```','',
        '## Feature combinations (validation-ranked)','',
        '```csv',pd.read_csv(out/'feature_combination_summary.csv').to_csv(index=False).strip(),'```']
(out/'REPORT.md').write_text('\\n'.join(report),encoding='utf-8')
archive=Path('/kaggle/working/db7_030_results.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(out.rglob('*')):
        if path.is_file():z.write(path,path.relative_to(out))
assert zipfile.ZipFile(archive).testzip() is None
print('Complete result bundle:',archive,archive.stat().st_size,'bytes')
display(Markdown((out/'REPORT.md').read_text()))
"""),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5}
path = root / "db7-030-hudgins-td4.ipynb"
with path.open("w", encoding="utf-8", newline="\n") as file:
    file.write(json.dumps(notebook, indent=1))
manifest = {"experiment_id": "DB7-030", "subjects": list(range(1, 21)),
    "raw_dataset": "rayaanraza1/ninapro-db7",
    "trace_rows": 695163,
    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "neural_fits": 0, "test_repetitions": [2, 5],
    "train_repetitions": [1, 3, 4, 6],
    "limitations": "Already-inspected DB7-016 test predictions; exploratory complementarity, no independent confirmation."}
(root / "db7-030-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(path, manifest["notebook_sha256"])
