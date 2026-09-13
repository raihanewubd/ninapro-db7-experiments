"""Build the self-contained energy ablation notebook from readable source modules."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def inline_module(path):
    source = path.read_text(encoding='utf-8')
    parts = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module in {
            '__future__', 'baseline_support', 'energy_model', 'reference_windows',
        }:
            continue
        if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
            continue
        parts.append(ast.get_source_segment(source, node))
    return '\n\n'.join(parts) + '\n'


def main():
    cells = []

    def add(kind, source):
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, id=f'energy-study-{len(cells):03}', metadata={},
                    source=source.splitlines(True))
        if kind == 'code':
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)

    def definitions(source):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            add('code', ast.get_source_segment(source, node) + '\n')

    add('markdown', '''# DB7 Exercise B: can explicit EMG energy reduce early-window errors?

This is a controlled follow-up to the completed C1 experiment. Train three independent models per subject:

| Arm | Model | Parameters |
|---|---|---:|
| C1 | Waveform + spectrogram + ACC/gyro/magnetometer | 551,542 |
| C2A | C1 plus recent amplitude, duplicated into a 24-input residual path | 555,510 |
| C2D | C1 plus recent amplitude and relative energy change, same residual path size | 555,510 |

**Protocol:** all 22 subjects; E1 `restimulus` labels 1–17; rest excluded; fixed within-subject 4 train / 1 validation / 1 test repetitions per gesture. Windows are 400 ms, stride 100 ms, boundary trim 100 ms. Seven internal 100 ms frames have a 50 ms hop. The previous window identities must match exactly.

**Questions:** Does explicit amplitude help? Does relative energy change add value beyond amplitude? How many early errors are recovered, and how many correct predictions become wrong in each phase?

GitHub runs seeds 42, 43 and 44 sequentially: 198 full fits, following six two-epoch smoke fits on S1/S22. Shared C1 parameters are copied from exactly the same initialization within each subject/seed, and the two added paths have identical initialization and capacity. Each model selects its own minimum-validation-loss checkpoint; no refit or test-based selection.

The test repetitions have already informed development, so this comparison is **exploratory**, with seed robustness but only one test repetition per subject/gesture. A later repetition-rotation study is needed for broader confirmation. Label-relative early thirds and 100 ms endpoint bins are **not physiological onset annotations**. The earliest endpoint is about 500 ms after the refined label begins. Zero-phase preprocessing is offline.

Use Kaggle's T4 GPU and dataset `rayaanraza1/ninapro-db7`. No packages are installed in this notebook.
''')
    add('markdown', '''## 1. Frozen preprocessing and training definitions

Preserve 20–450 Hz fourth-order Butterworth filtering and the 50 Hz notch, applied independently within each assigned repetition with zero phase. Input scaling uses training windows only. Retain all four sensor modalities. Keep ACC centering, augmentation, loss weighting and the LDA gate off to isolate the energy change. Adam, learning-rate schedule, early stopping and batch size remain those of C1.
''')
    definitions((SCRIPTS / 'baseline_support.py').read_text(encoding='utf-8'))
    add('code', "matplotlib.use('Agg')\n")
    add('markdown', '## 2. Freeze the historical train/validation/test window identities\n')
    add('code', (SCRIPTS / 'reference_windows.py').read_text(encoding='utf-8'))
    add('markdown', '''## 3. C1 and the two energy variants

For each EMG channel at each internal frame end, compute RMS over the most recent 25 ms and full 100 ms. The source is filtered EMG divided by the training-channel SD, before mean subtraction. Recover it from the standardized input using the saved training mean/SD.

- C2A: two copies of `log(RMS25 + 1e-6)`; no relative-change information.
- C2D: `log(RMS25 + 1e-6)` and `log(RMS25 + 1e-6) - log(RMS100 + 1e-6)`.

Fit each channel-feature's moments on training windows/frames only, freeze them, and divide both arms' standardized feature vectors by sqrt(2). The same 24→32→96 projection is added to the waveform sequence. Its final layer starts at zero, so both variants initially reproduce C1. These are explicit representations of existing signal information, not a validated onset detector. No phase or gesture label enters the model. All feature samples already lie inside the 400 ms window.
''')
    definitions(inline_module(SCRIPTS / 'energy_model.py'))
    add('code', 'c1_preflight = model_preflight\n')
    add('markdown', '''## 4. Training and diagnostics

Run numerical/GPU preflight before fitting. Fit and save input, spectral and energy scalers using training data only. Save selected checkpoints, shared-initialization hashes, histories, probabilities, embeddings, confusion/error tables, raw/filtered waveform cases, label disagreements and energy trajectories. Report recovered and newly introduced errors by subject, gesture, phase and endpoint time. Repeated seeds and overlapping windows are not independent trials.
''')
    definitions(inline_module(SCRIPTS / 'energy_study.py'))
    add('markdown', '''## 5. Run configuration

`SEED_BASE` controls initialization and shuffling; the repetition split seed stays 42. `SMOKE=True` uses S1/S22 and two epochs for integration checking. Full-study results come only from `SMOKE=False`.
''')
    add('code', 'Config.SEED_BASE = 42\nConfig.SMOKE = False\nConfig.AUTOMATION = {}\n')
    add('markdown', '## 6. Execute the study and export the result ZIP\nGitHub supplies run-specific provenance in this final cell.\n')
    add('code', 'RESULTS_DIRECTORY = run_energy_study()\nprint(RESULTS_DIRECTORY)\n')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python', version='3.12')))
    destination = ROOT / 'notebooks/db7-energy.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    combined = '\n\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    compile(combined, str(destination), 'exec')
    (SCRIPTS / 'db7_energy_export.py').write_text(combined, encoding='utf-8')
    manifest = dict(notebook=destination.name, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    subjects=list(range(1, 23)), arms=['C1', 'C2A', 'C2D'], seed_bases=[42, 43, 44],
                    full_fits=198, smoke_fits=6, window_ms=400, stride_ms=100, split_seed=42,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [SCRIPTS / n for n in ('baseline_support.py', 'reference_windows.py', 'energy_model.py', 'energy_study.py')]},
                    code_cells=sum(c['cell_type'] == 'code' for c in cells))
    (ROOT / 'package_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
