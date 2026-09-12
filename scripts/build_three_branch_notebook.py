"""Build a readable, self-contained Kaggle notebook without installing packages."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SCRIPTS = ROOT / 'scripts'
SOURCE = WORKSPACE / 'outputs/db7-modality-screen/scripts/db7_emg_acc.py'


def build_support():
    if not SOURCE.exists():
        # A repository clone already contains the frozen support/reference modules.
        support = (SCRIPTS / 'baseline_support.py').read_text(encoding='utf-8')
        reference = (SCRIPTS / 'reference_windows.py').read_text(encoding='utf-8')
        compile(support, 'baseline_support.py', 'exec')
        compile(reference, 'reference_windows.py', 'exec')
        return support, reference
    text = SOURCE.read_text(encoding='utf-8')
    names = {
        '_find_kaggle_input', 'Config', 'RepetitionEMGFilter', 'SubjectLoader',
        'ParallelMultiKernelBlock', 'ChannelAttentionBlock', 'TemporalAttentionPool',
        'OriginalMultiKernelAttention1DCNN', 'Trainer', 'DiagnosticTrainer',
        'json_write', 'save_figure', 'probability_metrics', 'learning_report',
        'predict_with_diagnostics', 'load_modalities', 'ModalityWindows', 'train_scaler',
    }
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or getattr(node, 'name', None) in names:
            nodes.append(ast.get_source_segment(text, node))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHANNEL_COUNTS' for t in node.targets):
            nodes.append(ast.get_source_segment(text, node))
    support = '"""Unmodified model/data/trainer definitions from the modality study."""\n' + '\n\n'.join(nodes) + '\n'
    compile(support, 'baseline_support.py', 'exec')
    (SCRIPTS / 'baseline_support.py').write_text(support, encoding='utf-8')

    path = WORKSPACE / 'outputs/db7-modality-results-34679784088/analysis/fit_summary.csv'
    references = {}
    for row in csv.DictReader(path.open(encoding='utf-8', newline='')):
        if row['arm'] == 'all_sensors':
            references[int(row['subject'])] = ast.literal_eval(row['window_key_hashes'])
    assert set(references) == set(range(1, 23))
    (ROOT / 'reference_window_hashes.json').write_text(json.dumps(references, indent=2), encoding='utf-8')
    reference_source = '# Saved window identities from the completed all-sensor study.\nREFERENCE_WINDOW_HASHES = ' + repr(references) + '\n'
    (SCRIPTS / 'reference_windows.py').write_text(reference_source, encoding='utf-8')
    return support, reference_source


def inline_module(path):
    text = path.read_text(encoding='utf-8')
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.ImportFrom) and node.module in {'baseline_support', 'three_branch_model', 'reference_windows', '__future__'}:
            continue
        if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
            continue
        nodes.append(ast.get_source_segment(text, node))
    return '\n\n'.join(nodes) + '\n'


def main():
    support, reference = build_support()
    cells = []

    def add(kind, source):
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, id=f'three-branch-{len(cells):02}', metadata={}, source=source.splitlines(True))
        if kind == 'code':
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)

    add('markdown', '# DB7 Exercise B: three-branch model versus all-sensor baseline\n\n'
        'One independently trained B0 baseline and C1 three-branch model per subject. '
        '22 subjects; 400 ms windows; 100 ms stride; fixed 4/1/1 repetitions. '
        'GitHub runs seed bases 42, 43 and 44 in separate Kaggle jobs. '
        'No RMS add-on, T-EKIM, extra centering, augmentation or LDA gate is used in this first comparison. '
        'The historical test split is exploratory because it has already informed model development. '
        'No packages are installed in this notebook. Enable the T4 GPU and attach rayaanraza1/ninapro-db7.\n')
    add('markdown', '## 1. Existing preprocessing, baseline and trainer\nThe definitions below preserve the previous protocol. Input means/SDs are fitted only on training windows.\n')
    # Keep natural definitions in separate cells for easier editing and inspection.
    tree = ast.parse(support)
    imports = [ast.get_source_segment(support, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    add('code', '\n'.join(imports) + "\nmatplotlib.use('Agg')\n")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            add('code', ast.get_source_segment(support, node) + '\n')
    add('markdown', '## 2. Freeze the earlier window identities\nEach newly loaded split must match its stored SHA-256 window-key hash before training.\n')
    add('code', reference)
    add('markdown', '## 3. Three-branch architecture\nEMG waveform + log-power spectrogram + ACC/gyro/magnetometer; seven aligned frames; four ordered frequency regions; 551,542 parameters.\n')
    model = inline_module(SCRIPTS / 'three_branch_model.py')
    for node in ast.parse(model).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        add('code', ast.get_source_segment(model, node) + '\n')
    add('code', 'c1_preflight = model_preflight\n')
    add('markdown', '## 4. Training, checkpoint evaluation and failure diagnostics\nAll scalers are frozen before validation/test evaluation. Selected-epoch training metrics use evaluation mode.\n')
    study = inline_module(SCRIPTS / 'three_branch_study.py')
    for node in ast.parse(study).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        add('code', ast.get_source_segment(study, node) + '\n')
    add('markdown', '## 5. Run configuration\nSet SMOKE=True for S1 and S22, two epochs each. SEED_BASE changes model initialization/shuffling only; split seed stays 42.\n')
    add('code', 'Config.SEED_BASE = 42\nConfig.SMOKE = False\nConfig.AUTOMATION = {}\n')
    add('markdown', '## 6. Train and save outputs\nGitHub replaces this final cell with run-specific provenance and seed. Full training begins only after its separate smoke job succeeds.\n')
    add('code', 'RESULTS_DIRECTORY = run_three_branch_study()\nprint(RESULTS_DIRECTORY)\n')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.12')))
    destination = ROOT / 'notebooks/db7-three-branch.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    combined = '\n\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    compile(combined, str(destination), 'exec')
    (SCRIPTS / 'db7_three_branch_export.py').write_text(combined, encoding='utf-8')
    manifest = dict(notebook=destination.name, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    baseline_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest() if SOURCE.exists() else 'f8e959571f4b222a96c1d29f4a37e21fc66ec310a210f026a5cc0f69342a90d4',
                    subjects=list(range(1, 23)), arms=['B0', 'C1'], seed_bases=[42, 43, 44],
                    full_fits=132, smoke_fits=4, code_cells=sum(c['cell_type'] == 'code' for c in cells))
    (ROOT / 'package_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
