"""Submit one notebook configuration to Kaggle; verify and retain its diagnostic outputs."""
import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare(job):
    assert job == 'smoke' or re.fullmatch(r'w(200|400|600)-s(50|100|200)', job)
    user = os.environ['KAGGLE_USERNAME']
    assert re.fullmatch(r'[A-Za-z0-9_-]+', user)
    run_id = os.environ.get('GITHUB_RUN_ID', '0')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    assert run_id.isdigit() and attempt.isdigit()
    reference = f'{user}/db7-diag-{run_id}-{attempt}-{job}'
    folder = ROOT/'diagnostic_submitted'/job
    notebook = json.loads((ROOT/'notebooks/db7-window-stride-diagnostics.ipynb').read_text(encoding='utf-8'))
    provenance = dict(kaggle_ref=reference, github_run_id=run_id, github_run_attempt=attempt,
                      github_commit=os.environ.get('GITHUB_SHA','local'), job=job)
    windows, strides = [200,400,600], [50,100,200]
    if job != 'smoke':
        windows = [int(job.split('-')[0][1:])]
        strides = [int(job.split('-')[1][1:])]
    source = (f'Config.SMOKE = {job == "smoke"!r}\n'
              f'Config.WINDOWS_MS = {windows!r}\nConfig.STRIDES_MS = {strides!r}\n'
              f'Config.AUTOMATION = {provenance!r}\n')
    notebook['cells'].insert(len(notebook['cells'])-1, dict(cell_type='code',id='automation-overrides',
        metadata={},source=source.splitlines(True),outputs=[],execution_count=None))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    write_json(folder/'experiment.ipynb', notebook)
    # Kaggle derives a new kernel's slug from its title. Keep title and id identical.
    write_json(folder/'kernel-metadata.json', dict(id=reference,title=f'DB7 Diag {run_id} {attempt} {job}',
        code_file='experiment.ipynb', language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True, enable_internet=False, machine_shape='NvidiaTeslaT4',
        dataset_sources=['rayaanraza1/ninapro-db7'],kernel_sources=[],competition_sources=[],model_sources=[]))
    provenance['sha256'] = hashlib.sha256((folder/'experiment.ipynb').read_bytes()).hexdigest()
    write_json(folder/'submission.json', provenance)
    return folder, reference, windows, strides


def command(arguments, log, timeout=300):
    result = subprocess.run(['kaggle',*arguments],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,timeout=timeout)
    output = result.stdout
    token = os.environ.get('KAGGLE_API_TOKEN','')
    if token:
        output = output.replace(token,'[REDACTED]')
    with log.open('a',encoding='utf-8') as stream:
        stream.write(output+'\n')
    print(output,flush=True)
    return result.returncode, output


def read_csv(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8'))))


def validate(folder, job, reference, windows, strides):
    files = list(folder.rglob('db7_diagnostic_study_*.zip'))
    assert len(files) == 1, f'Expected one diagnostic ZIP, found {len(files)}'
    subjects = [1] if job == 'smoke' else list(range(1,23))
    with zipfile.ZipFile(files[0]) as archive:
        names = set(archive.namelist())
        assert archive.testzip() is None
        assert not {'FAILURE.txt','SUMMARY_FAILURE.txt'} & names
        manifest = json.loads(archive.read('run_manifest.json'))
        completion = json.loads(archive.read('completion.json'))
        assert completion['success'] and manifest['automation']['kaggle_ref'] == reference
        assert manifest['labels'] == list(range(1,18)) and manifest['subjects'] == subjects
        assert not manifest['window_selection'] and manifest['windows_ms'] == windows and manifest['strides_ms'] == strides
        assert completion['cnn_fits'] == len(subjects)*len(windows)*len(strides)*2
        common_reference = {}
        for window in windows:
            for stride in strides:
                for subject in subjects:
                    base = f'w{window}/s{stride}/S{subject:02}'
                    inventory = read_csv(archive,base+'/repetition_inventory.csv')
                    for gesture in range(1,18):
                        rows = [row for row in inventory if int(row['gesture']) == gesture]
                        assert len(rows) == 6 and len({row['native_repetition'] for row in rows}) == 6
                        assert {split:sum(row['split']==split for row in rows) for split in ['train','validation','test']} == dict(train=4,validation=1,test=1)
                    for variant in ['original_acc','centered_acc']:
                        path = base+'/'+variant
                        fit = json.loads(archive.read(path+'/fit_manifest.json'))
                        assert fit['no_refit'] and not fit['window_selection']
                        assert fit['acc_centering'] == (variant == 'centered_acc')
                        assert fit['epochs_run'] == 2 if job == 'smoke' else 20 <= fit['epochs_run'] <= 150
                        assert json.loads(archive.read(path+'/activation_audit_manifest.json'))['running_statistics_unchanged']
                        for split in ['train','validation','test']:
                            for name in ['predictions.csv','probabilities.npz','features_embeddings.npz','metrics.csv','gesture_errors.csv']:
                                assert path+'/'+split+'/'+name in names
                        for name in ['attention.npz','training_similarity.csv','feature_correct_wrong.csv','phase_errors.csv',
                                     'repetition_failures.csv','output_stride_offsets.csv','waveform_cases.npz','gesture_error_bars.png']:
                            assert path+'/test/'+name in names
                        predictions = read_csv(archive,path+'/test/predictions.csv')
                        common = {(row['gesture'],row['native_repetition'],row['window_end']) for row in predictions if row['evaluate_common'] == 'True'}
                        assert common and common == common_reference.setdefault(subject,common)
                        for row in read_csv(archive,path+'/test/gesture_errors.csv'):
                            assert int(row['cnn_correct'])+int(row['cnn_wrong']) == int(row['windows'])
                            assert int(row['gate_correct'])+int(row['gate_wrong']) == int(row['windows'])
                            assert int(row['gate_correct'])-int(row['cnn_correct']) == int(row['recovered'])-int(row['harmed'])
        for name in ['run_manifest.json','completion.json','all_subject_metrics.csv','all_gesture_errors.csv','mean_subject_metrics.csv']:
            (folder/name).write_bytes(archive.read(name))
    write_json(folder/'validation.json',dict(success=True,job=job,cnn_fits=completion['cnn_fits'],
        zip_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest()))
    print(f'Verified {job}: {completion["cnn_fits"]} CNN fits and all required diagnostics.',flush=True)


def main():
    job = os.environ.get('DIAGNOSTIC_JOB','smoke')
    folder, reference, windows, strides = prepare(job)
    if os.environ.get('PREPARE_ONLY') == '1':
        return
    assert os.environ.get('KAGGLE_API_TOKEN'), 'KAGGLE_API_TOKEN is required in GitHub Secrets.'
    output = ROOT/'diagnostic_results'/job
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'kernel.json',dict(ref=reference,url='https://www.kaggle.com/code/'+reference))
    print('Kaggle URL: https://www.kaggle.com/code/'+reference,flush=True)
    try:
        code,_ = command(['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4'],output/'push.log')
        assert code == 0, 'Submission failed; inspect push.log before retrying.'
        deadline = time.monotonic()+(60 if job == 'smoke' else 310)*60
        status_failures = 0
        while True:
            code,status = command(['kernels','status',reference],output/'status.log')
            status_failures = status_failures+1 if code else 0
            assert status_failures < 5, 'Repeated status failures; kernel may still be active.'
            if code == 0 and 'COMPLETE' in status.upper():
                break
            if any('STATUS.'+value in status.upper() for value in ['ERROR','FAILED','CANCELED','CANCELLED']):
                raise RuntimeError('Kaggle execution failed; inspect downloaded log and failure ZIP.')
            if time.monotonic() > deadline:
                raise TimeoutError('Kaggle may still be running. Check the saved URL before retrying.')
            time.sleep(60)
    finally:
        for retry in range(3):
            code,_ = command(['kernels','output',reference,'-p',str(output),'--file-pattern',r'.*\.(zip|log)$','--force'],
                             output/'download.log',timeout=600)
            if code == 0 and list(output.glob('db7_diagnostic_study_*.zip')):
                break
            if retry < 2:
                time.sleep(30)
    validate(output,job,reference,windows,strides)


if __name__ == '__main__':
    main()
