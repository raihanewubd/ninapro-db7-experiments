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


ARMS = {'all_sensors': ('emg','acc','gyro','mag')}


def prepare(job):
    assert job == 'smoke' or job in ARMS
    user = os.environ['KAGGLE_USERNAME']
    assert re.fullmatch(r'[A-Za-z0-9_-]+', user)
    run_id = os.environ.get('GITHUB_RUN_ID','0')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT','1')
    assert run_id.isdigit() and attempt.isdigit()
    slug = f'db7-mod-{run_id}-{attempt}-{job.replace("_","-")}'
    reference = f'{user}/{slug}'
    folder = ROOT/'modality_submitted'/job
    path = ROOT/'notebooks'/('db7-'+('all-sensors' if job=='smoke' else job.replace('_','-'))+'.ipynb')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    provenance = dict(kaggle_ref=reference,github_run_id=run_id,github_run_attempt=attempt,
        github_commit=os.environ.get('GITHUB_SHA','local'),job=job,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source = f'Config.AUTOMATION = {provenance!r}\nConfig.SMOKE = {job == "smoke"!r}\n'
    if job == 'smoke':
        source += f'for arm, modalities in {ARMS!r}.items():\n    Config.ARM, Config.MODALITIES = arm, modalities\n    run_modality_screen()\n'
    else:
        source += 'RESULTS_DIRECTORY = run_modality_screen()\n'
    notebook['cells'][-1]['source'] = source.splitlines(True)
    for cell in notebook['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    write_json(folder/'experiment.ipynb',notebook)
    write_json(folder/'kernel-metadata.json',dict(id=reference,title=slug,code_file='experiment.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        machine_shape='NvidiaTeslaT4',dataset_sources=['rayaanraza1/ninapro-db7'],kernel_sources=[],competition_sources=[],model_sources=[]))
    write_json(folder/'submission.json',provenance)
    return folder, reference


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


def validate(folder, job, reference):
    files=list(folder.rglob('db7_modality_*.zip'))
    expected=list(ARMS) if job=='smoke' else [job]
    assert len(files)==len(expected), f'Expected {len(expected)} result archives, found {len(files)}'
    subjects=[1] if job=='smoke' else list(range(1,23))
    seen=set(); hashes={}
    for path in files:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert 'FAILURE.txt' not in archive.namelist()
            manifest=json.loads(archive.read('run_manifest.json'))
            done=json.loads(archive.read('completion.json'))
            arm=manifest['arm']
            assert arm in expected and arm not in seen
            seen.add(arm)
            assert done['success'] and done['cnn_fits']==len(subjects) and done['subjects']==subjects
            assert manifest['subjects']==subjects and manifest['smoke']==(job=='smoke')
            assert manifest['automation']['kaggle_ref']==reference
            assert manifest['labels']==list(range(1,18)) and manifest['window_ms']==400 and manifest['stride_ms']==100
            assert manifest['modalities']==list(ARMS[arm])
            assert json.loads(archive.read('preflight.json'))['success']
            assert len(read_csv(archive,'all_subject_metrics.csv'))==len(subjects)*3
            for s in subjects:
                fit=json.loads(archive.read(f'S{s:02}/fit_manifest.json'))
                assert fit['window_key_hashes']==hashes.setdefault(s,fit['window_key_hashes'])
                assert fit['epochs_run']==2 if job=='smoke' else 20<=fit['epochs_run']<=150
                for split in ['train','validation','test']:
                    assert f'S{s:02}/{split}/predictions.csv' in archive.namelist()
            for name in ['run_manifest.json','completion.json','all_subject_metrics.csv','mean_subject_metrics.csv']:
                target=folder/arm/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(name))
    write_json(folder/'validation.json',dict(success=True,arms=sorted(seen),cnn_fits=len(subjects)*len(seen)))
    print('Verified completed arms:',sorted(seen),flush=True)


def main():
    job = os.environ.get('MODALITY_JOB','smoke')
    folder, reference = prepare(job)
    if os.environ.get('PREPARE_ONLY') == '1':
        return
    assert os.environ.get('KAGGLE_API_TOKEN'), 'KAGGLE_API_TOKEN is required in GitHub Secrets.'
    output = ROOT/'modality_results'/job
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
            if code == 0 and list(output.glob('db7_modality_*.zip')):
                break
            if retry < 2:
                time.sleep(30)
    validate(output,job,reference)


if __name__ == '__main__':
    main()
