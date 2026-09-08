"""GitHub orchestrates; Kaggle runs the existing DB7 notebook.

No token is written to files or inserted into the notebook. Each seed receives
a unique private Kaggle notebook, avoiding stale-output or version collisions.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def parse_seeds(text):
    items = text.split(',')
    if not 1 <= len(items) <= 3 or any(not re.fullmatch(r'\d{1,9}', s.strip()) for s in items):
        raise ValueError('Use one to three comma-separated integer seeds, e.g. 42,43,44.')
    seeds = [int(s.strip()) for s in items]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct.')
    return seeds


def prepare(config, username, seed, scope, run_id, attempt):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', username):
        raise ValueError('Set KAGGLE_USERNAME to your Kaggle username, not email or display name.')
    if scope not in ('smoke', 'all'):
        raise ValueError('Scope must be smoke or all.')
    if not re.fullmatch(r'\d+', run_id) or not re.fullmatch(r'\d+', attempt):
        raise ValueError('Run ID and attempt must be numeric.')
    title = f'DB7 AV {run_id} {attempt} Seed {seed}'
    slug = title.lower().replace(' ', '-')
    ref = f'{username}/{slug}'
    folder = ROOT/'submitted'/f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    nb = json.loads((ROOT/config['notebook']).read_text(encoding='utf-8'))
    config_cells = [i for i,c in enumerate(nb['cells'])
                    if c['cell_type']=='code' and 'class Config:' in ''.join(c['source'])]
    if len(config_cells) != 1:
        raise ValueError('Expected one Config cell in the supplied notebook.')
    provenance = dict(github_commit=os.environ.get('GITHUB_SHA', 'local-prepare'),
        github_repository=os.environ.get('GITHUB_REPOSITORY', ''),
        github_run_id=run_id, github_run_attempt=attempt,
        kaggle_ref=ref, model_seed=seed, split_seed=42, scope=scope,
        dataset_sources=config['dataset_sources'])
    overrides = (
        '# GitHub experiment overrides; credentials are never embedded.\n'
        f'Config.MODEL_SEED = {seed}\n'
        'Config.SEED = 42\n'
        f"Config.RUN_SUBJECTS = {'[1]' if scope=='smoke' else 'list(range(1,23))'}\n"
        "Config.DIAG_EXPERIMENTS = [('baseline','both'),('amplitude_velocity','both')]\n"
        'Config.EVALUATE_TEST = False\n'
        'Config.REFIT_ON_TRAIN_PLUS_VAL = False\n'
        f'Config.AUTOMATION = {provenance!r}\n'
        "assert Config.DEVICE.type == 'cuda', 'GPU missing: verify Kaggle accelerator/quota.'\n"
        "print('AUTOMATION:', Config.AUTOMATION)\n")
    nb['cells'].insert(config_cells[0]+1, dict(cell_type='code',metadata={},outputs=[],
        execution_count=None,source=overrides.splitlines(keepends=True)))
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]; cell['execution_count']=None
            ast.parse(''.join(cell['source']))
    notebook_path=folder/'experiment.ipynb'
    notebook_path.write_text(json.dumps(nb,ensure_ascii=False),encoding='utf-8')
    metadata=dict(id=ref,title=title,code_file='experiment.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,
        machine_shape=config['accelerator'],dataset_sources=config['dataset_sources'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    write_json(folder/'kernel-metadata.json',metadata)
    provenance['submitted_notebook_sha256']=hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    write_json(folder/'submission.json',provenance)
    print('Kaggle notebook:',f'https://www.kaggle.com/code/{ref}',flush=True)
    return folder,ref


def command(args, log_path, timeout=180, check=True):
    result=subprocess.run(['kaggle',*args],text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,timeout=timeout)
    # Avoid persisting a credential even if an unexpected dependency prints it.
    output=result.stdout
    token=os.environ.get('KAGGLE_API_TOKEN','')
    if token: output=output.replace(token,'[REDACTED]')
    with log_path.open('a',encoding='utf-8') as stream:
        stream.write(output+'\n')
    print(output,flush=True)
    if check and result.returncode:
        raise RuntimeError(f'Kaggle {args[1] if len(args)>1 else args[0]} failed; see saved log.')
    return result.returncode,output


def parse_status(output):
    match=re.search(r'has status\s+[\"\x27]([^\"\x27]+)',output,re.I)
    return match.group(1).rsplit('.',1)[-1].lower() if match else None


def wait_for_completion(ref, directory, max_minutes):
    deadline=time.monotonic()+max_minutes*60
    failures=0
    while time.monotonic()<deadline:
        code,output=command(['kernels','status',ref],directory/'status.log',check=False)
        if code:
            failures+=1
            if failures>=5: raise RuntimeError('Five consecutive status API failures; check Kaggle directly.')
        else:
            failures=0
            status=parse_status(output)
            if status=='complete': return
            if status in ('error','failed','cancelled','canceled','cancelacknowledged'):
                raise RuntimeError(f'Kaggle run ended with status {status}.')
        time.sleep(60)
    raise TimeoutError('Monitoring time limit reached. Kaggle may still be running; check its link before starting another run.')


def validate_zip(directory, scope, seed):
    archives=list(directory.rglob('db7_amplitude_velocity_*.zip'))
    if len(archives)!=1:
        raise RuntimeError(f'Expected one diagnostic ZIP, found {len(archives)}. Check the Kaggle run.')
    with zipfile.ZipFile(archives[0]) as z:
        if 'FAILURE.txt' in z.namelist():
            raise RuntimeError('Notebook ZIP contains FAILURE.txt; diagnostic run was not successful.')
        manifest=json.loads(z.read('run_manifest.json'))
        if manifest.get('MODEL_SEED')!=seed or manifest.get('EVALUATE_TEST') is not False:
            raise RuntimeError('Returned output manifest does not match requested experiment.')
        actual=list(csv.DictReader(io.StringIO(z.read('model_subject_summary.csv').decode())))
        subjects=[1] if scope=='smoke' else list(range(1,23))
        expected={(s,v,'both') for s in subjects for v in ('baseline','amplitude_velocity')}
        keys=[(int(r['subject']),r['variant'],r['modality']) for r in actual]
        if set(keys)!=expected or len(keys)!=len(expected):
            raise RuntimeError('Incomplete or mismatched subject/model results.')
        for name in ('model_subject_summary.csv','paired_preprocessing_comparison.csv','READ_ME_RESULTS.md'):
            (directory/name).write_bytes(z.read(name))
    print('Verified diagnostic ZIP:',archives[0],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true',help='Build payloads without any Kaggle API calls.')
    args=parser.parse_args()
    config=json.loads((ROOT/'experiment.json').read_text())
    username=os.environ.get('KAGGLE_USERNAME','')
    seeds=parse_seeds(os.environ.get('MODEL_SEEDS','42'))
    scope=os.environ.get('EXPERIMENT_SCOPE','smoke')
    run_id=os.environ.get('GITHUB_RUN_ID',str(int(time.time())))
    attempt=os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not args.prepare_only and not os.environ.get('KAGGLE_API_TOKEN'):
        raise ValueError('Missing repository secret KAGGLE_API_TOKEN.')
    for seed in seeds:
        folder,ref=prepare(config,username,seed,scope,run_id,attempt)
        if args.prepare_only: continue
        results=ROOT/'results'/f'seed-{seed}'
        results.mkdir(parents=True,exist_ok=True)
        write_json(results/'kernel.json',dict(ref=ref,url=f'https://www.kaggle.com/code/{ref}'))
        completed=False
        # Push is intentionally not retried automatically: its response could be
        # lost after Kaggle accepts it, and retrying could launch another version.
        try:
            command(['kernels','push','-p',str(folder),'--accelerator',config['accelerator']],results/'push.log')
            wait_for_completion(ref,results,int(config['max_minutes_per_seed']))
            completed=True
        finally:
            # Also try to collect partial ZIPs and logs after failures/timeouts.
            for attempt_download in range(3):
                try:
                    code,_=command(['kernels','output',ref,'-p',str(results),
                        '--file-pattern',r'.*\.zip$','--force'],results/'download.log',timeout=300,check=False)
                    if code==0 and (not completed or list(results.rglob('db7_amplitude_velocity_*.zip'))): break
                except Exception as exc:
                    print('Output collection attempt failed:',type(exc).__name__,flush=True)
                if attempt_download<2: time.sleep(30)
        validate_zip(results,scope,seed)
    print('Finished. No external calls were made.' if args.prepare_only else 'All requested experiments completed and outputs verified.')


if __name__=='__main__':
    main()
