"""Launch a smoke check, then Exercise B gate and subject CV on Kaggle; collect auditable outputs."""
import argparse, ast, csv, hashlib, io, json, os, re, subprocess, time, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def command(args,path,timeout=300,check=True):
    result=subprocess.run(['kaggle',*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    token=os.environ.get('KAGGLE_API_TOKEN','');output=result.stdout
    if token:output=output.replace(token,'[REDACTED]')
    with path.open('a',encoding='utf-8') as f:f.write(output+'\n')
    print(output,flush=True)
    if check and result.returncode:raise RuntimeError(f'Kaggle command failed; see {path.name}')
    return result.returncode,output

def prepare(scope):
    username=os.environ.get('KAGGLE_USERNAME','')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',username):raise ValueError('Set repository variable KAGGLE_USERNAME')
    rid=os.environ.get('GITHUB_RUN_ID',str(int(time.time())));attempt=os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not rid.isdigit() or not attempt.isdigit():raise ValueError('Invalid run identity')
    title=f'DB7 B Gate CV {rid} {attempt} {scope}';ref=username+'/'+title.lower().replace(' ','-')
    folder=ROOT/'gate_cv_submitted'/scope;folder.mkdir(parents=True,exist_ok=True)
    notebook=json.loads((ROOT/'notebooks/db7-gate-cv-experiment.ipynb').read_text(encoding='utf-8'))
    indexes=[i for i,c in enumerate(notebook['cells']) if c['cell_type']=='code' and 'class Config:' in ''.join(c['source'])]
    if len(indexes)!=1:raise ValueError('Expected exactly one configuration cell')
    provenance=dict(github_commit=os.environ.get('GITHUB_SHA','local'),github_repository=os.environ.get('GITHUB_REPOSITORY',''),
        github_run_id=rid,github_run_attempt=attempt,kaggle_ref=ref,scope=scope)
    code=f"Config.RUN_SUBJECTS = {'[1]' if scope=='smoke' else 'list(range(1,23))'}\nConfig.AUTOMATION = {provenance!r}\nConfig.SCOPE = {scope!r}\nassert Config.DEVICE.type == 'cuda', 'Kaggle GPU is required'\n"
    notebook['cells'].insert(len(notebook['cells'])-1,dict(cell_type='code',id='automation-overrides',metadata={},source=code.splitlines(keepends=True),outputs=[],execution_count=None))
    for c in notebook['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));c['outputs']=[];c['execution_count']=None
    path=folder/'experiment.ipynb';path.write_text(json.dumps(notebook,ensure_ascii=False),encoding='utf-8')
    write_json(folder/'kernel-metadata.json',dict(id=ref,title=title,code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',
        dataset_sources=['rayaanraza1/ninapro-db7'],competition_sources=[],kernel_sources=[],model_sources=[]))
    provenance['submitted_notebook_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();write_json(folder/'submission.json',provenance)
    return folder,ref

def wait(ref,results,minutes):
    deadline=time.monotonic()+minutes*60;failures=0
    while time.monotonic()<deadline:
        code,output=command(['kernels','status',ref],results/'status.log',check=False)
        if code:
            failures+=1
            if failures>=5:raise RuntimeError('Five status-request failures')
        else:
            failures=0;match=re.search(r'has status\s+[\"\x27]([^\"\x27]+)',output,re.I)
            state=match.group(1).rsplit('.',1)[-1].lower() if match else None
            if state=='complete':return
            if state in ('error','failed','cancelled','canceled','cancelacknowledged'):raise RuntimeError(f'Kaggle status {state}')
        time.sleep(60)
    raise TimeoutError('Kaggle may still be running; inspect its URL before retrying')

def validate(results,scope,ref):
    archives=list(results.rglob('db7_gate_cv_*.zip'))
    if len(archives)!=1:raise RuntimeError('Missing or ambiguous experiment ZIP')
    with zipfile.ZipFile(archives[0]) as z:
        if 'FAILURE.txt' in z.namelist():raise RuntimeError('Notebook reported FAILURE.txt')
        manifest=json.loads(z.read('run_manifest.json'));done=json.loads(z.read('completion.json'))
        assert manifest['exercise']=='B' and manifest['labels']==list(range(1,18))
        assert manifest['within_subject_test_evaluated'] is False and manifest['cross_subject_test_evaluated'] is True
        assert manifest['AUTOMATION']['kaggle_ref']==ref and done['success']
        assert done['within_subject_count']==(1 if scope=='smoke' else 22)
        assert done['cross_subject_fold_count']==(1 if scope=='smoke' else 5)
        expected=[f'within_subject/S{s:02}' for s in ([1] if scope=='smoke' else range(1,23))]
        expected += [f'cross_subject/fold_{f}' for f in ([0] if scope=='smoke' else range(1,6))]
        for prefix in expected:
            for suffix in ['best_selection.pt','history.csv','emg_lda.joblib','normalizer.npz','fit_manifest.json','validation/probabilities.npz','validation/gesture_recovery.csv']:
                assert prefix+'/'+suffix in z.namelist(),prefix+'/'+suffix
            fit=json.loads(z.read(prefix+'/fit_manifest.json'))
            if prefix.startswith('cross_subject'):
                assert prefix+'/test/probabilities.npz' in z.namelist()
                tr,va,te=map(set,[fit['train_subjects'],fit['validation_subjects'],fit['test_subjects']])
                assert not(tr&va or tr&te or va&te)
        folds=json.loads(z.read('fold_assignments.json'))
        assert len(folds)==5 and sorted(s for f in folds for s in f['test_subjects'])==list(range(1,23))
        for name in ['model_summary.csv','subject_metrics.csv','completion.json','run_manifest.json','READ_ME_RESULTS.md','fold_assignments.json','aggregate_subject_mean.csv']:(results/name).write_bytes(z.read(name))
    print('Verified complete Exercise B gate and subject CV:',archives[0],flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    requested=os.environ.get('GATE_CV_SCOPE','all')
    if requested not in ('smoke','all'):raise ValueError('GATE_CV_SCOPE must be smoke or all')
    if not args.prepare_only and not os.environ.get('KAGGLE_API_TOKEN'):raise ValueError('Missing KAGGLE_API_TOKEN secret')
    for scope in (['smoke','all'] if requested=='all' else ['smoke']):
        folder,ref=prepare(scope)
        if args.prepare_only:continue
        results=ROOT/'gate_cv_results'/scope;results.mkdir(parents=True,exist_ok=True)
        write_json(results/'kernel.json',dict(ref=ref,url=f'https://www.kaggle.com/code/{ref}'))
        print('Kaggle URL:',f'https://www.kaggle.com/code/{ref}',flush=True)
        try:
            command(['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4'],results/'push.log')
            wait(ref,results,45 if scope=='smoke' else 280)
        finally:
            for retry in range(3):
                try:
                    code,_=command(['kernels','output',ref,'-p',str(results),'--file-pattern',r'.*\.zip$','--force'],results/'download.log',timeout=600,check=False)
                    if code==0 and list(results.rglob('db7_gate_cv_*.zip')):break
                except Exception as exc:print('Collection attempt:',type(exc).__name__,flush=True)
                if retry<2:time.sleep(30)
        validate(results,scope,ref)
    print('Payloads prepared; no API calls.' if args.prepare_only else 'Gate and CV experiment completed and verified.',flush=True)

if __name__=='__main__':main()
