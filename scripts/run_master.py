"""Submit one stage of the same master notebook and validate its output."""
import ast,hashlib,json,os,re,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def prepare(job):
 assert job in ['smoke','windows','cross1','cross2','cross3','cross4','cross5']
 user=os.environ['KAGGLE_USERNAME'];assert re.fullmatch(r'[A-Za-z0-9_-]+',user)
 rid=os.environ.get('GITHUB_RUN_ID','0');attempt=os.environ.get('GITHUB_RUN_ATTEMPT','1');assert rid.isdigit() and attempt.isdigit()
 ref=f'{user}/db7-master-{rid}-{attempt}-{job}';folder=ROOT/'master_submitted'/job
 nb=json.loads((ROOT/'notebooks/db7-master-study.ipynb').read_text(encoding='utf-8'))
 provenance=dict(kaggle_ref=ref,github_run_id=rid,github_run_attempt=attempt,github_commit=os.environ.get('GITHUB_SHA','local'),job=job)
 stage='all' if job=='smoke' else 'windows' if job=='windows' else 'cross';fold=int(job[-1]) if job.startswith('cross') else 1
 src=f'Config.STUDY_STAGE={stage!r}\nConfig.STUDY_SMOKE={job=="smoke"!r}\nConfig.STUDY_FOLD={fold}\nConfig.AUTOMATION={provenance!r}\n'
 nb['cells'].insert(len(nb['cells'])-1,dict(cell_type='code',id='automation-overrides',metadata={},source=src.splitlines(True),outputs=[],execution_count=None))
 for c in nb['cells']:
  if c['cell_type']=='code':ast.parse(''.join(c['source']))
 write(folder/'experiment.ipynb',nb)
 write(folder/'kernel-metadata.json',dict(id=ref,title=f'DB7 Master {rid} {attempt} {job}',code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',dataset_sources=['rayaanraza1/ninapro-db7'],kernel_sources=['raihanbd/db7-b-gate-cv-34371947678-1-all'],competition_sources=[],model_sources=[]))
 provenance['sha256']=hashlib.sha256((folder/'experiment.ipynb').read_bytes()).hexdigest();write(folder/'submission.json',provenance)
 return folder,ref
def command(args,log,timeout=300):
 p=subprocess.run(['kaggle',*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
 out=p.stdout;token=os.environ.get('KAGGLE_API_TOKEN','')
 if token:out=out.replace(token,'[REDACTED]')
 with log.open('a',encoding='utf-8') as f:f.write(out+'\n')
 print(out,flush=True);return p.returncode,out
def validate(folder,job,ref):
 files=list(folder.rglob('db7_master_*.zip'));assert len(files)==1
 with zipfile.ZipFile(files[0]) as z:
  assert z.testzip() is None and 'FAILURE.txt' not in z.namelist();m=json.loads(z.read('run_manifest.json'));c=json.loads(z.read('completion.json'))
  assert c['success'] and m['automation']['kaggle_ref']==ref and m['labels']==list(range(1,18))
  assert 'stage1_existing_errors/phase_by_subject_gesture.csv' in z.namelist()
  if job in ['windows','smoke']:
   assert c['window_subjects']==(1 if job=='smoke' else 22)
   choice=json.loads(z.read('stage2_windows/selection.json'));assert choice['test_used_for_choice'] is False
   for sid in ([1] if job=='smoke' else range(1,23)):
    for w in [200,400,600]:assert f'stage2_windows/S{sid:02}/w{w}/validation/probabilities.npz' in z.namelist()
    for w in {400,choice['chosen_window_ms']}:assert f'stage2_windows/S{sid:02}/w{w}/reserved_test/probabilities.npz' in z.namelist()
  if job!='windows':
   fold=0 if job=='smoke' else int(job[-1]);assert c['cross_folds']==[fold];base=f'stage3_4_cross_subject/fold_{fold}'
   f=json.loads(z.read(base+'/fold.json'));a,b,d=map(set,[f['train_subjects'],f['validation_subjects'],f['test_subjects']]);assert not(a&b or a&d or b&d)
   for variant in ['emg','acc','both','both_acc_centered']:
    assert base+'/'+variant+'/test/probabilities.npz' in z.namelist()
    assert json.loads(z.read(base+'/'+variant+'/activation_audit_manifest.json'))['running_statistics_unchanged']
  for name in ['run_manifest.json','completion.json']:write(folder/name,json.loads(z.read(name)))
 print('Verified master study job:',job,flush=True)
def main():
 job=os.environ.get('MASTER_JOB','smoke');folder,ref=prepare(job)
 if os.environ.get('PREPARE_ONLY')=='1':return
 assert os.environ.get('KAGGLE_API_TOKEN'),'Kaggle token required'
 out=ROOT/'master_results'/job;out.mkdir(parents=True,exist_ok=True);write(out/'kernel.json',dict(ref=ref,url='https://www.kaggle.com/code/'+ref));print('Kaggle URL: https://www.kaggle.com/code/'+ref,flush=True)
 try:
  code,_=command(['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4'],out/'push.log');assert code==0
  deadline=time.monotonic()+(45 if job=='smoke' else 260)*60;failed=0
  while True:
   code,s=command(['kernels','status',ref],out/'status.log');failed=failed+1 if code else 0;assert failed<5
   if 'COMPLETE' in s.upper():break
   if any('STATUS.'+x in s.upper() for x in ['ERROR','FAILED','CANCELED','CANCELLED']):raise RuntimeError('Kaggle job failed')
   if time.monotonic()>deadline:raise TimeoutError('Kaggle may still be active; inspect its URL before retrying')
   time.sleep(60)
 finally:
  for retry in range(3):
   code,_=command(['kernels','output',ref,'-p',str(out),'--file-pattern',r'.*\.zip$','--force'],out/'download.log',timeout=600)
   if code==0 and list(out.glob('db7_master_*.zip')):break
   if retry<2:time.sleep(30)
 validate(out,job,ref)
if __name__=='__main__':main()
