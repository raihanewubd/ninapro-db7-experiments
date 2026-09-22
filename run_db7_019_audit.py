"""Submit one CPU Kaggle diagnostic notebook; never retrain DB7-018."""
from pathlib import Path
import os,json,hashlib,subprocess,time,re,sys,zipfile
root=Path.cwd();out=root/'audit_results';out.mkdir(exist_ok=True)
manifest=json.loads((root/'db7-019-manifest.json').read_text())
nb=root/'db7-019-brb-recovery-audit.ipynb'
assert hashlib.sha256(nb.read_bytes()).hexdigest()==manifest['notebook_sha256']
assert os.environ.get('KAGGLE_USERNAME')=='beautifulminnd'
ref=os.environ.get('KAGGLE_MONITOR_REF','').strip()
def call(args):
 p=subprocess.run(['kaggle',*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
 text=p.stdout
 for key in ['KAGGLE_API_TOKEN','KAGGLE_KEY']:
  if os.environ.get(key):text=text.replace(os.environ[key],'[REDACTED]')
 print(text,flush=True)
 return p.returncode,text
if not ref:
 assert os.environ['GITHUB_RUN_ATTEMPT']=='1','Use monitor_ref to resume an existing submission'
 slug='db7-019-recovery-'+os.environ['GITHUB_RUN_ID'];ref='beautifulminnd/'+slug
 submit=root/'audit_submission';submit.mkdir(exist_ok=True)
 (submit/'experiment.ipynb').write_bytes(nb.read_bytes())
 (submit/'kernel-metadata.json').write_text(json.dumps(dict(id=ref,title=slug,code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=[manifest['source_kernel']],model_sources=[])))
 (out/'launch.json').write_text(json.dumps(dict(state='push_attempted',reference=ref,manifest=manifest),indent=2))
 code,text=call(['kernels','push','-p',str(submit)])
 assert code==0 and 'successfully pushed' in text,'Ambiguous push: inspect launch.json before retry; do not duplicate'
else:assert re.fullmatch(r'beautifulminnd/[a-z0-9-]+',ref)
(out/'launch.json').write_text(json.dumps(dict(state='submitted_or_monitoring',reference=ref,manifest=manifest),indent=2))
if os.environ.get('GITHUB_STEP_SUMMARY'):
 with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write('DB7-019 CPU audit: https://www.kaggle.com/code/'+ref+'\n')
deadline=time.monotonic()+7200
while True:
 code,text=call(['kernels','status',ref]);assert code==0
 if re.search(r'\bcomplete\b',text,re.I):break
 if re.search(r'\b(error|failed|cancelled)\b',text,re.I):
  call(['kernels','output',ref,'-p',str(out/'kaggle')])
  raise RuntimeError(text)
 if time.monotonic()>deadline:raise TimeoutError('Resume with monitor_ref; notebook may still run')
 time.sleep(60)
code,text=call(['kernels','output',ref,'-p',str(out/'kaggle')]);assert code==0
archives=list((out/'kaggle').rglob('db7_019_recovery_audit.zip'));assert len(archives)==1
with zipfile.ZipFile(archives[0]) as z:
 assert z.testzip() is None
 done=json.loads(z.read('completion.json'));assert done['success'] and done['subjects']==list(range(1,23)) and done['neural_fits']==0 and done['source_archive_sha256']==manifest['source_archive_sha256']
(out/'verification.json').write_text(json.dumps(done,indent=2));print('DB7-019 VERIFIED COMPLETE',flush=True)
