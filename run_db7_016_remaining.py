"""Submit and collect DB7-016 continuation; run only from manual Actions."""
import os,json,subprocess,time,hashlib,shutil,urllib.request
from pathlib import Path
folder=Path('submission');folder.mkdir(exist_ok=True)
out=Path('continuation_results');out.mkdir(exist_ok=True)
username=os.environ['KAGGLE_USERNAME'].strip()
assert username=='beautifulminnd','Unexpected account; review before launching'
source=Path('db7-016-remaining.ipynb')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f8350a5892fecd02d3d9659a0db84ddee1471695c2dded6150f2fae9d1d005bd'
ref=username+'/db7-016-remaining-'+os.environ['GITHUB_RUN_ID']+'-'+os.environ['GITHUB_RUN_ATTEMPT']
shutil.copyfile(source,folder/'experiment.ipynb')
metadata=dict(id=ref,title='DB7 016 remaining 64 fits',code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',dataset_sources=['rayaanraza1/ninapro-db7'],competition_sources=[],kernel_sources=[],model_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata))
def cli(args,timeout=180,check=True):
 p=subprocess.run(['kaggle',*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
 print(p.stdout,flush=True)
 if check and p.returncode:raise RuntimeError('Kaggle command failed: '+args[0])
 return p
# Reads must succeed before GPU submission.
cli(['datasets','files','rayaanraza1/ninapro-db7'])
req=urllib.request.Request('https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics',data=b'{}',headers={'Authorization':'Bearer '+os.environ['KAGGLE_API_TOKEN'].strip(),'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=45) as response:quota=json.load(response)
assert 'error' not in quota,'Quota request failed'
print('Account quota:',json.dumps(quota),flush=True)
record=dict(experiment_id='DB7-016',parent_version=349618438,remaining_fits=64,kaggle_ref=ref,github_run_id=os.environ['GITHUB_RUN_ID'],notebook_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),status='prepared')
(out/'submission.json').write_text(json.dumps(record,indent=2))
cli(['kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4','--timeout','18000'])
record['status']='submitted';(out/'submission.json').write_text(json.dumps(record,indent=2))
with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write('Submitted 64 missing DB7-016 fits: https://www.kaggle.com/code/'+ref+'\nBoth GPUs required by notebook assertions.\n')
try:
 for _ in range(160):
  r=cli(['kernels','status',ref],check=False)
  (out/'last_status.txt').write_text(r.stdout)
  text=r.stdout.lower()
  if r.returncode==0 and any(x in text for x in ['complete','error','cancel','failed']):break
  time.sleep(90)
 else:raise TimeoutError('Monitoring deadline reached; check Kaggle before retrying')
finally:
 cli(['kernels','output',ref,'-p',str(out),'--file-pattern',r'.*\.zip$','--force'],timeout=1200,check=False)
if 'complete' not in text:raise RuntimeError('Kaggle did not report complete; partial outputs collected when available')
