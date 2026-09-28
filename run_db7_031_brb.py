"""Submit DB7-031 once; download and verify the completed Kaggle experiment."""
from pathlib import Path
import os,json,hashlib,re,subprocess,time,zipfile

root=Path(__file__).resolve().parent
out=root/'db7_031_action_results';out.mkdir(exist_ok=True)
manifest=json.loads((root/'db7-031-manifest.json').read_text())
nb=root/manifest['notebook']
assert hashlib.sha256(nb.read_bytes()).hexdigest()==manifest['notebook_sha256']
assert os.environ.get('KAGGLE_USERNAME')=='beautifulminnd'

def call(args,timeout=1800):
    result=subprocess.run(['kaggle',*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    message=result.stdout
    for key in ('KAGGLE_API_TOKEN','KAGGLE_KEY'):
        if os.environ.get(key):message=message.replace(os.environ[key],'[REDACTED]')
    print(message,flush=True)
    return result.returncode,message

ref=os.environ.get('KAGGLE_MONITOR_REF','').strip()
if ref:
    assert re.fullmatch(r'beautifulminnd/db7-031-brb-[0-9]+',ref)
else:
    assert os.environ.get('GITHUB_RUN_ATTEMPT')=='1','Use monitor_ref for an existing kernel; do not submit a duplicate'
    ref='beautifulminnd/db7-031-brb-'+os.environ['GITHUB_RUN_ID']
    stage=root/'db7_031_submission';stage.mkdir(exist_ok=True)
    (stage/'experiment.ipynb').write_bytes(nb.read_bytes())
    metadata={'id':ref,'title':ref.split('/')[-1],'code_file':'experiment.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':[manifest['raw_dataset']],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (stage/'kernel-metadata.json').write_text(json.dumps(metadata))
    (out/'launch.json').write_text(json.dumps({'state':'push_attempted','kernel':ref,'manifest':manifest},indent=2))
    code,message=call(['kernels','push','-p',str(stage),'--accelerator','NvidiaTeslaT4','--timeout','43200'])
    if code or 'successfully pushed' not in message.lower() or 'not valid dataset sources' in message.lower():
        raise RuntimeError('Push failed or ambiguous; inspect launch.json before resubmission')
(out/'launch.json').write_text(json.dumps({'state':'submitted_or_monitoring','kernel':ref,'manifest':manifest},indent=2))
if os.environ.get('GITHUB_STEP_SUMMARY'):
    with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write(f'DB7-031: https://www.kaggle.com/code/{ref}\n')
deadline=time.monotonic()+18000
while True:
    code,message=call(['kernels','status',ref])
    if code:raise RuntimeError('Kaggle status call failed')
    if re.search(r'\bcomplete\b',message,re.I):break
    if re.search(r'\b(error|failed|cancelled)\b',message,re.I):
        call(['kernels','output',ref,'-p',str(out/'kaggle'),'--file-pattern',r'.*\.(log|json|txt)$'])
        raise RuntimeError('Kaggle experiment failed; inspect downloaded logs')
    if time.monotonic()>deadline:raise TimeoutError('Kaggle may still be running; use monitor_ref rather than submitting again')
    time.sleep(60)
code,_=call(['kernels','output',ref,'-p',str(out/'kaggle'),'--file-pattern',r'.*(db7_031_results\.zip|\.log)$'])
assert code==0
archives=list((out/'kaggle').rglob('db7_031_results.zip'));assert len(archives)==1
with zipfile.ZipFile(archives[0]) as z:
    assert z.testzip() is None
    done=json.loads(z.read('completion.json'))
    assert done['success'] and done['subjects']==list(range(1,21)) and done['seeds']==[42,43,44]
    assert done['final_brb_fits']==60 and done['validation_brb_fits']==240 and done['test_window_seed_evaluations']==695163
    assert done['devices_used']==['cuda:0','cuda:1'] and done['all_parameter_groups_updated'] and not done['test_used_for_selection']
    summary=out/'summary';summary.mkdir(exist_ok=True)
    for name in ('completion.json','summary.json','REPORT.md','subject_seed_metrics.csv','subject_gesture_metrics.csv','subject_accuracy.png','gesture_recall.png'):
        (summary/name).write_bytes(z.read(name))
(out/'verification.json').write_text(json.dumps({'completion':done,'kernel':ref,'result_zip_sha256':hashlib.sha256(archives[0].read_bytes()).hexdigest()},indent=2))
print('DB7-031 VERIFIED COMPLETE',ref,flush=True)
