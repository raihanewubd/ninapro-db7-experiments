"""Launch one stage; verify its output before dependent GitHub jobs start."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,time,zipfile

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True,choices=range(1,8));ap.add_argument('--cache-ref',default='');ap.add_argument('--previous-ref',default='');ap.add_argument('--monitor-ref',default='');a=ap.parse_args()
    root=Path.cwd();out=root/'suite_results';out.mkdir(exist_ok=True)
    manifest=json.loads((root/'db7-suite-manifest.json').read_text());entry=manifest['experiments'][a.stage-1]
    nb=root/entry['notebook'];assert hashlib.sha256(nb.read_bytes()).hexdigest()==entry['sha256']
    assert os.environ['KAGGLE_USERNAME']=='beautifulminnd'
    def valid(ref):assert re.fullmatch(r'beautifulminnd/[a-z0-9-]+',ref);return ref
    def call(args):
        result=subprocess.run(['kaggle',*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
        output=result.stdout
        for name in ['KAGGLE_API_TOKEN','KAGGLE_KEY']:
            if os.environ.get(name):output=output.replace(os.environ[name],'[REDACTED]')
        print(output,flush=True);return result.returncode,output
    gpu=a.stage in [1,7];ref=a.monitor_ref
    if not ref:
        assert os.environ.get('GITHUB_RUN_ATTEMPT')=='1','Use a monitor reference; never blindly duplicate a submission'
        slug=f'db7-{21+a.stage:03d}-'+os.environ['GITHUB_RUN_ID'];ref='beautifulminnd/'+slug
        sources=[manifest['source_kernel']] if a.stage==1 else list(dict.fromkeys([valid(a.cache_ref),valid(a.previous_ref)]))
        submit=root/'suite_submission';submit.mkdir(exist_ok=True);(submit/'experiment.ipynb').write_bytes(nb.read_bytes())
        metadata=dict(id=ref,title=slug,code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=gpu,enable_internet=False,dataset_sources=['rayaanraza1/ninapro-db7'] if gpu else [],competition_sources=[],kernel_sources=sources,model_sources=[])
        if gpu:metadata['machine_shape']='NvidiaTeslaT4'
        (submit/'kernel-metadata.json').write_text(json.dumps(metadata))
        (out/'launch.json').write_text(json.dumps(dict(state='push_attempted',reference=ref,stage=a.stage,manifest=entry),indent=2))
        args=['kernels','push','-p',str(submit)]+(['--accelerator','NvidiaTeslaT4','--timeout','43200'] if gpu else [])
        code,text=call(args);assert code==0 and 'successfully pushed' in text,'Inspect launch record before retrying ambiguous submission'
    else:valid(ref)
    (out/'launch.json').write_text(json.dumps(dict(state='submitted_or_monitoring',reference=ref,stage=a.stage,manifest=entry),indent=2))
    with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('reference='+ref+'\n')
    with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write(f"DB7-{21+a.stage:03d}: https://www.kaggle.com/code/{ref}\n")
    deadline=time.monotonic()+19800
    while True:
        code,status=call(['kernels','status',ref]);assert code==0
        if re.search(r'\bcomplete\b',status,re.I):break
        if re.search(r'\b(error|failed|cancelled)\b',status,re.I):
            call(['kernels','output',ref,'-p',str(out/'kaggle'),'--file-pattern',r'.*\.(log|json|txt)$'])
            raise RuntimeError(status)
        if time.monotonic()>deadline:raise TimeoutError('Kaggle may still run; use monitor-ref rather than resubmit')
        time.sleep(60)
    archive_name=f'db7_{21+a.stage:03d}_results.zip'
    code,_=call(['kernels','output',ref,'-p',str(out/'kaggle'),'--file-pattern',re.escape(archive_name)+'$']);assert code==0
    archives=list((out/'kaggle').rglob(archive_name));assert len(archives)==1
    with zipfile.ZipFile(archives[0]) as z:
        assert z.testzip() is None
        done=json.loads(z.read('completion.json'));assert done['success'] and done['stage']==a.stage and done['subjects']==list(range(1,23)) and not done['test_used_for_selection']
        summary=out/'summary';summary.mkdir(exist_ok=True)
        for name in ['completion.json','selected_pipeline.json','summary.csv','REPORT.md','subject_accuracy.png']:
            (summary/name).write_bytes(z.read(name))
    (out/'verification.json').write_text(json.dumps(done,indent=2));print('VERIFIED COMPLETE',ref,flush=True)

if __name__=='__main__':main()
