"""Analyze saved modality predictions; no training and no Kaggle compute."""
import io,json,os,subprocess,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN='34582518772'; REPO='raihanewubd/ninapro-db7-experiments'
OUT=Path('modality_analysis');OUT.mkdir(exist_ok=True)
ARMS=['emg','acc','gyroscope','magnetometer','emg_acc','emg_acc_gyroscope','emg_acc_magnetometer']
items=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/actions/runs/{RUN}/artifacts?per_page=100']))['artifacts']
metrics=[];preds={};fits=[];keys={};sources=[]
for arm in ARMS:
 item=next(x for x in items if x['name']==f'db7-modalities-{RUN}-1-{arm}')
 dest=Path(arm+'.zip')
 with dest.open('wb') as f:subprocess.run(['gh','api',f'repos/{REPO}/actions/artifacts/{item["id"]}/zip'],stdout=f,check=True)
 with zipfile.ZipFile(dest) as outer:
  assert outer.testzip() is None
  inner_names=[n for n in outer.namelist() if n.endswith('.zip') and '/db7_modality_' in n]
  assert len(inner_names)==1,inner_names
  with zipfile.ZipFile(io.BytesIO(outer.read(inner_names[0]))) as z:
   assert z.testzip() is None
   readj=lambda n:json.loads(z.read(n))
   readcsv=lambda n:pd.read_csv(io.BytesIO(z.read(n)))
   manifest=readj('run_manifest.json');done=readj('completion.json')
   assert done['success'] and done['cnn_fits']==22 and not manifest['smoke'] and manifest['arm']==arm
   assert manifest['subjects']==list(range(1,23))
   assert manifest['automation']['github_run_id']==RUN
   sources.append(dict(arm=arm,artifact_id=item['id'],manifest=manifest))
   metrics.append(readcsv('all_subject_metrics.csv'))
   pp=[]
   for s in range(1,23):
    fit=readj(f'S{s:02}/fit_manifest.json')
    assert fit['window_key_hashes']==keys.setdefault(s,fit['window_key_hashes'])
    fits.append(dict(arm=arm,**fit))
    pp.append(readcsv(f'S{s:02}/test/predictions.csv'))
   preds[arm]=pd.concat(pp).sort_values(['subject','gesture','native_repetition','window_start','window_end']).reset_index(drop=True)
 dest.unlink()
 print('Verified',arm,flush=True)
d=pd.concat(metrics);f=pd.DataFrame(fits);d.to_csv(OUT/'subject_metrics.csv',index=False);f.to_csv(OUT/'fit_summary.csv',index=False)
summary=d.groupby(['modality','split']).accuracy.mean().unstack()*100
summary['test_macro_f1']=d[d.split=='test'].groupby('modality').f1_macro.mean()*100
summary['pooled_test']=pd.Series({a:100*p.correct.mean() for a,p in preds.items()})
summary['wrong_windows']=pd.Series({a:int((~p.correct).sum()) for a,p in preds.items()})
summary['parameters']=f.groupby('arm').parameters.first()
summary=summary.reindex(ARMS);summary.to_csv(OUT/'summary.csv')
t=d[d.split=='test'].pivot(index='subject',columns='modality',values='accuracy')*100;t=t[ARMS];t.to_csv(OUT/'test_by_subject.csv')
identity=['subject','gesture','native_repetition','window_start','window_end']
for a in ARMS:assert preds[a][identity].equals(preds['emg'][identity])
rng=np.random.default_rng(42);pairs=[]
for a,b in [('emg_acc','emg'),('emg_acc','acc'),('emg_acc_gyroscope','emg_acc'),('emg_acc_magnetometer','emg_acc')]:
 delta=(t[a]-t[b]).to_numpy();lo,hi=np.quantile(delta[rng.integers(0,22,(20000,22))].mean(1),[.025,.975])
 pc=preds[a].correct;pb=preds[b].correct
 pairs.append(dict(candidate=a,reference=b,gain_pp=delta.mean(),low_pp=lo,high_pp=hi,subjects_better=int((delta>0).sum()),subjects_worse=int((delta<0).sum()),recovered=int((pc&~pb).sum()),harmed=int((~pc&pb).sum()),both_wrong=int((~pc&~pb).sum())))
pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'paired_gains.csv',index=False)
phase=[];conf=[];rep=[];gesture=[]
for a,p in preds.items():
 p.to_csv(OUT/f'test_predictions_{a}.csv',index=False)
 for phase_name,g in p.groupby('phase'):phase.append(dict(arm=a,phase=phase_name,windows=len(g),error_percent=100*(~g.correct).mean()))
 r=p.groupby(['subject','gesture','native_repetition']).correct.agg(['size','sum']);rep.append(dict(arm=a,whole_trial_wrong=int((r['sum']==0).sum()),mostly_wrong_trials=int((r['sum']/r['size']<.5).sum()),perfect_trials=int((r['sum']==r['size']).sum())))
 c=p[~p.correct].groupby(['gesture','prediction']).size().sort_values(ascending=False).head(8)
 for (truth,pred),n in c.items():conf.append(dict(arm=a,gesture=truth,prediction=pred,wrong=int(n)))
 g=p.groupby(['subject','gesture']).correct.agg(['size','sum']).reset_index();g['arm']=a;g['error_percent']=100*(1-g['sum']/g['size']);gesture.append(g)
phase=pd.DataFrame(phase);rep=pd.DataFrame(rep);conf=pd.DataFrame(conf);gesture=pd.concat(gesture)
for name,frame in [('phase_errors',phase),('trial_errors',rep),('top_confusions',conf),('subject_gesture_errors',gesture)]:frame.to_csv(OUT/(name+'.csv'),index=False)
correct=np.column_stack([preds[a].correct.to_numpy() for a in ARMS]);p=preds['emg'].copy();p['all_wrong']=~correct.any(1)
common=p.groupby('subject').all_wrong.agg(['sum','size']);common['oracle_percent']=100*(1-common['sum']/common['size']);common.to_csv(OUT/'all_models_wrong.csv')
common_g=p[p.all_wrong].groupby(['subject','gesture']).size().sort_values(ascending=False).head(20)
fig,ax=plt.subplots(figsize=(11,5));summary[['train','validation','test']].plot.bar(ax=ax);ax.set(ylabel='Mean subject accuracy (%)',ylim=(0,100));fig.tight_layout();fig.savefig(OUT/'accuracy.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,8));im=ax.imshow(t.values,vmin=40,vmax=100,cmap='viridis',aspect='auto');ax.set_xticks(range(7),ARMS,rotation=35,ha='right');ax.set_yticks(range(22),['S'+str(x) for x in t.index]);fig.colorbar(im,ax=ax,label='Test accuracy (%)');fig.tight_layout();fig.savefig(OUT/'subjects.png',dpi=160);plt.close(fig)
def table(frame):return frame.round(3).to_markdown()
report=f'''# DB7 modality results — run {RUN}
All seven full runs verified: 154 fits. Identical train/validation/test window identities across arms. Exercise B E1 labels 1–17, within-subject 4/1/1 repetitions, 400 ms window, 100 ms stride. One seed per subject. No gate, no augmentation, no per-window ACC centering.

## Accuracy
{table(summary)}

## Paired comparisons
Gains and bootstrap intervals are percentage points, with subjects as resampling units. Intervals are exploratory, conditional on the single repetition split and seed, and not multiplicity adjusted. Recovery/harm are matched test window counts.
{table(pairs)}

## Per-subject test accuracy
{table(t)}

## Whole-trial failures
Each arm has 374 test trials (22 subjects × 17 gestures).
{table(rep)}

## Phase error percentages
{table(phase.pivot(index='arm',columns='phase',values='error_percent'))}

## Training duration and checkpoint
{table(f.groupby('arm')[['selected_epoch','epochs_run']].agg(['mean','min','max']))}

## Shared failures
Test windows per arm: {len(p)}. Windows wrong in every arm: {int(p.all_wrong.sum())}. Hypothetical perfect selection among seven predictions: {common.oracle_percent.mean():.3f}% mean-subject accuracy. This is a hindsight bound, not a deployable classifier or signal-information ceiling.
{table(common)}

Top subject/gesture shared failures:
{common_g.to_string()}

## Leading directional confusions
{table(conf)}

## Interpretation limits
One test repetition per subject/gesture and one training seed. Overlapping windows are not independent trials. This study estimates sensor effects for this within-subject split, not cross-subject transfer. Input channel changes also change first-layer parameter counts. Correct/wrong comparisons identify associations and complementary predictions, not proof of missing physiological features. A superior sensor arm still needs repetition-fold/seed confirmation; test results must not be used to tune and then claim an unbiased new test result.
'''
(OUT/'REPORT.md').write_text(report);(OUT/'sources.json').write_text(json.dumps(sources,indent=2))
print(report,flush=True)
if os.environ.get('GITHUB_STEP_SUMMARY'):Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(report)
