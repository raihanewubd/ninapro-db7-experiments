"""Package the hierarchical BRB with pinned parent evidence and readable cells."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]

def cell(kind,text):
    obj={'cell_type':kind,'id':hashlib.sha256((kind+text).encode()).hexdigest()[:12],
         'metadata':{},'source':text.splitlines(True)}
    if kind=='code':obj.update(execution_count=None,outputs=[])
    return obj

def main():
    core=(ROOT/'scripts/db7_030_hudgins.py').read_text(encoding='utf-8')
    source=(ROOT/'scripts/db7_031_brb.py').read_text(encoding='utf-8')
    parent=(ROOT/'data/db7-031-parent.json').read_text(encoding='utf-8')
    cells=[cell('markdown','''# DB7-031 — train a hierarchical BRB on all 48 Hudgins features

## Question and agreed model
Can a BRB directly classify the 17 Exercise B gestures from 48 EMG time-domain features?
The user selected a hierarchical design: each of 12 channel modules receives MAV, WL, ZC and SSC.
Every feature has **three reference values**, producing **81 rules per channel and 972 rules total**.
Each rule has 17 consequent beliefs. Channel beliefs are fused with a complete-belief ER calculation.
The model retains all 48 input features. It is not a full 3^48 rule grid and is not an SI-to-I switch.

## What is trained
Adam jointly learns **consequent belief logits, 972 rule weights, 48 attribute weights and 12 channel-fusion weights** through the final gesture cross-entropy loss.
Initial consequent beliefs equal smoothed fitting-class priors; rule and attribute weights start at one; fusion weights start at 1/12.
Rule weights are positive exponentials. Attribute weights are positive and normalized by the maximum within each module.
Fusion weights are positive and sum to one. Consequent beliefs use softmax, so each rule sums to one.
Raw rule/attribute/fusion logits are projected to [-2,2]; belief logits to [-12,12].

## Preserved data protocol
S1–S20; E1 restimulus labels1–17; rest excluded; training repetitions1/3/4/6; test2/5.
200ms windows at2kHz;50ms training stride and10ms test stride, matching DB7-030.
EMG filtering: per-gesture-repetition fourth-order20–450Hz Butterworth followed by50Hz Q30 notch, both zero-phase.
Features are the same 48 MAV/WL/ZC/SSC values. No feature selection or neural retraining.
BRB membership scaling uses training-derived raw feature references, not a new test-fitted scaler.

## Validation and reference values
Four leave-one-training-repetition-out BRB fits per subject/seed select the earliest epoch with highest mean validation accuracy.
Every fold fits feature references from its three fitting repetitions:5th,50th,95th percentiles. Tied references receive a tiny deterministic expansion to remain ordered.
References are fixed during gradient training; rule, attribute and consequent parameters are trained.
For each outer fold, ZC/SSC threshold-ratio screening is nested within those three fitting repetitions using the existing LDA procedure; its amplitude scales also exclude the inner held repetition.
Final BRB training uses all four training repetitions and the verified DB7-030 final threshold choice. Test labels never select epochs, references or thresholds.

## Prespecified optimization and compute
Three genuine BRB seeds42/43/44; up to60 validation epochs; batch512; Adam learning rate0.02; cosine schedule T_max60; regularization0.001; gradient norm limit5.
These are BRB-specific optimization settings, not the earlier CNN's13-epoch schedule.
240 validation BRB fits plus60 final BRB fits. Two processes use cuda:0 and cuda:1 on Kaggle T4x2, dividing subjects between GPUs.
The run aborts if two GPUs are unavailable. A CPU synthetic self-test checks ER equivalence, gradients for every learned parameter group and loss reduction before full fitting.

## Evaluation limits
Primary comparator: matched all48-feature regularized LDA. SI also uses inertial signals, so its accuracy is not a modality-matched comparator.
These recordings were previously inspected; the experiment remains exploratory. Overlapping windows and multiple seeds are not independent trials.
Mean subject accuracy, pooled accuracy, macro-F1, per-gesture errors and paired subject-level uncertainty are reported.
No claim of98% accuracy is made before results.
'''),
        cell('markdown','## 1. Write the unchanged DB7-030 preprocessing implementation'),
        cell('code','%%writefile /kaggle/working/db7_030_hudgins.py\n'+core),
        cell('markdown','## 2. Write the trainable hierarchical BRB implementation\nThe implementation below exposes membership matching, ER aggregation, training, rule export, validation and evaluation as separate functions.'),
        cell('code','%%writefile /kaggle/working/db7_031_brb.py\n'+source),
        cell('markdown','## 3. Record verified parent thresholds and raw-data hashes\nThese choices came only from DB7-030 training repetitions. Parent test accuracy is used as a reproduction check for the LDA comparator, never to tune the BRB.'),
        cell('code','%%writefile /kaggle/working/db7_031_parent.json\n'+parent),
        cell('markdown','## 4. Train on both GPUs and verify completion\nInitial/trained rule beliefs, rule weights, attribute weights, fusion weights and raw reference values are exported per subject/seed. Test predictions are saved only after epoch selection and final fitting.'),
        cell('code',"""import os,sys,subprocess,json
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
subprocess.run([sys.executable,'/kaggle/working/db7_031_brb.py'],check=True)
out=Path('/kaggle/working/db7_031_results')
done=json.loads((out/'completion.json').read_text())
assert done['success'] and done['final_brb_fits']==60 and done['validation_brb_fits']==240
assert done['devices_used']==['cuda:0','cuda:1'] and not done['test_used_for_selection']
assert done['test_window_seed_evaluations']==695163 and done['all_parameter_groups_updated']
print(json.dumps(done,indent=2))
"""),
        cell('markdown','## 5. Present measured accuracy and subject/gesture diagnostics'),
        cell('code',"""import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
display(Markdown((out/'REPORT.md').read_text()))
metrics=pd.read_csv(out/'subject_seed_metrics.csv')
display(metrics)
by_subject=metrics.groupby('subject')[['accuracy','lda_accuracy']].mean()*100
ax=by_subject.plot.bar(figsize=(14,5),ylim=(0,100),ylabel='Test accuracy (%)',title='Hierarchical BRB versus matched Hudgins LDA')
plt.tight_layout();plt.savefig(out/'subject_accuracy.png',dpi=160);plt.show()
g=pd.read_csv(out/'subject_gesture_metrics.csv')
grid=g.groupby(['subject','gesture']).recall.mean().unstack()*100
fig,ax=plt.subplots(figsize=(12,7));im=ax.imshow(grid,vmin=0,vmax=100,aspect='auto',cmap='viridis')
ax.set(xticks=range(17),xticklabels=range(1,18),yticks=range(20),yticklabels=range(1,21),xlabel='Gesture',ylabel='Subject',title='BRB mean recall by subject and gesture')
fig.colorbar(im,ax=ax,label='Recall (%)');fig.tight_layout();fig.savefig(out/'gesture_recall.png',dpi=160);plt.show()
# Refresh the archive to include these plots.
import zipfile
with zipfile.ZipFile('/kaggle/working/db7_031_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for file in sorted(out.rglob('*')):
        if file.is_file():archive.write(file,file.relative_to(out))
print('Download db7_031_results.zip for all models, rules, weights and predictions.')
""")]
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    dest=ROOT/'db7-031-hudgins-hierarchical-brb.ipynb'
    with dest.open('w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(notebook,indent=1))
    manifest={'experiment_id':'DB7-031','notebook':dest.name,'notebook_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'parent_sha256':hashlib.sha256(parent.encode()).hexdigest(),
        'raw_dataset':'rayaanraza1/ninapro-db7','subjects':list(range(1,21)),'seeds':[42,43,44],
        'features':48,'references_per_feature':3,'rules_per_channel':81,'modules':12,
        'trained_parameters':['consequent beliefs','rule weights','attribute weights','fusion weights'],
        'validation_fits':240,'final_fits':60,'gpus_required':2}
    (ROOT/'db7-031-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(dest,manifest['notebook_sha256'])

if __name__=='__main__':main()
