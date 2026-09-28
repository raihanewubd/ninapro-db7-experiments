"""DB7-031: train a hierarchical 48-input, three-reference BRB classifier.

Twelve channel BRBs each contain 3^4=81 rules. Their 17-class beliefs are
combined using ER. Consequents, rule/attribute weights and fusion weights
are jointly optimized. Test repetitions never select parameters or epochs.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, itertools, json, time, subprocess, sys, zipfile, hashlib
import numpy as np
import pandas as pd
import db7_030_hudgins as features

SEEDS=(42,43,44)
MAX_EPOCHS=60
BATCH=512
LR=.02
PENALTY=.001
NAMES=[f'{f}_CH{c:02}' for f in features.FAMILIES for c in range(1,13)]
BITS=np.array(list(itertools.product(range(3),repeat=4)),dtype=np.int64)
ACTIVE_BITS=np.array(list(itertools.product(range(2),repeat=4)),dtype=np.int64)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def reference_values(x):
    """Three raw-feature references, fitted using fitting repetitions only."""
    r=np.percentile(x,[5,50,95],axis=0).T
    span=np.maximum(r[:,2]-r[:,0],np.maximum(np.max(np.abs(x),axis=0)*1e-6,1e-12))
    tiny=span*1e-6
    r[:,0]=np.minimum(r[:,0],r[:,1]-tiny)
    r[:,2]=np.maximum(r[:,2],r[:,1]+tiny)
    assert r.shape==(48,3) and np.all(np.diff(r,axis=1)>0)
    return r

def encode(x,refs):
    """Only 16 of 81 rules can activate per channel; retain matching degrees."""
    x=np.asarray(x,float).reshape(-1,4,12).transpose(0,2,1)
    r=refs.reshape(4,12,3).transpose(1,0,2)
    low=(x>=r[None,:,:,1]).astype(np.int64)
    a=np.where(low==0,r[None,:,:,0],r[None,:,:,1])
    b=np.where(low==0,r[None,:,:,1],r[None,:,:,2])
    frac=np.clip((x-a)/(b-a),0,1)
    matches=np.where(ACTIVE_BITS[None,None,:,:]==0,1-frac[:,:,None,:],frac[:,:,None,:])
    index=((low[:,:,None,:]+ACTIVE_BITS[None,None,:,:])*np.array([27,9,3,1])).sum(-1)
    valid=np.all(matches>0,axis=-1)
    assert valid.any(axis=-1).all()
    return index,np.log(np.maximum(matches,1e-30)).astype(np.float32),valid

def er_numpy(a,beta,axis):
    product=np.prod(1+a[...,None]*(beta-1),axis=axis)
    ignorance=np.prod(1-a,axis=axis)
    v=product-ignorance[...,None]
    return v/v.sum(-1,keepdims=True)

def make_model(prior):
    import torch
    from torch import nn
    class HierarchicalBRB(nn.Module):
        def __init__(self):
            super().__init__()
            initial=torch.tensor(np.log(prior),dtype=torch.float32).repeat(12,81,1)
            self.belief_logits=nn.Parameter(initial.clone())
            self.register_buffer('initial_logits',initial)
            self.rule_logits=nn.Parameter(torch.zeros(12,81))
            self.attribute_logits=nn.Parameter(torch.zeros(12,4))
            self.fusion_logits=nn.Parameter(torch.zeros(12))
        def weights(self):
            aw=self.attribute_logits.exp()
            aw=aw/aw.amax(dim=-1,keepdim=True)
            return self.belief_logits.softmax(-1),self.rule_logits.exp(),aw,self.fusion_logits.softmax(-1)
        @staticmethod
        def er(a,beta,dim):
            v=torch.prod(1+a.unsqueeze(-1)*(beta-1),dim=dim)-torch.prod(1-a,dim=dim).unsqueeze(-1)
            v=v.clamp_min(1e-12)
            return v/v.sum(-1,keepdim=True)
        def forward(self,index,logmatches,valid):
            beta,rw,aw,fw=self.weights()
            channel=torch.arange(12,device=index.device)[None,:,None]
            score=(logmatches*aw[None,:,None,:]).sum(-1)+self.rule_logits[channel,index]
            activation=score.masked_fill(~valid,-torch.inf).softmax(-1)
            channel_beliefs=self.er(activation,beta[channel,index],dim=2)
            fusion=fw[None,:].expand(len(index),-1)
            return self.er(fusion,channel_beliefs,dim=1)
        def regularizer(self):
            return PENALTY*((self.belief_logits-self.initial_logits).square().mean()+
                self.rule_logits.square().mean()+self.attribute_logits.square().mean()+self.fusion_logits.square().mean())
    return HierarchicalBRB()

def tensors(x,refs,device):
    import torch
    idx,logm,valid=encode(x,refs)
    return (torch.as_tensor(idx,device=device,dtype=torch.long),
            torch.as_tensor(logm,device=device),torch.as_tensor(valid,device=device))

def probabilities(model,data):
    import torch
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(data[0]),BATCH):
            result.append(model(*(v[start:start+BATCH] for v in data)).cpu().numpy())
    p=np.concatenate(result)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-6)
    return p

def fit(x,y,refs,seed,epochs,device,val=None):
    import torch
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    prior=np.bincount(y,minlength=17).astype(float)+.5
    prior/=prior.sum()
    model=make_model(prior).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    train=tensors(x,refs,device);target=torch.as_tensor(y,device=device,dtype=torch.long)
    validation=tensors(val[0],refs,device) if val is not None else None
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=MAX_EPOCHS)
    history=[]
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(y),device=device);total=0.
        for begin in range(0,len(y),BATCH):
            ix=order[begin:begin+BATCH]
            optimizer.zero_grad(set_to_none=True)
            p=model(*(v[ix] for v in train))
            loss=-p[torch.arange(len(ix),device=device),target[ix]].clamp_min(1e-12).log().mean()+model.regularizer()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            with torch.no_grad():
                model.belief_logits.clamp_(-12,12)
                model.rule_logits.clamp_(-2,2)
                model.attribute_logits.clamp_(-2,2)
                model.fusion_logits.clamp_(-2,2)
            total+=float(loss.detach())*len(ix)
        row={'epoch':epoch,'training_objective':total/len(y),'learning_rate':optimizer.param_groups[0]['lr']}
        if validation is not None:
            pred=probabilities(model,validation).argmax(1)
            row['validation_accuracy']=float(np.mean(pred==val[1]))
        history.append(row);scheduler.step()
    return model,initial,history

def select_ratio(runs,allowed):
    """Nested LDA threshold screening: outer validation rep never participates."""
    scores=[]
    for ratio in features.THRESHOLD_RATIOS:
        fold_scores=[]
        for held in allowed:
            fitting=tuple(r for r in allowed if r!=held)
            scale=features.threshold_scale(runs,fitting)
            x,meta=features.extract(runs,0,'train',scale*ratio)
            rep=meta.native_repetition.to_numpy();y=meta.gesture.to_numpy(int)
            train=np.isin(rep,fitting);validation=rep==held
            pred,_=features.predict(features.fit_lda(x[train],y[train]),x[validation])
            fold_scores.append(float(np.mean(pred==y[validation])))
        scores.append({'ratio':ratio,'mean_accuracy':float(np.mean(fold_scores)),'fold_scores':fold_scores})
    chosen=sorted(scores,key=lambda r:(-r['mean_accuracy'],abs(r['ratio']-.01)))[0]['ratio']
    return chosen,scores

def metrics(y,p):
    pred=p.argmax(1);cm=np.bincount(y*17+pred,minlength=289).reshape(17,17)
    diag=np.diag(cm);rows=cm.sum(1);cols=cm.sum(0)
    recall=np.divide(diag,rows,out=np.zeros(17,float),where=rows>0)
    f1=np.divide(2*diag,rows+cols,out=np.zeros(17,float),where=(rows+cols)>0)
    return {'accuracy':float(np.mean(pred==y)),'macro_f1':float(f1.mean()),
            'balanced_accuracy':float(recall.mean()),'windows':len(y)},cm

def export_rules(folder,model,initial,refs):
    import torch
    final={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rows=[];fusion=[]
    for state,weights in [('initial',initial),('trained',final)]:
        model.load_state_dict(weights)
        beta,rw,aw,fw=[v.detach().cpu().numpy() for v in model.weights()]
        for c in range(12):
            fusion.append({'state':state,'channel':c+1,'fusion_weight':float(fw[c])})
            for r,bits in enumerate(BITS):
                row={'state':state,'channel':c+1,'rule':r+1,'rule_weight':float(rw[c,r])}
                for f,name in enumerate(features.FAMILIES):
                    row[f'{name}_level']=('low','medium','high')[bits[f]]
                    row[f'{name}_reference']=float(refs[f*12+c,bits[f]])
                    row[f'{name}_attribute_weight']=float(aw[c,f])
                row.update({f'belief_G{k+1}':float(beta[c,r,k]) for k in range(17)})
                rows.append(row)
    pd.DataFrame(rows).to_csv(folder/'initial_trained_rules.csv.gz',index=False)
    pd.DataFrame(fusion).to_csv(folder/'initial_trained_fusion_weights.csv',index=False)
    pd.DataFrame(refs,columns=['low','medium','high']).assign(feature=NAMES).to_csv(folder/'reference_values.csv',index=False)
    model.load_state_dict(final)
    torch.save({'state_dict':final,'initial_state_dict':initial,'references':refs.tolist()},folder/'model.pt')
    return {k:float((final[k]-initial[k]).abs().max()) for k in ['belief_logits','rule_logits','attribute_logits','fusion_logits']}

def run_subject(raw,subject,out,parent,device,max_epochs=MAX_EPOCHS,seeds=SEEDS):
    import torch
    folder=out/f'S{subject:02}';folder.mkdir(parents=True,exist_ok=True)
    runs,raw_source=features.load_subject(raw,subject)
    old=parent[str(subject)]
    assert raw_source['raw_emg_sha256']==old['raw_emg_sha256'],'Raw input differs from DB7-030'
    prepared=[];audit=[]
    for held in features.TRAIN_REPS:
        allowed=tuple(r for r in features.TRAIN_REPS if r!=held)
        ratio,screen=select_ratio(runs,allowed)
        scale=features.threshold_scale(runs,allowed)
        x,meta=features.extract(runs,subject,'train',scale*ratio)
        rep=meta.native_repetition.to_numpy();y=meta.gesture.to_numpy(int)-1
        ti=np.isin(rep,allowed);vi=rep==held
        refs=reference_values(x[ti])
        prepared.append((x[ti],y[ti],x[vi],y[vi],refs))
        audit.append({'held_repetition':held,'fitting_repetitions':allowed,'threshold_ratio':ratio,
                      'nested_threshold_scores':screen,'threshold_scale':scale.tolist(),
                      'reference_values':refs.tolist(),'train_windows':int(ti.sum()),'validation_windows':int(vi.sum())})
        print(f'S{subject:02} {device}: prepared held repetition {held}',flush=True)
    scale=features.threshold_scale(runs)
    assert np.allclose(scale,old['threshold_scale_final'],rtol=1e-10,atol=1e-15)
    xtrain,train_meta=features.extract(runs,subject,'train',scale*old['threshold_ratio'])
    ytrain=train_meta.gesture.to_numpy(int)-1
    xtest,test_meta=features.extract(runs,subject,'test',scale*old['threshold_ratio'])
    ytest=test_meta.gesture.to_numpy(int)-1
    assert len(xtest)==old['test_windows_10ms']
    refs=reference_values(xtrain)
    write(folder/'fold_audit.json',{'raw_source':raw_source,'folds':audit,
          'final_threshold_ratio':old['threshold_ratio'],'final_fitting_repetitions':features.TRAIN_REPS,
          'test_repetitions':features.TEST_REPS,'test_used_for_selection':False})
    baseline_model=features.fit_lda(xtrain,ytrain+1)
    _,lda_p=features.predict(baseline_model,xtest)
    lda_metric,lda_cm=metrics(ytest,lda_p)
    assert abs(lda_metric['accuracy']-old['lda_test_accuracy'])<1e-12,'LDA input parity failed'
    rows=[]
    for seed in seeds:
        seed_dir=folder/f'seed{seed}';seed_dir.mkdir(exist_ok=True)
        cv=[];curves=[]
        for held,(x,y,v,vy,r) in zip(features.TRAIN_REPS,prepared):
            model,_,history=fit(x,y,r,seed,max_epochs,device,val=(v,vy))
            curves.append([h['validation_accuracy'] for h in history])
            cv.extend({'held_repetition':held,**h} for h in history)
            del model
        mean_curve=np.mean(curves,axis=0)
        selected=int(np.argmax(mean_curve))+1  # tie: earliest epoch
        pd.DataFrame(cv).to_csv(seed_dir/'validation_history.csv',index=False)
        model,initial,history=fit(xtrain,ytrain,refs,seed,selected,device)
        pd.DataFrame(history).to_csv(seed_dir/'training_history.csv',index=False)
        test_data=tensors(xtest,refs,device)
        p=probabilities(model,test_data)
        result,cm=metrics(ytest,p)
        train_accuracy=float(np.mean(probabilities(model,tensors(xtrain,refs,device)).argmax(1)==ytrain))
        changes=export_rules(seed_dir,model,initial,refs)
        assert all(v>0 for v in changes.values()),'A requested parameter group did not update'
        predictions=test_meta.copy();predictions['seed']=seed
        predictions['BRB_pred']=p.argmax(1)+1;predictions['LDA_pred']=lda_p.argmax(1)+1
        predictions.to_csv(seed_dir/'predictions.csv.gz',index=False)
        np.savez_compressed(seed_dir/'probabilities.npz',brb=p,lda=lda_p.astype(np.float32))
        pd.DataFrame(cm,index=np.arange(1,18),columns=np.arange(1,18)).to_csv(seed_dir/'confusion.csv',index_label='true_gesture')
        pd.DataFrame({'gesture':np.arange(1,18),'windows':cm.sum(1),'correct':np.diag(cm),
                      'wrong':cm.sum(1)-np.diag(cm),'recall':np.diag(cm)/cm.sum(1)}).to_csv(seed_dir/'gesture_metrics.csv',index=False)
        result.update(subject=subject,seed=seed,selected_epoch=selected,validation_accuracy=float(mean_curve[selected-1]),
                      train_accuracy=train_accuracy,lda_accuracy=lda_metric['accuracy'],device=device,
                      gpu_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
                      parameter_changes=changes,success=True)
        write(seed_dir/'completion.json',result)
        rows.append({k:v for k,v in result.items() if k!='parameter_changes'})
        print(f'S{subject:02} seed{seed} {device}: epoch={selected}; BRB={result["accuracy"]:.5f}; LDA={lda_metric["accuracy"]:.5f}',flush=True)
        del model,test_data
    return rows

def self_test():
    import torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(42)
    x=rng.normal(size=(85,48)).astype(np.float32);y=np.arange(85)%17
    x+=y[:,None]*.3
    refs=reference_values(x);enc=encode(x,refs)
    assert enc[0].shape==(85,12,16)
    prior=np.ones(17)/17;model=make_model(prior).double()
    with torch.no_grad():
        model.belief_logits.add_(torch.tensor(rng.normal(0,.3,(12,81,17))))
        model.rule_logits.add_(torch.tensor(rng.normal(0,.1,(12,81))))
        model.attribute_logits.add_(torch.tensor(rng.normal(0,.1,(12,4))))
    td=(torch.tensor(enc[0][:3]),torch.tensor(enc[1][:3],dtype=torch.float64),torch.tensor(enc[2][:3]))
    p=model(*td);beta,rw,aw,fw=[v.detach().numpy() for v in model.weights()]
    score=(enc[1][:3]*aw[None,:,None,:]).sum(-1)+np.log(rw[np.arange(12)[None,:,None],enc[0][:3]])
    score=np.where(enc[2][:3],score,-np.inf);a=np.exp(score-score.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
    channels=er_numpy(a,beta[np.arange(12)[None,:,None],enc[0][:3]],2)
    expected=er_numpy(np.broadcast_to(fw,(3,12)),channels,1)
    assert np.allclose(p.detach().numpy(),expected,atol=1e-10)
    loss=-p[:,0].log().mean();loss.backward()
    for name in ['belief_logits','rule_logits','attribute_logits','fusion_logits']:
        parameter=getattr(model,name);grad=parameter.grad.flatten()
        ix=int(grad.abs().argmax());analytical=float(grad[ix]);eps=1e-5
        with torch.no_grad():parameter.view(-1)[ix]+=eps
        plus=float(-model(*td)[:,0].log().mean().detach())
        with torch.no_grad():parameter.view(-1)[ix]-=2*eps
        minus=float(-model(*td)[:,0].log().mean().detach())
        with torch.no_grad():parameter.view(-1)[ix]+=eps
        assert np.isclose(analytical,(plus-minus)/(2*eps),rtol=2e-3,atol=1e-7),name
    learned,initial,history=fit(x,y,refs,42,8,'cpu',val=(x,y))
    assert history[-1]['training_objective']<history[0]['training_objective']
    for name in ['belief_logits','rule_logits','attribute_logits','fusion_logits']:
        assert float((learned.state_dict()[name]-initial[name]).abs().max())>0,name
    print('SELF TEST PASSED: ER equivalence, all four gradient groups, learning loss, weights updated',flush=True)

def summarize(out):
    rows=[];gestures=[];devices=set()
    for subject in range(1,21):
        for seed in SEEDS:
            folder=out/f'S{subject:02}'/f'seed{seed}'
            r=json.loads((folder/'completion.json').read_text())
            assert r['success'] and all(v>0 for v in r['parameter_changes'].values())
            devices.add(r['device']);rows.append({k:v for k,v in r.items() if k!='parameter_changes'})
            gestures.append(pd.read_csv(folder/'gesture_metrics.csv').assign(subject=subject,seed=seed))
    df=pd.DataFrame(rows);df.to_csv(out/'subject_seed_metrics.csv',index=False)
    pd.concat(gestures).to_csv(out/'subject_gesture_metrics.csv',index=False)
    means=df.groupby('subject')[['accuracy','lda_accuracy']].mean()
    delta=(means.accuracy-means.lda_accuracy).to_numpy()*100
    rng=np.random.default_rng(42);boot=delta[rng.integers(0,20,(20000,20))].mean(1)
    signs=rng.choice([-1,1],(100000,20));null=np.abs((signs*delta).mean(1))
    pv=float((1+(null>=abs(delta.mean())-1e-12).sum())/(len(null)+1))
    summary={'BRB_mean_subject_accuracy_pct':float(df.accuracy.mean()*100),
        'LDA_mean_subject_accuracy_pct':float(df.lda_accuracy.mean()*100),
        'BRB_pooled_accuracy_pct':float(np.average(df.accuracy,weights=df.windows)*100),
        'LDA_pooled_accuracy_pct':float(np.average(df.lda_accuracy,weights=df.windows)*100),
        'BRB_mean_macro_f1_pct':float(df.macro_f1.mean()*100),
        'mean_gain_pp':float(delta.mean()),'gain_subject_bootstrap95_pp':np.quantile(boot,[.025,.975]).tolist(),
        'subject_signflip_p':pv,'subjects_improved':int((delta>0).sum()),'subjects_harmed':int((delta<0).sum())}
    write(out/'summary.json',summary)
    completion={'success':True,'experiment_id':'DB7-031','subjects':list(range(1,21)),'seeds':list(SEEDS),
        'final_brb_fits':60,'validation_brb_fits':240,'test_window_seed_evaluations':int(df.windows.sum()),
        'test_used_for_selection':False,'rules_per_channel':81,'channel_modules':12,'features':48,
        'references_per_feature':3,'all_parameter_groups_updated':True,'devices_used':sorted(devices)}
    assert completion['test_window_seed_evaluations']==695163 and devices=={'cuda:0','cuda:1'}
    write(out/'completion.json',completion)
    report=['# DB7-031 hierarchical BRB results','',
        'Direct EMG-only gesture classifier: 48 features, three references each,12 channel BRBs of81 rules, learned ER fusion weights.',
        'Consequent beliefs, rule weights, attribute weights and fusion weights were jointly trained. Reference values were fitted on training data, then fixed.',
        'Train1/3/4/6, test2/5;200ms;50ms training/10ms test;20 subjects;three seeds;fourfold training repetition validation.',
        'Outer-validation preprocessing uses nested training-only threshold screening; test was not used for selection.',
        'Previously inspected DB7 recordings: exploratory, not independent confirmation. Compare primarily with matched EMG-only LDA; SI also uses inertial inputs.','',
        '```json',json.dumps(summary,indent=2),'```','',
        'Initial/trained rules and weights, references, validation histories, predictions,confusions and per-gesture metrics are stored per subject/seed.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--input',type=Path);parser.add_argument('--out',type=Path,default=Path('/kaggle/working/db7_031_results'))
    parser.add_argument('--parent',type=Path,default=Path('/kaggle/working/db7_031_parent.json'))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--subjects',type=int,nargs='+')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test:self_test();return
    args.out.mkdir(parents=True,exist_ok=True)
    parent=json.loads(args.parent.read_text());raw=args.input or features.find_root()
    if args.worker:
        torch.cuda.set_device(args.device)
        write(args.out/f'worker_{args.device[-1]}_runtime.json',{'device':args.device,'gpu_name':torch.cuda.get_device_name(args.device),'pid':os.getpid(),'subjects':args.subjects,'torch_version':torch.__version__})
        for subject in args.subjects:run_subject(raw,subject,args.out,parent,args.device)
        return
    assert torch.cuda.device_count()>=2,'Two GPUs required; verify T4 x2 allocation before fitting'
    self_test()
    write(args.out/'PROTOCOL.json',{'experiment_id':'DB7-031','max_epochs':MAX_EPOCHS,'batch_size':BATCH,'optimizer':'Adam','learning_rate':LR,'regularization':PENALTY,
        'train_repetitions':features.TRAIN_REPS,'test_repetitions':features.TEST_REPS,'window_ms':200,'training_stride_ms':50,'test_stride_ms':10,
        'reference_values':'Training feature 5th/50th/95th percentiles, strictly ordered with tiny expansion for ties; fixed after fitting.',
        'architecture':'12 channel BRBs x81 rules;4 antecedents per module;17 consequent classes;trainable-weight ER fusion.',
        'learned':['consequent beliefs','rule weights','attribute weights','fusion weights'],
        'attribute_weight_constraint':'exp(logweight), normalized by channel maximum; raw logweights projected to[-2,2].',
        'rule_weight_constraint':'exp(logweight),logweight in[-2,2]; activations normalized across matching rules.',
        'epoch_selection':'Earliest epoch maximizing mean fourfold training-repetition validation accuracy, up to60; final refit on all4 training reps.',
        'fold_threshold_selection':'Nested LDA ratio screening on the three outer-fitting repetitions only; final threshold is the verified DB7-030 train-only choice.',
        'test_used_for_selection':False,'seeds':SEEDS,'subjects':list(range(1,21)),'neural_fits':0})
    workers=[]
    for device in range(2):
        subjects=list(range(device+1,21,2))
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--device',f'cuda:{device}',
             '--input',str(raw),'--out',str(args.out),'--parent',str(args.parent),'--subjects',*map(str,subjects)]
        workers.append(subprocess.Popen(cmd))
    codes=[p.wait() for p in workers]
    assert codes==[0,0],f'Worker failures {codes}'
    summarize(args.out)
    archive=args.out.parent/'db7_031_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(args.out.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(args.out))
    assert zipfile.ZipFile(archive).testzip() is None
    print('DB7-031 COMPLETE',archive,flush=True)

if __name__=='__main__':main()
