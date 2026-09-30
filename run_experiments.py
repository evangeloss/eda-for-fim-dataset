"""Four matched training arms; no pre-existing checkpoint required."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from data import codebook,make_dataset
from models import ControlledSFCNN,nmse

ARMS={'large_only':'none','mixed':'none','mixed_ratio':'ratio','mixed_geometry':'geometry'}


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def loader(data,batch,shuffle=False,seed=0):
    tensors=[torch.from_numpy(data[k]) for k in ('X','H','G','alpha')]
    return DataLoader(TensorDataset(*tensors),batch_size=batch,shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)


def evaluate(model,data,batch,device):
    model.eval(); values=[]
    with torch.no_grad():
        for x,h,g,a in loader(data,batch):
            x,h,g,a=(t.to(device) for t in (x,h,g,a))
            values.extend(nmse(model(x,g,a),h).cpu().tolist())
    return np.asarray(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--train-scenes',type=int,default=4096)
    p.add_argument('--validation-scenes',type=int,default=200)
    p.add_argument('--test-scenes',type=int,default=200)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    p.add_argument('--geometry-seed',type=int,default=2026)
    p.add_argument('--path-count',type=int,default=3)
    p.add_argument('--pair-start',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    a=p.parse_args()
    if a.quick:
        a.epochs=1;a.train_scenes=12;a.validation_scenes=6;a.test_scenes=6;a.seeds=[11]
    if min(a.epochs,a.train_scenes,a.validation_scenes,a.test_scenes,a.batch_size,a.path_count)<1 or a.batch_size<2:
        p.error('Positive counts and batch-size >=2 required')
    if a.output.exists() and any(a.output.iterdir()): p.error('Use a fresh output directory to preserve earlier runs')
    a.output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    ratios=[0,.01,.02,.05,.1,.2,.3,.5] if not a.quick else [0,.5]
    train_ratios=[.01,.02,.05,.1,.2,.3,.5]
    snrs=[0,10,20] if not a.quick else [20]
    config={**vars(a),'output':str(a.output),'device':str(device),'torch':torch.__version__,
            'train_ratios':train_ratios,'test_ratios':ratios,'snrs':snrs,
            'selection':'minimum validation mean linear NMSE at alpha=0.5, balanced SNR',
            'quick_is_not_evidence':a.quick}
    (a.output/'configuration.json').write_text(json.dumps(config,indent=2))
    book=codebook(a.geometry_seed)
    np.savez_compressed(a.output/'codebook.npz',**dict(zip(['PB','PU','ZB','ZU','pilots'],book)))
    errors=[];summaries=[];history=[];counts={}
    for seed in a.seeds:
        folder=a.output/f'seed_{seed}';folder.mkdir()
        def generate(split,n,alphas,snrs,diag=False):
            return make_dataset(n,split,seed,book,alphas,snrs,a.path_count,a.pair_start,diag)
        print(f'Seed {seed}: generating training and validation scenes',flush=True)
        mixed=generate('train',a.train_scenes,train_ratios,snrs)
        large=generate('train',a.train_scenes,[.5],snrs)
        validation=generate('validation',a.validation_scenes,[.5],snrs)
        # Shared latent scenes and initialization; disjoint train/validation/test IDs.
        write_csv(folder/'mixed_training_manifest.csv',mixed['meta'])
        write_csv(folder/'large_training_manifest.csv',large['meta'])
        write_csv(folder/'validation_manifest.csv',validation['meta'])
        models={}
        for arm,mode in ARMS.items():
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            model=ControlledSFCNN(mode,width=8 if a.quick else 64).to(device)
            counts[arm]=sum(t.numel() for t in model.parameters())
            optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
            train=large if arm=='large_only' else mixed
            batches=loader(train,a.batch_size,True,seed)
            best=float('inf');best_state=None
            for epoch in range(a.epochs):
                model.train();loss_sum=0.;n=0
                for x,h,g,alpha in batches:
                    x,h,g,alpha=(t.to(device) for t in (x,h,g,alpha))
                    optimizer.zero_grad(set_to_none=True)
                    loss=nmse(model(x,g,alpha),h).mean()
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();loss_sum+=loss.item()*len(x);n+=len(x)
                val=float(evaluate(model,validation,a.batch_size,device).mean())
                history.append(dict(seed=seed,arm=arm,epoch=epoch+1,train_nmse=loss_sum/n,validation_nmse=val))
                print(f'{seed} {arm} epoch {epoch+1}/{a.epochs}: train={loss_sum/n:.5g} val={val:.5g}',flush=True)
                if val<best:
                    best=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    torch.save({'state_dict':best_state,'mode':mode,'width':8 if a.quick else 64,
                                'epoch':epoch+1,'validation_nmse':val,'configuration':config},folder/f'{arm}_best.pt')
                write_csv(a.output/'training_history.csv',history)
            model.load_state_dict(best_state);models[arm]=model.cpu()
        del mixed,large,validation,optimizer,train,batches
        for alpha in ratios:
            for snr in snrs:
                test=generate('test',a.test_scenes,[alpha],[snr],True)
                predictions={}
                for arm,model in models.items():
                    model.to(device)
                    predictions[arm]=evaluate(model,test,a.batch_size,device)
                    model.cpu()
                x,h=test['X'],test['H']
                predictions['first_view']=np.sum((x[:,:4]-h)**2,axis=(1,2,3))/np.sum(h*h,axis=(1,2,3))
                predictions['mean_view']=np.sum((x.reshape(-1,8,4,25,25).mean(1)-h)**2,axis=(1,2,3))/np.sum(h*h,axis=(1,2,3))
                for arm,values in predictions.items():
                    summaries.append(dict(seed=seed,arm=arm,alpha=alpha,snr_db=snr,n=len(values),
                                          mean_nmse=float(values.mean()),nmse_db=float(10*np.log10(max(values.mean(),1e-30))),
                                          median_nmse=float(np.median(values)),p90_nmse=float(np.quantile(values,.9))))
                    errors.extend(dict(seed=seed,arm=arm,model_nmse=float(v),**r) for v,r in zip(values,test['meta']))
                print(f'Test seed={seed} alpha={alpha:g} SNR={snr:g} complete',flush=True)
                write_csv(a.output/'per_scene_errors.csv',errors)
                write_csv(a.output/'summary.csv',summaries)
        del models
    (a.output/'parameter_counts.json').write_text(json.dumps(counts,indent=2))
    # Paired CIs use each seed's test scenes; never treat amplitude/SNR repeats as independent.
    comparisons=[];rng=np.random.default_rng(4321)
    for seed in a.seeds:
        for snr in snrs:
            subset=[r for r in errors if r['seed']==seed and r['alpha']==.5 and r['snr_db']==snr]
            base={r['scene_id']:r['model_nmse'] for r in subset if r['arm']=='mixed'}
            for arm in ['large_only','mixed_ratio','mixed_geometry']:
                difference=np.array([r['model_nmse']-base[r['scene_id']] for r in subset if r['arm']==arm])
                boot=difference[rng.integers(len(difference),size=(1000,len(difference)))].mean(1)
                comparisons.append(dict(seed=seed,snr_db=snr,alpha=.5,arm=arm,reference='mixed',
                                        mean_nmse_difference=float(difference.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    write_csv(a.output/'paired_comparisons.csv',comparisons)
    lines=['# Controlled deformation experiments',f'Quick smoke run: {a.quick}; seeds: {a.seeds}; epochs: {a.epochs}',
           f'Train scenes per arm/seed: {a.train_scenes}; path count: {a.path_count}; geometry seed: {a.geometry_seed}',
           'Primary endpoint: alpha=0.5. Entries are dB of mean linear NMSE per seed, then averaged across seeds.',
           '| Arm | SNR | Mean dB across seeds | Min / max seed dB |','|---|---|---|---|']
    for arm in [*ARMS,'first_view','mean_view']:
        for snr in snrs:
            v=[r['nmse_db'] for r in summaries if r['arm']==arm and r['alpha']==.5 and r['snr_db']==snr]
            lines.append(f'| {arm} | {snr} | {np.mean(v):.3f} | {min(v):.3f} / {max(v):.3f} |')
    lines+=['','Lower is better. See summary.csv for all amplitudes and paired_comparisons.csv for per-seed paired confidence intervals.',
            'Fixed codebook: geometry and amplitude contain equivalent varying information. Geometry results do not establish generalization to unseen shapes.',
            'Equal total optimization steps; mixed arms have fewer high-deformation exposures. Difference is allocation + multi-regime learning, not a pure causal test of either.',
            'Check training_history.csv for convergence before drawing conclusions. These are new controlled models, not evaluations of previous checkpoints.']
    (a.output/'PASTE_BACK.md').write_text('\n'.join(lines),encoding='utf-8')
    (a.output/'run_status.json').write_text(json.dumps({'completed':True}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__': main()
