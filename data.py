"""Independent scenes, paired across regimes; physics matches the EDA convention."""
from types import SimpleNamespace
import numpy as np
import physics as ph
from metrics import describe


def codebook(seed=2026):
    a=SimpleNamespace(fc=28e9,spacing=.125,NH_B=5,NV_B=5,NH_U=5,NV_U=5,M=8)
    rng=np.random.default_rng(seed)
    pb,pu,zb,zu=ph.make_codebook(a,rng)
    pilots=ph.generate_multi_pilots(25,1,rng)
    return pb,pu,zb,zu,pilots


def pack(h):
    return np.stack((h[:,:,0].real,h[:,:,0].imag,h[:,:,1].real,h[:,:,1].imag))


def make_dataset(n, split, seed, book, alphas, snrs, paths=3, pair=0, diagnostics=False):
    if split not in ('train','validation','test'): raise ValueError('Unknown split')
    if not 0<=pair<31 or n<1: raise ValueError('Invalid scene count or pair')
    split_id={'train':1,'validation':2,'test':3}[split]
    schedule=np.random.default_rng(np.random.SeedSequence([seed,split_id,987]))
    combinations=[(a,s) for a in alphas for s in snrs]
    order=np.arange(n)%len(combinations); schedule.shuffle(order)
    pb,pu,zb,zu,S=book
    X=[]; H=[]; G=[]; meta=[]
    for i in range(n):
        alpha,snr=combinations[order[i]]
        # Per-scene streams do not depend on deformation/SNR: exact pairing.
        prng=np.random.default_rng(np.random.SeedSequence([seed,split_id,i,0]))
        nrng=np.random.default_rng(np.random.SeedSequence([seed,split_id,i,1]))
        p=ph.generate_path_parameters(paths,28e9,1e5,prng)
        h0=ph.build_H_fim_from_paths(p,pb,np.zeros_like(pb),pu,np.zeros_like(pu),3e8/28e9,1e5,32)[:,:,pair:pair+2]
        hs=[]; estimates=[]; variances=[]
        for m in range(8):
            full=ph.build_H_fim_from_paths(p,pb,alpha*zb[m],pu,alpha*zu[m],3e8/28e9,1e5,32)
            y,var,_=ph.pilot_transmission_multi(full,S,snr,nrng)
            r=np.einsum('bvk,vu->buk',y[:,:,pair:pair+2,0],S[:,:,0].conj().T)
            hs.append(full[:,:,pair:pair+2]); estimates.append(r); variances.append(var)
        ridge=np.array([r/(1+v) for r,v in zip(estimates,variances)])
        scale=max(float(np.abs(ridge).max()),1e-8)
        X.append(np.concatenate([pack(r) for r in ridge])/scale)
        H.append(pack(h0)/scale)
        G.append(np.concatenate([alpha*zb.ravel(),alpha*zu.ravel()])/(3e8/28e9))
        row=dict(scene_id=f'{split}_{seed}_{i}',split=split,sample=i,alpha=float(alpha),snr_db=float(snr),path_count=paths,pair_start=pair)
        if diagnostics:
            row.update(describe(hs,h0,estimates,variances))
            row['rho_mean']=float(np.mean([abs(np.vdot(h,h0))/np.sqrt(ph.energy(h)*ph.energy(h0)) for h in hs]))
            row['clean_difference']=float(np.mean([ph.nmse(h,h0) for h in hs]))
        meta.append(row)
    return dict(X=np.asarray(X,dtype=np.float32),H=np.asarray(H,dtype=np.float32),
                G=np.asarray(G,dtype=np.float32),alpha=np.array([r['alpha'] for r in meta],dtype=np.float32),meta=meta)
