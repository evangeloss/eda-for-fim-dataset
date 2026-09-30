"""Validate exported targets, pairing and audit joins after the quick run."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'smoke_results')
    rows=list(csv.DictReader((root/'per_scene.csv').open()))
    assert len(rows)==24
    for r in rows:
        if float(r['alpha'])==0:
            assert abs(float(r['rho_mean'])-1)<1e-12
            assert float(r['residual_nmse'])<1e-24
            assert abs(float(r['view_effective_rank'])-1)<1e-12
    target=None
    for file in sorted((root/'L3_snr10').glob('inputs_alpha_*.npz')):
        z=np.load(file)
        assert z['X'].shape==(8,32,25,25)
        assert z['H_target'].shape==(8,4,25,25)
        raw=z['H_target']*z['scale'][:,None,None,None]
        if target is None: target=raw
        else: np.testing.assert_allclose(raw,target,rtol=2e-6,atol=1e-8)
        errors=((z['X'][:,:4]-z['H_target'])**2).sum((1,2,3))/(z['H_target']**2).sum((1,2,3))
        matching=[r for r in rows if float(r['alpha'])==float(z['alpha'])]
        np.testing.assert_allclose(errors,[float(r['ridge_first_view_nmse']) for r in matching],rtol=2e-6)
    assert target is not None, 'Run quick with --export-inputs first'
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        keys=['seed','pair_start','path_count','snr_db','alpha','sample']
        with (d/'pred.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys+['model_nmse']);w.writeheader()
            w.writerows({**{k:r[k] for k in keys},'model_nmse':r['ridge_first_view_nmse']} for r in rows)
        subprocess.run([sys.executable,'audit_model.py','--scenes',str(root/'per_scene.csv'),'--predictions',str(d/'pred.csv'),'--output',str(d/'associations.csv')],check=True)
        assert len(list(csv.DictReader((d/'associations.csv').open())))==24
        (d/'manifest.csv').write_text('scene_id,split,alpha,snr_db,path_count,geometry_id,normalization_mode\na,train,0.1,10,3,g,observation_only\na,test,0.2,10,3,g,observation_only\nb,train,0.1,10,3,g,observation_only\n')
        subprocess.run([sys.executable,'audit_coverage.py',str(d/'manifest.csv'),'--output',str(d/'coverage.json')],check=True)
        assert json.loads((d/'coverage.json').read_text())['cross_split_scene_ids']==['a']
    print('Verified: zero-deformation identities, paired targets, exported residual errors, model join and leakage detection.')


if __name__=='__main__': main()
