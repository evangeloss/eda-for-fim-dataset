"""Join independently supplied per-scene model errors to EDA measurements."""
import argparse
import csv
from pathlib import Path
import numpy as np

KEYS = ['seed','pair_start','path_count','snr_db','alpha','sample']
METRICS = ['rho_mean','residual_nmse','view_effective_rank','phase_weighted_rms_rad',
           'magnitude_difference_nmse','input_scale','normalized_residual_rms','target_energy']


def key(r):
    return tuple(float(r[k]) for k in KEYS)


def ranks(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts-1)/2)[inv]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('model_associations.csv'))
    a = p.parse_args()
    rows = list(csv.DictReader(a.scenes.open()))
    pred = list(csv.DictReader(a.predictions.open()))
    lookup = {key(r):float(r['model_nmse']) for r in pred}
    if len(lookup)!=len(pred):
        raise ValueError('Duplicate prediction keys')
    if any(not np.isfinite(v) or v<0 for v in lookup.values()):
        raise ValueError('model_nmse must be finite, nonnegative LINEAR error')
    if len({key(r) for r in rows})!=len(rows):
        raise ValueError('Duplicate scene keys')
    if set(lookup)!={key(r) for r in rows}:
        raise ValueError('Predictions must match every EDA scene exactly; do not join unrelated evaluations')
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in ['path_count','snr_db','alpha']),[]).append(r)
    out=[]
    for group, subset in groups.items():
        y = ranks([lookup[key(r)] for r in subset])
        for metric in METRICS:
            x=ranks([float(r[metric]) for r in subset])
            corr = float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else None
            out.append(dict(zip(['path_count','snr_db','alpha'],group),metric=metric,n=len(x),spearman=corr))
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'Saved {a.output}; associations are descriptive, not causal.')


if __name__=='__main__':
    main()
