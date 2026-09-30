"""Audit a real scene manifest, not an assumed training distribution."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--edges',nargs='+',type=float,default=[0,.02,.05,.1,.2,.3,.5])
    p.add_argument('--output',type=Path,default=Path('coverage_report.json'))
    a=p.parse_args()
    edges=np.asarray(a.edges)
    if len(edges)<2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0):
        raise ValueError('Bin edges must be finite and increasing')
    rows=list(csv.DictReader(a.manifest.open()))
    required={'scene_id','split','alpha','snr_db','path_count','geometry_id','normalization_mode'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Required columns: {sorted(required)}')
    splits=defaultdict(set); groups=defaultdict(set); values=defaultdict(list)
    for r in rows:
        alpha=float(r['alpha'])
        if not np.isfinite(alpha) or alpha<0:
            raise ValueError('Invalid alpha')
        splits[r['scene_id']].add(r['split'])
        if alpha<edges[0]: label=f'below {edges[0]:g}'
        elif alpha>edges[-1]: label=f'above {edges[-1]:g}'
        else:
            i=min(int(np.searchsorted(edges,alpha,side='right'))-1,len(edges)-2)
            label=f'[{edges[i]:g}, {edges[i+1]:g}' + (']' if i==len(edges)-2 else ')')
        group=(r['split'],label,r['snr_db'],r['path_count'],r['geometry_id'],r['normalization_mode'])
        groups[group].add(r['scene_id'])
        for metric in ['rho_mean','residual_nmse','input_scale','normalized_residual_rms','target_energy']:
            if r.get(metric):
                v=float(r[metric])
                if not np.isfinite(v): raise ValueError(f'Nonfinite {metric}')
                values[(group,metric)].append(v)
    report={'cross_split_scene_ids':sorted(k for k,v in splits.items() if len(v)>1),
            'groups':[dict(zip(['split','alpha_bin','snr_db','path_count','geometry_id','normalization_mode'],g),unique_scenes=len(ids)) for g,ids in groups.items()],
            'row_weighted_metric_quantiles':[{'group':g,'metric':m,'p10_p50_p90':np.quantile(v,[.1,.5,.9]).tolist()} for (g,m),v in values.items()],
            'note':'Scene IDs must identify the same latent scene globally. Counts deduplicate scene IDs; optional quantiles weight rows. Missing metric fields are not inferred.'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Saved {a.output}; cross-split scene IDs: {len(report["cross_split_scene_ids"])}')


if __name__=='__main__': main()
