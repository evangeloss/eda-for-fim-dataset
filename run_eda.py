"""Run paired deformation experiments and produce a compact paste-back report."""
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import physics


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--export-inputs', action='store_true', help='Save normalized CNN inputs and targets; may use substantial disk space')
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--diagnostic-samples', type=int, default=10)
    p.add_argument('--alphas', nargs='+', type=float, default=[0,.01,.02,.05,.1,.2,.3,.5])
    p.add_argument('--snrs', nargs='+', type=float, default=[0,10,20])
    p.add_argument('--paths', nargs='+', type=int, default=[1,3,4])
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--pair-start', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('results'))
    args = p.parse_args()
    if args.quick:
        args.samples, args.diagnostic_samples = 8, 1
        args.alphas, args.snrs, args.paths = [0,.2,.5], [10], [3]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, scenes = [], []
    for L in args.paths:
        for snr in args.snrs:
            folder = args.output / f'L{L}_snr{snr:g}'
            options = ['--samples',str(args.samples),'--diagnostic-samples',str(args.diagnostic_samples),
                       '--L',str(L),'--snr-db',str(snr),'--seed',str(args.seed),
                       '--pair-start',str(args.pair_start),'--output',str(folder),
                       '--alphas', *map(str,args.alphas)]
            experiment = physics.parse_args(options)
            experiment.export_inputs = args.export_inputs
            physics.run(experiment)
            report = json.loads((folder/'report.json').read_text())
            for summary, detail in zip(report['summary'], report['details']):
                meta = dict(path_count=L,snr_db=snr,seed=args.seed,pair_start=args.pair_start,alpha=detail['alpha'])
                summaries.append({**meta, **summary})
                for i, row in enumerate(detail['samples']):
                    neighbors = {f'{kind}_nn_{k}':v[i] for kind in ('clean','noisy')
                                 for k,v in detail[f'nearest_neighbors_{kind}'].items()}
                    scenes.append({**meta, **row, **neighbors})
    write_csv(args.output/'summary.csv', summaries)
    write_csv(args.output/'per_scene.csv', scenes)
    # Independent scenes are resampled; paired amplitude differences reuse scene IDs.
    rng = np.random.default_rng(args.seed + 9000)
    intervals = []
    for s in summaries:
        group = [r for r in scenes if all(r[k]==s[k] for k in ('path_count','snr_db','alpha'))]
        for metric in ['rho_mean','residual_nmse','ridge_first_view_nmse','view_effective_rank']:
            x = np.array([r[metric] for r in group])
            medians = np.median(x[rng.integers(len(x),size=(1000,len(x)))],axis=1)
            intervals.append({**{k:s[k] for k in ('path_count','snr_db','alpha')},'metric':metric,
                              'median':float(np.median(x)),'ci_low':float(np.quantile(medians,.025)),
                              'ci_high':float(np.quantile(medians,.975))})
    write_csv(args.output/'bootstrap_intervals.csv', intervals)
    config = {**vars(args),'output':str(args.output),'numpy':np.__version__,'python':platform.python_version()}
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2))
    lines = ['# Deformation EDA results', json.dumps(config),
             'Quick runs are smoke tests, not evidence. Values below are medians; errors are linear NMSE.',
             'L | SNR | b/lambda | similarity | clean difference | ridge residual | view rank | local failures/diagnostics',
             '---|---|---|---|---|---|---|---']
    for s in summaries:
        lines.append(f"{s['path_count']} | {s['snr_db']:g} | {s['alpha']:g} | {s['rho_mean_median']:.4f} | "
                     f"{s['residual_nmse_median']:.4g} | {s['ridge_first_view_nmse_median']:.4g} | "
                     f"{s['view_effective_rank_median']:.3f} | {s['local_target_failure_count']}/{s['diagnostic_samples']}")
    lines += ['', 'No CNN predictions or actual training-set coverage were measured by this simulation.',
              'Attach summary.csv, per_scene.csv and bootstrap_intervals.csv for deeper analysis.',
              'Oracle knows angles/delays; local diagnostics do not prove global identifiability.',
              'Geometry is a fresh fixed random codebook, not automatically the checkpoint training codebook.']
    (args.output/'PASTE_BACK.md').write_text('\n'.join(lines),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,2,figsize=(11,8))
        for ax, metric in zip(axes.flat,['rho_mean_median','residual_nmse_median','view_effective_rank_median','ridge_first_view_nmse_median']):
            for L in args.paths:
                for snr in args.snrs:
                    rows = sorted([s for s in summaries if s['path_count']==L and s['snr_db']==snr],key=lambda s:s['alpha'])
                    ax.plot([s['alpha'] for s in rows],[s[metric] for s in rows],'.-',label=f'L={L}, SNR={snr:g}')
            ax.set(xlabel='b / wavelength',ylabel=metric.replace('_median',''))
            ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output/'overview.png',dpi=160)
        plt.close(fig)
    except ImportError:
        print('matplotlib unavailable; all numerical outputs were saved.')
    print(f'Paste back: {(args.output/"PASTE_BACK.md").resolve()}')


if __name__ == '__main__':
    main()
