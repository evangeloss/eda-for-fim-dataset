# Deformation EDA

Run this project before changing the network. It studies whether increasing deformation changes channel similarity, residual difficulty, view diversity and local recoverability. No GPU, checkpoint or training is required. Python 3.10+.

## Run

Extract the ZIP and open a terminal in the folder containing `run_eda.py`:

```sh
python -m pip install -r requirements.txt
python run_eda.py --quick --output smoke_results
python run_eda.py --output results
```

The quick run checks functionality with only 8 scenes and 1 local diagnostic per amplitude; do not interpret it as scientific evidence. The full default uses 200 independent scenes per path count, 10 local diagnostics, 8 amplitudes, 3 SNRs and path counts 1, 3, 4. Local Jacobian calculations can be slow on CPU. Each completed SNR/path group is saved immediately. Runtime depends on your CPU; no runtime estimate is assumed.

For a smaller first substantive run:

```sh
python run_eda.py --samples 100 --diagnostic-samples 5 --paths 3 --snrs 10 20 --output first_results
```

For Kaggle/Colab, upload and extract the ZIP, change directory to the extracted folder and execute the same commands with `!` before each command. There are no local workspace dependencies. Do not paste scripts as notebook cells; run them as files so imports work.

**Send back `results/PASTE_BACK.md` first.** For detailed interpretation, also attach `summary.csv`, `per_scene.csv`, `bootstrap_intervals.csv`, and, if available, `overview.png`. No automatic pass/fail verdict is assigned.

## Experimental controls

- Same latent paths and standardized noise draws across amplitudes and SNRs, within a path-count group. Noise variance still follows each view's full-band received power.
- 5x5 BS and UE arrays, lambda/8 spacing, 28 GHz, 100 kHz bandwidth parameter, K=32, 8 views, full unitary pilots. Additional views add measurement energy.
- The default observes ONLY tones 0 and 1 while retaining the original 32-tone phase/noise calculation. Repeat with `--pair-start 15 --output results_pair15` for frequency sensitivity. Do not pool overlapping pairs as independent scenes.
- Each scene gets one sample ID; results across amplitude are paired. Bootstrap intervals resample independent scenes within each amplitude, not subcarriers. They are pointwise, not simultaneous intervals or confidence intervals for paired differences.
- Geometry is a seeded random codebook with maximum normal displacement b per array/view. It is not necessarily the codebook used by a saved model. Repeat with `--seed 2027 --output results_seed2027` to check dependence on geometry and scene sampling.
- The inherited physics preserves the original polar-angle convention despite its `el` variable names. Phase-only far-field simulation excludes coupling, blockage and deformation-dependent path gains.

## Questions and metrics

| Question | Fields in per_scene.csv | Interpretation |
|---|---|---|
| Does the channel move away from the undeformed target? | rho_mean, residual_nmse, energy_ratio | Mean phase-invariant complex similarity, mean relative squared difference, mean energy ratio across views |
| Is the change mostly phase or magnitude? | phase_weighted_rms_rad, magnitude_difference_nmse | Wrapped phase RMS weighted by product magnitudes; magnitude-only relative squared difference |
| Do views provide different structure? | view_coherence_mean, view_effective_rank | Mean off-diagonal normalized Gram magnitude; participation rank `(sum eigenvalues)^2/sum(eigenvalues^2)` |
| Does the network's residual task grow? | ridge_first_view_nmse, normalized_residual_rms, input_scale, normalized_target_peak | Residual relative energy and observation-normalized residual size; scale and target peak diagnose normalization tails |
| Does simple averaging fail? | first_view_nmse, mean_view_nmse, ridge_mean_view_nmse | Unregularized deprojected first/mean views and ridge-regularized mean, all compared to undeformed target |
| Are nearby observations associated with different targets? | clean_nn_*, noisy_nn_* | Nearest-neighbour input and target distances compared with random target distances |
| Is inversion locally unstable? | jacobian_condition, target_null_fraction, target_locally_estimable, target_crb_nmse | Local all-path-parameter diagnostics with known geometry |
| Would knowing path angles/delays help? | gain_oracle_nmse, gain_oracle_noiseless_nmse | Gain-only inverse problem with privileged path information |

All NMSE fields are LINEAR, not dB. Similarity ignores common phase rotation; relative difference does not. Phase RMS is descriptive, not an unwrapped physical phase displacement. Zero-energy channels are rejected. Ridge metrics use the exact unitary, single-pilot simplification R=YS*/(1+sigma2); the main runner always uses one pilot. Observation normalization uses the peak across every view and both selected tones, without the target.

Nearest-neighbour diagnostics use raw received observations with symmetric energy-normalized pair distances, not CNN feature embeddings. Inspect neighbour input distances before claiming ambiguity. Sparse random neighbours cannot prove identifiability. Local CRBs are conditional on fixed noise covariance and known geometry; they are not CNN error predictions. Only `diagnostic-samples` scenes have these expensive fields; missing values are not zeros. Inspect failure/missing counts and increase diagnostics before drawing conclusions.

## Model errors: optional second stage

The default run is DATA analysis and does not run a trained CNN. To link failures to the measured properties, evaluate your model on the EXACT same generated scenes, geometry, noise, tone pair, input layout and normalization. Use `physics.run` as the integration point: `H0`, `deprojected`, `vs` and `row` inside its scene loop provide targets and observations. `R_m=deprojected[m]/(1+vs[m])`. Pack normalized R in view order as Re(k), Im(k), Re(k+1), Im(k+1), matching the training implementation. A residual model must have its reference estimate added back before scoring. Do not substitute aggregate errors from older runs or load arbitrary checkpoints.

To avoid modifying the generator, add `--export-inputs` to the run command. Each group then contains `inputs_alpha_*.npz` with `X` (N,32,25,25), `H_target` (N,4,25,25), scale and identifiers. These are observation-normalized inputs and FULL channel targets, not residual targets. Evaluate your model adapter on X; add `X[:,:4]` if the model predicts a residual, then calculate per-sample `sum((prediction-H_target)**2)/sum(H_target**2)`. Check your model's forward method: some already add that reference. Exporting the full sweep can require substantial disk space.

Export `predictions.csv` with these columns:

```csv
seed,pair_start,path_count,snr_db,alpha,sample,model_nmse
```

Use one row for every row of per_scene.csv; `model_nmse` is squared error divided by target energy over both tones. Then:

```sh
python audit_model.py --scenes results/per_scene.csv --predictions predictions.csv --output results/model_associations.csv
```

This validates exact ID coverage and computes Spearman associations WITHIN amplitude/SNR/path-count groups to avoid interpreting deformation itself as a confounding correlation. It does not verify externally supplied predictions' provenance or establish causality. Compare matched single-view and multiview models, and a model trained only at large deformation, in separate audits.

## Actual training coverage: required before diagnosing distribution shift

The synthetic sweep cannot tell us what the training set contained. Export the actual training/validation manifest with scene ID, split, b/lambda, SNR, path count, geometry ID, normalization mode and, ideally, the same per-scene metrics. Count unique scenes per combination, verify no scene crosses splits, and compare train/test quantiles within matched groups. Keep every pair from a scene in one split. Send that manifest with the results for the coverage analysis. This project deliberately does not manufacture a training distribution or claim to audit an unavailable dataset.

Use column names `scene_id,split,alpha,snr_db,path_count,geometry_id,normalization_mode` and run:

```sh
python audit_coverage.py training_manifest.csv --output results/coverage_report.json
```

The audit counts unique scenes by group, flags cross-split scene IDs, and reports optional metric quantiles. Scene IDs must be globally consistent, not restarted separately in each split. Optional metric quantiles weight rows, so export one row per scene/condition or account for pair-count imbalance.

## Decision guide

- Large matrix difference with good oracle/local diagnostics: investigate representation, residual scale and training coverage; not proof of information loss.
- Poor noise-conditioned diagnostics: investigate geometry, pilots and measurement budget as well as the estimator.
- Low view correlation with increasing rank: views may add diversity; investigate fusion before discarding them.
- High error only outside training support: run a matched coverage experiment before changing architecture.
- Failure even on a well-trained large-deformation-only model: investigate ambiguity and representation with controlled baselines.

`physics.py` is a self-contained copy of the existing channel estimability experiment plus additional descriptive metrics. Original project files were not modified.

