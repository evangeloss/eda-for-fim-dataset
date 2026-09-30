# Controlled large-deformation experiments

This package trains new models to distinguish training-distribution difficulty from the usefulness of known deformation information. It requires no previous checkpoint or EDA result ZIP. It does not modify the original project.

## Kaggle: easiest route

1. Extract the ZIP and upload the extracted files to a public GitHub repository, at the root or in one folder.
2. Create a Kaggle Python notebook. Enable Internet and a GPU.
3. Paste `kaggle_launcher.py` into one cell, or import the supplied `Kaggle_Experiments.ipynb` notebook.
4. Set REPO_URL to your GitHub repository URL. Leave BRANCH empty for its default branch. Run.
5. Download `experiment_results.zip` from the printed output folder. Send it back with `PASTE_BACK.md`.

The launcher starts a fresh checkout and results folder every time. Completed models, incremental CSVs and logs are retained in a partial-results ZIP when a Python error interrupts training. A killed Kaggle session may not execute that cleanup, so use Kaggle's saved output files. There is no automatic training resume. Full runs train 12 models (four arms x three seeds); runtime depends on the GPU. CPU is suitable for the smoke test, not recommended for the full run.

Set QUICK=True for a wiring check, then False for the experiment. QUICK uses a smaller network, one epoch, tiny datasets and one seed; its scores have no scientific interpretation.

## Local commands

```sh
python -m pip install -r requirements.txt
python run_experiments.py --quick --output smoke_results
python run_experiments.py --output results
```

Always use a fresh output directory. Additional independent experiments:

```sh
python run_experiments.py --path-count 1 --output results_L1
python run_experiments.py --path-count 4 --output results_L4
python run_experiments.py --geometry-seed 2027 --output results_geometry2027
python run_experiments.py --pair-start 15 --output results_pair15
```

Each command retrains all arms. Changing the geometry seed is replication on another fixed codebook, NOT an unseen-geometry test of an already trained model.

## Four controlled arms

| Arm | Training amplitudes b/lambda | Extra inference information |
|---|---|---|
| large_only | 0.5 only | Constant metadata |
| mixed | Balanced 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5 | Constant metadata |
| mixed_ratio | Same data as mixed | Actual amplitude |
| mixed_geometry | Same data as mixed | Actual BS/UE displacement coordinates divided by wavelength |

The fourth arm (amplitude-only) is essential: for a fixed codebook, geometry is a deterministic function of amplitude. A geometry win alone therefore does not prove the model learned spatial physics or can generalize to new shapes. The ratio control helps distinguish benefits of available metadata from its encoding.

All arms allocate the same five-block, width-64 residual CNN and FiLM encoder parameters, start with identical weights for a given seed, and use identical optimization settings. Unconditioned arms receive zero metadata but still have trainable constant FiLM modulation. This is a controlled SFCNN variant, not an exact rerun of your old model/checkpoint. Equal parameter allocation does not imply identical effective capacity: input-dependent conditioning is the treatment under test.

## Data and fairness

- Default: 4096 independent training scenes per arm/seed, 200 validation scenes, and 200 test scenes per amplitude/SNR. Three paths. Seeds 11, 22, 33.
- Every scene uses one pair of adjacent tones (0,1) from K=32. All 8 views share propagation paths. Full-band noise generation precedes pair selection.
- Same latent training scenes and standardized noise across large-only/mixed arms. The three mixed arms receive exactly the same arrays. Independent RNG streams separate train/validation/test scenes. SNR is balanced across 0, 10, 20 dB; mixed amplitude/SNR combinations are balanced to within one scene.
- Geometry seed 2026 reproduces the EDA generator convention. Path sampling differs from the prior EDA run, so do not join old EDA rows to this experiment by sample index.
- Test amplitudes: 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5. Within a seed the SAME test latent scenes and standardized noise are reused across amplitudes and SNRs. These repeats are not additional independent scenes.
- Equal epochs, batch size and total optimization steps across arms. Large-only sees more large-deformation exposures. Consequently, large-only versus mixed tests the practical training allocation AND learning several regimes together; it does not isolate those two causes.
- Fixed 30 epochs, Adam 0.001, gradient clipping 1.0, mean per-scene channel NMSE loss. No early stopping. Every arm's best checkpoint is selected using the SAME validation criterion: mean linear NMSE at alpha=0.5, balanced across SNR. The primary question is large-deformation performance; small-deformation tests measure secondary trade-offs.
- Five 3x3 convolution/BatchNorm/ReLU blocks; metadata-conditioned FiLM; 1x1 four-channel output. Full prediction = first-view estimate + network correction. Normalization is observation-only peak magnitude over all views and both tones.
- Full unitary pilots, 5x5 arrays at lambda/8 spacing, phase-only far-field channel and the original polar-angle convention. Geometry is known, with no geometry measurement error. Extra views add measurement energy. No checkpoint from earlier work is loaded.

## What to send back

- PASTE_BACK.md: compact primary-endpoint scores.
- summary.csv: mean linear NMSE, dB of that mean, median and p90 by arm/seed/amplitude/SNR.
- per_scene_errors.csv: matched errors plus similarity, residual and phase diagnostics.
- paired_comparisons.csv: per-seed bootstrap intervals for mean NMSE differences at alpha=0.5; negative favors the named arm over mixed. Pointwise 95% intervals, not multiplicity-adjusted; they resample scenes, not repeated subcarriers or conditions. Seed variability is reported separately.
- training_history.csv: convergence and validation history. Check whether more epochs/data are needed before interpreting a poor score as a structural limitation.
- seed_*/ manifests and best checkpoints, codebook.npz, parameter_counts.json and configuration.json: reproducibility.

Baselines `first_view` and `mean_view` are computed from exactly the same observations. All reported model predictions are full channel estimates, not raw residuals. PASTE_BACK averages per-seed dB scores; summary.csv retains linear means for other aggregation choices. Condition-specific test errors are evaluated only after checkpoint selection.

## Interpretation guide

- Large-only consistently beats mixed at 0.5: concentrated training helps; next distinguish exposure count from cross-regime interference.
- Ratio and geometry both beat mixed: known deformation information helps this setup.
- Geometry beats ratio: its encoding may help optimization, but fixed-codebook data cannot establish new-shape generalization.
- All models remain poor: inspect learning curves and training fit first. This is not proof of information loss; use the EDA oracle/local results as separate conditional diagnostics.
- A small difference in one seed is not a stable result. Compare paired intervals and replication across seeds.

Validation performed before packaging: data dimensions, undeformed identity, paired target directions and split separation; Python syntax checks. Full training requires PyTorch and has not been executed in the authoring environment. Use QUICK=True to validate the complete training pipeline on Kaggle before the full run.
