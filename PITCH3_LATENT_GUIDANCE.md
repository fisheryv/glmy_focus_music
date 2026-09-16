# Pitch-3 latent topology guidance

This development track preserves the signed 18-D scorer and introduces a
separate, post-validation three-coordinate teacher for low-resource latent
control experiments.

## Frozen teacher

The exact teacher uses only local Pitch path-topology descriptors:

1. `h0_observed_persistence` - local connectivity and state coverage;
2. `self_transition_ratio` - local stability and dwell;
3. `directed_recurrence` - directed repetition.

The transform is fitted only on discovery/180s. Each coordinate is centered by
the discovery median and divided by a robust discovery scale. The downstream
Focus target is the coordinate-wise 10th-90th percentile band of discovery
Focus. The artifact is intentionally marked as a post-validation downstream
refreeze and cannot authorize sampling guidance.

Rebuild and verify the signed files with:

```powershell
$env:PYTHONPATH = 'src;packages/pyglmy/src'
.\.venv\Scripts\python.exe scripts\build_focus_pitch3_fingerprint.py
.\.venv\Scripts\python.exe -m pytest tests\test_pitch3_fingerprint.py
```

## Exact trajectory labels

Collect ACE predicted-clean latent snapshots and independently decode every
step-4/5/6 snapshot exactly as in the existing trajectory protocol. The
descriptor table must contain one `decoded_snapshot_exact_v1` row per snapshot,
including `pitch_descriptors_json`, `audio_sha256`, and `ood_label`.

```powershell
focus-pitch3-labels `
  --trajectory-manifest runs/<run>/trajectories/trajectory_manifest.csv `
  --descriptor-table runs/<run>/descriptors/exact_snapshot_descriptors.csv `
  --output-manifest runs/<run>/pitch3/manifest.csv `
  --exact-label-table runs/<run>/pitch3/exact_labels.csv `
  --split-manifest runs/<run>/pitch3/splits.json
```

Prompt and trajectory identities must stay in one split. Formal labels require
decoded snapshot audio with matching hashes; `--engineering-smoke` produces a
non-qualifying plumbing artifact only.

## Latent topology control head

`LatentTopologyControlHead` consumes `[B,T,64]` predicted-clean ACE latents and
the sampling time/step. A depthwise temporal downsampler and lightweight
bidirectional Transformer produce three coordinate means, three log variances,
an OOD logit, and the frozen Focus logit. The default configuration remains
below the approximately 7M-parameter low-resource reference scale.

```powershell
focus-pitch3-train `
  --manifest runs/<run>/pitch3/manifest.csv `
  --output-dir runs/<run>/pitch3/checkpoints `
  --device cuda:0
```

Training checkpoints bind the fingerprint, configuration, and data-manifest
SHA-256 values. Issued checkpoints always set
`guidance_promotion_eligible=false` and `production_authorization=false`.
Real CUDA training, OOD calibration, decoded-exact direction agreement,
candidate ranking, and audio/prompt non-inferiority must all be rerun before any
sampling-time corrector can be enabled.

## V2.6-GR-R development run

V2.6-GR-R keeps the V2.6-GR network and frozen evaluation gates, but replaces
the trajectory-anchor sampler with family-aware snapshot pairs. Every 6-ID/2-OOD
batch contains exactly two low-, two middle-, and two high-Band snapshots and
three same-family/same-step contrasts: low-high, low-middle, and middle-high.
The coordinate-2 ranking and trajectory-delta auxiliaries are disabled. Model
selection first requires all pooled gates and the predeclared E2 development
Band floor (`0.5143053354087133`), then minimizes the group deficit.

```bash
python scripts/train_pitch3_control_head.py \
  --fingerprint metadata/focus_pitch3_fingerprint_v1.json \
  --manifest runs/pitch3_ltch/ood_v2/pitch3_training_manifest_augmented.csv \
  --config configs/pitch3_control_head_training_v26_grr.toml \
  --output-dir runs/pitch3_ltch/models_pitch3_v26_grr_seed_20260919 \
  --device cuda:1

python scripts/evaluate_pitch3_control_head.py screen-development \
  --fingerprint metadata/focus_pitch3_fingerprint_v1.json \
  --manifest runs/pitch3_ltch/ood_v2/pitch3_training_manifest_augmented.csv \
  --checkpoint runs/pitch3_ltch/models_pitch3_v26_grr_seed_20260919/pitch3_control_head_seed_20260919.pt \
  --output-dir runs/pitch3_ltch/development_screen_v26_grr_seed_20260919 \
  --batch-size 8 \
  --device cuda:1
```

Do not run calibration unless both the pooled and required group development
screens pass. The qualification split remains excluded from training,
checkpoint selection, and this development decision.

## V3.7-DTE latent topology energy

V3.7-DTE retains the frozen Pitch-3 exact Band as the teacher and evaluation
target, but removes the hard analytic Band calculation from the proxy's
guidance path. The global proxy directly predicts
`log1p(exact_pitch3_target_band_loss)` with a latent-primary non-negative scalar
head and a bounded prompt residual. Its three-coordinate head is latent-only,
width-normalized, and auxiliary: coordinate predictions never enter the
guidance energy. The anchored local term remains
`R(z,c)-R(z0,c)` and uses the V3.5R direction/flat objective.

The existing V3 exact-local dataset is reused without relabeling:

```bash
bash scripts/run_pitch3_lte_v37_dte.sh train
bash scripts/run_pitch3_lte_v37_dte.sh ensemble
bash scripts/run_pitch3_lte_v37_dte.sh model-screen
```

The default run uses seeds `20260938 20260939 20260940` and devices from
`LTE_DEVICES` (default `cuda:1 cuda:2 cuda:3`). Do not run `guidance` until the
model-only development screen passes every frozen model gate.

## V3.8-A logit ranking

V3.8-A keeps the V3.7 model and energy readout, `Eglobal = softplus(a)`.
Same-prompt ranking uses the explicit pre-softplus logit `a` at temperature 1.
The family soft-Spearman term is replaced by RankNet on all sortable base pairs
within each family: mean within each unordered energy-stratum pair, then mean
over nonempty stratum pairs, then mean over families. The zero bucket and three
positive-energy strata are fitted on the current training subset only. Pairs
with `abs(exact_band_i - exact_band_j) <= 1e-6` are excluded, including ties.
Value supervision, width-normalized coordinate auxiliary loss, anchored local
direction/flat objective, and inference gates retain their V3.7 definitions.
Ensembles average member energies; there is no ensemble training logit.

Global loss normalization and validation loss use full-family base batches;
local terms use prompt batches. `pitch3_lte_training_statistics.json` and
`pitch3_lte_run_protocol.json` bind the fitted scales and selected sample IDs.
The model revision is `v3.8a_logit_stratified_rank_energy` even though its model
configuration deliberately retains `potential_mode = "direct_anchored_v37"`.

Run the numerical tests in the server Torch environment first:

```bash
python -m pytest tests/test_pitch3_lte.py tests/test_pitch3_lte_v38a.py -ra
export LTE_DEVICES="cuda:1 cuda:2 cuda:3"
bash scripts/run_pitch3_lte_v38a.sh cv
```

`cv` is the default stage. It uses seed 20260941 and five deterministic folds
of the original 20 train families, holding out four families per fold. All
prompt variants, seeds, anchors and local pairs stay with their family. The
original development families never enter CV training or selection. Each fold
refits strata/scales on its 16 training families. The default CV run trains
only global energy (`R=0`) and writes `runs/pitch3_lte_v38a/cv/` plus the
hash-checked `pitch3_lte_cv_summary.json`. CV checkpoints cannot be loaded into
the development screen, guidance, or a guidance ensemble.

Matched V3.7 control, with the same CV seed and folds:

```bash
LTE_TRAIN_CONFIG="$PWD/configs/pitch3_lte_v37_dte.toml" \
LTE_V38A_CV_ROOT="$PWD/runs/pitch3_lte_v38a/cv_v37_control" \
bash scripts/run_pitch3_lte_v38a.sh cv
```

To measure the local residual increment in a separate output directory:

```bash
LTE_CV_GLOBAL_ONLY=0 \
LTE_V38A_CV_ROOT="$PWD/runs/pitch3_lte_v38a/cv_with_local" \
bash scripts/run_pitch3_lte_v38a.sh cv
```

The A/control comparison includes the corrected full-family normalization for
A; their normalized loss totals are not directly comparable. Compare exact
metrics. The summary reports all folds and never selects a model automatically.
These CV runs retain the existing per-run global checkpoint rule (global gate
deficit, minimum family correlation, then normalized loss); they do not change
the published gates or claim independent qualification.

Once the development experiment is specified, the existing three-seed workflow
is available:

```bash
bash scripts/run_pitch3_lte_v38a.sh train
bash scripts/run_pitch3_lte_v38a.sh ensemble
bash scripts/run_pitch3_lte_v38a.sh model-screen
# Equivalent: bash scripts/run_pitch3_lte_v38a.sh all
```

Default seeds are `20260941 20260942 20260943`; outputs are in
`runs/pitch3_lte_v38a/`. These full-data training commands retain the existing
development-based checkpoint selection, so they are not a single-look blind
evaluation. Use CV results to specify the final experiment before running them.
`LTE_GLOBAL_ONLY=1` enables the explicit zero-local control for full-data runs;
use a separate `LTE_V38A_RUN_ROOT` for that experiment. The default keeps the
original two-stage local training. `LTE_FINGERPRINT` can point to the archived
server fingerprint; its hash must match the existing exact-local dataset.
No raw audio generation, dataset rewrite, or threshold change is required.
