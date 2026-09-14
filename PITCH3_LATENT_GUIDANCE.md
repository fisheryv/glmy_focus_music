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
