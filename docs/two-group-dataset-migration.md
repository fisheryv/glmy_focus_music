# Focus–Classical two-group dataset migration

Migration date: 2026-08-02.

## Outcome

The canonical study corpus now contains only Jamendo Open Focus and Classical music.
The former Pop comparison group was not deleted: it was moved to the local, Git-ignored
archive `dataset_archive/pop_music_legacy_2026-08-02/`.

| Item | Canonical | Archived Pop |
|---|---:|---:|
| Source tracks | 600 | 300 |
| Focus / Classical | 300 / 300 | — |
| Discovery / validation / holdout | 416 / 139 / 45 | 224 / 76 / 0 |
| 180 s + 300 s segments | 1,200 | 600 |

The archive contains the Pop raw audio, preprocessed WAV files, canonical feature views,
sidecars, experimental Pop feature subtrees, Pop-only metadata rows, source manifests,
and the former three-group state model. Its `file_inventory.csv` records 6,914 files,
11.314 GiB, and a SHA-256 for every archived file. The inventory itself has SHA-256
`c050b27f9f20db03b32503f087a95d570e321c53b7013f96ad47ff8675141eb8`.

Per-track licenses remain authoritative. Archiving Pop does not change its CC attribution,
NC, ND, or SA obligations.

## New canonical feature model

The state model was refitted only on discovery/180s segments from Focus (195) and
Classical (221), with balanced sampling for acoustic and rhythm windows. Model SHA-256:

`a39de25ea9f410ca5f96e86f4255d44be233a9179549f602d379f6cb72f9412f`

All 1,200 canonical segments were transformed with zero failures. Machine-readable
sources are `metadata/dataset_summary.json`, `metadata/preprocessing_summary.json`,
`metadata/feature_summary.json`, and `metadata/pop_archive_migration_audit.json`.

## Evidence boundary

This migration updates the dataset and canonical feature layer; it does not convert old
three-group analyses into two-group evidence. Existing topology, TDA, repetition,
classification, and generation reports remain historical until rerun using the new
manifest and model. Their old numerical results should not be quoted as current
Focus–Classical findings.
