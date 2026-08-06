# Analysis result status after the 2026-08-02 two-group migration

Canonical data now contain Jamendo Open Focus (300) and Classical (300) only. Pop has
been removed from all canonical manifests and physically preserved under
`dataset_archive/pop_music_legacy_2026-08-02/`.

The canonical state model was refitted using only discovery/180s data from Focus (195)
and Classical (221). All 1,200 canonical segments were transformed successfully with
model SHA-256
`a39de25ea9f410ca5f96e86f4255d44be233a9179549f602d379f6cb72f9412f`.

Current validity:

- `metadata/track_index.csv`, `licenses.csv`, split files, and `dataset_summary.json`:
  current two-group dataset.
- `metadata/preprocessed_segments.csv` and `preprocessing_summary.json`: current.
- `metadata/feature_segments.csv`, `feature_summary.json`, and
  `features/models/state_model.*`: current two-group feature layer.
- The combined standard `metadata/topology_*` tables still use the pre-symmetric
  Classical 221/79/0 split and are historical. Use the separately rerun per-view
  outputs below. The primary analysis is validation/180s; validation/300s is
  sensitivity, and holdout is excluded from group tests.
- Standard pitch Path Homology is current for the symmetric two-group feature layer:
  all 1,200 segment views succeeded. Current outputs are under
  `metadata/pitch_topology_*`, `metadata/pitch_statistical_tests.csv`,
  `metadata/pitch_pairwise_tests.csv`, `metadata/pitch_permanova.csv`,
  `graphs/pitch`, `homology/*/pitch`, `docs/path-homology-pitch-analysis.md`, and
  `runs/pitch_path_homology_open/`.
- The former three-group Path Homology snapshot is preserved under
  `analysis_archive/path_homology_three_group_2026-08-02/` and is historical only.
- Specialized `pitch_v2` is current for the two-group dataset. Its codebook was refitted
  from balanced Focus/Classical discovery/180s beats, all 1,200 segments completed, and
  the numeric outputs, report, and figures are under `metadata/pitch_v2_*`,
  `features/pitch_v2`, `graphs/pitch_v2`, `homology/*/pitch_v2`,
  `docs/path-homology-pitch-v2-analysis.md`, and
  `runs/pitch_v2_path_homology_open/`. The former three-group `pitch_v2` snapshot is
  archived under `analysis_archive/pitch_v2_three_group_2026-08-02/`.
- Specialized rhythm Path Homology is current for the two-group dataset. The existing
  two-group discovery/180s state model was retained without post-hoc tuning; all 1,200
  rhythm segment views were recomputed. Current outputs are under `metadata/rhythm_*`,
  `graphs/rhythm`, `homology/*/rhythm`, `docs/path-homology-rhythm-analysis.md`, and
  `runs/rhythm_path_homology_open/`. The former three-group report, statistics, figures,
  and PDF are archived under
  `analysis_archive/rhythm_path_homology_three_group_2026-08-02/`.
- Specialized structure Path Homology is current for the two-group dataset. The existing
  two-group discovery/180s structure model was retained without post-hoc tuning; all
  1,200 structure segment views were recomputed. Current outputs are under
  `metadata/structure_*`, `features/structure`, `graphs/structure`,
  `homology/*/structure`, `docs/path-homology-structure-analysis.md`, and
  `runs/structure_path_homology_open/`. This view remains an exploratory extension
  outside the original three-view confirmatory family. The former three-group report,
  statistics, figures, and PDF are archived under
  `analysis_archive/structure_path_homology_three_group_2026-08-02/`. The duplicate
  legacy PDF at `output/pdf/path_homology_structure_analysis.pdf` is currently locked by
  another process and remains historical; use the current Markdown report instead.
- Specialized `modulation_tertile` Path Homology is current for the two-group dataset.
  It sums the three salient-band normalized SMP energy shares, fits two frozen tertile
  edges from group-balanced discovery/180s windows, and maps every valid window to one
  of exactly three states: Low, Medium, or High. All 1,200 segment views completed.
  Current outputs are under `metadata/modulation_tertile_*`,
  `features/modulation_tertile`, `graphs/modulation_tertile`,
  `homology/*/modulation_tertile`, `docs/path-homology-modulation-analysis.md`, and
  `runs/modulation_tertile_path_homology_open/`. This specialized view is separate from
  the standard `modulation` view, which retains up to 27 joint three-band states.
- Pop raw audio, preprocessed audio, Pop feature trees, Pop metadata rows, and the former
  three-group state model: archived and audit-only.
- TDA, repetition, four-group analyses,
  and generation-target result files remain historical until separately rerun with the
  two-group manifest and new state model.
- The phase-lifted Path Homology branch has now been rerun separately on the current
  two-group feature layer. Its current outputs use the prefix
  `metadata/phase_lifted_path_homology_*`, the report is
  `docs/phase-lifted-path-homology-analysis.md`, and PNG/SVG figures are under
  `runs/phase_lifted_path_homology_20260802/`. The older generic `repetition_*`
  outputs remain historical because they include the former Pop-based selection and
  sliding-window branches.

Do not carry three-group p-values, effect sizes, codebook-dependent descriptors, or
generation targets into the simplified study. The former H2 hypothesis was explicitly
defined as Focus vs Pop; because Pop is absent, it is not applicable to the new dataset
and was not retargeted post hoc to Classical.
