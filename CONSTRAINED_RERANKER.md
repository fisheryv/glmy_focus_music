# Constrained reranker calibration and fresh confirmation

This v2 route preserves the existing exact-topology, blind-quality, prompt, and diversity gates.
It does not reinterpret the failed `topology_bestof8_formal_v1` run. That run remains a negative
formal result and is not reused for calibration or confirmation.

## Frozen design

- Calibration: 64 prompts from four calibration-only families, seeds beginning at `2026100100`.
- Confirmation: 32 pre-frozen prompts from 32 families absent from the previous formal,
  train/development/calibration/qualification manifests, with seeds beginning at `2026102000`.
- Candidate pool: eight 180-second ACE-Step XL-Turbo generations per prompt.
- Prompt guard: selected CLAP alignment must not be below candidate 0; margin remains zero.
- Diversity guard: after every proposed substitution, every prompt's cohort nearest-neighbor
  diversity must remain at or above the all-baseline cohort; margin remains zero.
- Topology guard: substitutions must strictly reduce frozen exact 18-D loss.
- If no candidate satisfies every guard, candidate 0 is retained.

The selector uses the same frozen CLAP contract later checked by the formal gate. Consequently,
the prompt/diversity gate verifies compliance with the declared selection contract; it is not an
independent semantic-generalization test. Blind quality remains independent, and paper claims
must state this limitation.

## 1. Freeze prompt cohorts

```bash
bash scripts/run_constrained_reranker.sh prepare-prompts
```

Inspect:

```text
runs/topology_rerank_lora_v1/constrained/prompts/manifest.json
```

Expected cohorts:

- calibration: `p08`, `p19`, `p24`, `p31`
- confirmation: 32 new `q01`-`q32` families from
  `generation/prompts/ace_constrained_confirmation_v1.csv`

## 2. Calibration

```bash
bash scripts/run_constrained_reranker.sh calibration-run
bash scripts/run_constrained_reranker.sh calibration-semantics
bash scripts/run_constrained_reranker.sh calibration-apply
```

Review:

```text
runs/topology_rerank_lora_v1/constrained/rerank/
  topology_constrained_calibration_v1/constrained/constrained_report.json
```

Only continue when `calibration_supported=true`. The freeze command refuses unsupported
calibration rather than relaxing either margin:

```bash
bash scripts/run_constrained_reranker.sh freeze-selector
```

This creates `runs/topology_rerank_lora_v1/constrained/frozen_selector.json` before confirmation
generation starts.

## 3. Fresh confirmation

```bash
bash scripts/run_constrained_reranker.sh confirmation-run
bash scripts/run_constrained_reranker.sh confirmation-semantics
bash scripts/run_constrained_reranker.sh confirmation-apply
```

Continue only when the confirmation report says
`confirmation_ready_for_noninferiority=true`. Otherwise report the constrained reranker as not
supported; do not alter the selector and rerun on these confirmation prompts.

## 4. Unchanged non-inferiority gate

```bash
bash scripts/run_constrained_reranker.sh init-evidence
bash scripts/run_constrained_reranker.sh quality-template
```

Complete a new blinded quality study for the new selected/baseline pairs. Do not reuse ratings
from the failed v1 formal run. Then:

```bash
export QUALITY_TABLE=/absolute/path/to/blind_quality_aggregated.csv
bash scripts/run_constrained_reranker.sh final-metrics
bash scripts/run_constrained_reranker.sh evaluate-evidence
```

Check all three criteria in the new `noninferiority_report.json`. Only if exact topology and all
unchanged zero-margin criteria pass:

```bash
bash scripts/run_constrained_reranker.sh issue-gate
```

Gate issuance reloads and hash-verifies the frozen calibration selector, semantic audit,
selection contract, exact descriptors, and candidate manifest before recomputing the summary.
The passed artifact is written separately as
`metadata/ace_constrained_reranking_effect_gate.json`; the failed v1 evidence is not overwritten.

## Stop rules

- `calibration_supported=false`: stop before confirmation.
- `confirmation_ready_for_noninferiority=false`: stop before blind evaluation.
- Any non-inferiority criterion fails: preserve the result and do not issue a gate.
- Do not train the formal LoRA teacher until the new constrained reranking gate passes.
