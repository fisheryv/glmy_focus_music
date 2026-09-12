# Constrained-reranker distilled LoRA v2

This route is authorized only by the passed constrained-reranker v2 gate. It does not reuse
calibration or confirmation audio as LoRA training data and does not overwrite the historical
best-of-8 LoRA route.

## Frozen design

- Teacher cohort: 320 family-disjoint `train` prompts.
- Candidate support: 16 new 180-second generations per prompt, beginning at seed `2026130100`.
- Selection: the calibration-frozen global constrained-reranker v2 selector.
- Teacher rows: selected winner with `topology_focus`; when the selected winner differs from
  candidate 0, candidate 0 is also included as untagged baseline replay.
- Teacher export is hash-bound to the passed v2 gate, frozen selector, teacher selection
  contract, candidate manifest, exact descriptor table, and source audio.
- LoRA hyperparameters remain frozen in `configs/topology_lora_v2.json` before preprocessing.

The teacher run plans 5,120 generated tracks. Confirm disk capacity before `teacher-run`.

## Teacher generation and constrained selection

```bash
bash scripts/run_topology_rerank_lora_v2.sh prepare-prompts
bash scripts/run_topology_rerank_lora_v2.sh check-gate
bash scripts/run_topology_rerank_lora_v2.sh teacher-run
bash scripts/run_topology_rerank_lora_v2.sh teacher-semantics
bash scripts/run_topology_rerank_lora_v2.sh teacher-apply
```

Inspect:

```text
runs/topology_rerank_lora_v1/lora_v2/rerank/
  topology_lora_teacher_constrained_v2/constrained/constrained_report.json
  topology_lora_teacher_constrained_v2/constrained/feasibility_audit.json
  topology_lora_teacher_constrained_v2/constrained/selection_contract.json
```

For the 320-prompt teacher cohort, `confirmation_ready_for_noninferiority=false` is expected
because this is not the 32-prompt formal confirmation design. Continue only when
`teacher_selection_ready=true`, `search_complete=true`, and both frozen semantic guards pass.

Export the hash-bound dataset:

```bash
bash scripts/run_topology_rerank_lora_v2.sh export-teacher
```

Review `runs/topology_rerank_lora_v1/lora_v2/teacher/teacher_report.json`. Do not edit the
dataset JSON or source audio after export.

## LoRA preprocessing and training

```bash
bash scripts/run_topology_rerank_lora_v2.sh preprocess-lora
bash scripts/run_topology_rerank_lora_v2.sh train-lora
bash scripts/run_topology_rerank_lora_v2.sh finalize-lora
```

The finalized adapter remains `trained_unqualified`; training alone is not production evidence.

## Development scale selection and qualification

Run all three predeclared scales on the same development prompts and seeds:

```bash
LORA_SCALE=0.5  bash scripts/run_topology_rerank_lora_v2.sh validate-development
LORA_SCALE=0.75 bash scripts/run_topology_rerank_lora_v2.sh validate-development
LORA_SCALE=1.0  bash scripts/run_topology_rerank_lora_v2.sh validate-development
bash scripts/run_topology_rerank_lora_v2.sh select-scale
```

Read the selected scale from:

```bash
jq '.selected_scale' runs/topology_rerank_lora_v1/lora_v2/scale_selection.json
```

Use that single value once on qualification, without returning to development tuning:

```bash
LORA_SCALE=<selected-scale> \
  bash scripts/run_topology_rerank_lora_v2.sh validate-qualification
```

Qualification topology support still does not waive the separately required blinded quality,
prompt, and diversity non-inferiority evaluation for Base versus LoRA.
