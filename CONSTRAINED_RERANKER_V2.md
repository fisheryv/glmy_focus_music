# Global constrained reranker v2

V2 keeps the frozen topology, prompt, diversity, technical-quality, and blind-quality gates.
It changes only the prospectively declared selection procedure and candidate support:

- best-of-16 with new calibration seeds;
- deterministic exact global assignment instead of sequential greedy substitutions;
- zero prompt and diversity margins;
- no topology-neutral support substitutions;
- fail-closed search with a hashed feasibility audit.

V1 artifacts under `runs/topology_rerank_lora_v1/constrained` remain unchanged negative
calibration evidence. V2 writes to `runs/topology_rerank_lora_v1/constrained_v2`.

## Calibration

Run in order:

```bash
bash scripts/run_constrained_reranker_v2.sh prepare-prompts
bash scripts/run_constrained_reranker_v2.sh calibration-run
bash scripts/run_constrained_reranker_v2.sh calibration-semantics
bash scripts/run_constrained_reranker_v2.sh calibration-apply
```

Inspect both:

```text
runs/topology_rerank_lora_v1/constrained_v2/rerank/
  topology_constrained_calibration_v2/constrained/feasibility_audit.json
  topology_constrained_calibration_v2/constrained/constrained_report.json
```

The feasibility audit reports the complete search size, candidate-filter funnel, required
effectful-pool majority, and maximum feasible effectful pools. Continue only when
`calibration_supported=true`; neither an incomplete search nor an optimistic upper bound may
authorize confirmation.

Freeze the supported selector before generating any confirmation audio:

```bash
bash scripts/run_constrained_reranker_v2.sh freeze-selector
```

## Untouched confirmation

The confirmation manifest retains the previously reserved 32 unseen prompt families and uses
new best-of-16 seeds beginning at `2026120100`:

```bash
bash scripts/run_constrained_reranker_v2.sh confirmation-run
bash scripts/run_constrained_reranker_v2.sh confirmation-semantics
bash scripts/run_constrained_reranker_v2.sh confirmation-apply
```

Continue only when `confirmation_ready_for_noninferiority=true`. Do not inspect confirmation and
then change the selector.

## Unchanged blind non-inferiority gate

```bash
bash scripts/run_constrained_reranker_v2.sh init-evidence
bash scripts/run_constrained_reranker_v2.sh quality-template
```

Complete the new blinded quality table, then run:

```bash
export QUALITY_TABLE=/absolute/path/to/blind_quality_aggregated.csv
bash scripts/run_constrained_reranker_v2.sh final-metrics
bash scripts/run_constrained_reranker_v2.sh evaluate-evidence
bash scripts/run_constrained_reranker_v2.sh issue-gate
```

The last command is authorized only after every unchanged formal gate passes. LoRA winner
distillation remains blocked until that point.
