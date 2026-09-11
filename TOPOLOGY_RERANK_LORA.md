# Exact reranking and topology-distilled LoRA

This route replaces LTSN as the active generation-control experiment. LTSN remains an
exploratory negative result and has no authorization role in this pipeline.

```text
Frozen exact 18-D scorer
  |-- direct: generate N=8 -> technical-quality filter -> minimum exact loss -> winner
  `-- distill: train-split winners -> topology_focus LoRA -> one-shot generation
```

The exact scorer is both the teacher and final evaluator. LoRA never differentiates through
topology and is not sampler-time latent guidance: it distills repeated best-of-8 decisions into
ACE-Step's attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Untagged baseline
replay samples preserve the base distribution; winner samples receive the `topology_focus` tag.

## Fixed boundaries

- The direct branch must first pass the existing formal 32-prompt x 8-candidate topology and
  quality/prompt/diversity non-inferiority gate.
- Teacher data use only the `train` prompt families. Prompt families cannot cross splits.
- Development prompts select one of the predeclared scales: 0.5, 0.75, or 1.0.
- Qualification uses fresh prompt families and seeds, and must use exactly the selected scale.
- A qualification topology pass still does not authorize a production claim. Blind quality,
  prompt-consistency, and diversity non-inferiority evidence remains required.
- Frozen thresholds are never lowered to force a pass.

## Environment

Run on the Linux/CUDA machine from the repository root. By default the wrapper uses
`ACE-Step-1.5/.venv/bin/python`. Override it only when necessary:

```bash
export PYTHON_BIN=/path/to/ACE-Step-1.5/.venv/bin/python
```

The teacher run contains 320 x 8 = 2560 three-minute candidates. With 48 kHz stereo float WAV,
raw candidates alone are roughly 170 GiB; exact-scoring intermediates and LoRA tensors can push
the working requirement into the 250-400 GiB range. The new reranking configuration disables
latent saving.

## A. Direct best-of-8 evidence

```bash
bash scripts/run_topology_rerank_lora.sh prepare-prompts
bash scripts/run_topology_rerank_lora.sh rerank-formal
bash scripts/run_topology_rerank_lora.sh init-formal-evidence
```

Complete the generated blind-evaluation table using the frozen non-inferiority protocol, then:

```bash
export NONINFERIORITY_TABLE=/absolute/path/to/noninferiority_results.csv
bash scripts/run_topology_rerank_lora.sh evaluate-formal-evidence
bash scripts/run_topology_rerank_lora.sh issue-formal-gate
```

If gate issuance fails, stop. Do not generate a LoRA teacher dataset.

## B. Winner teacher set and LoRA

The reranking run is resumable; rerun the same command after an interruption.

```bash
bash scripts/run_topology_rerank_lora.sh rerank-train
bash scripts/run_topology_rerank_lora.sh export-teacher
bash scripts/run_topology_rerank_lora.sh preprocess-lora
bash scripts/run_topology_rerank_lora.sh train-lora
bash scripts/run_topology_rerank_lora.sh finalize-lora
```

Important outputs:

- `runs/topology_rerank_lora_v1/teacher/teacher_report.json`
- `runs/topology_rerank_lora_v1/teacher/ace_lora_dataset.json`
- `runs/topology_rerank_lora_v1/audit/preprocess_plan.json`
- `runs/topology_rerank_lora_v1/audit/train_plan.json`
- `runs/topology_rerank_lora_v1/lora_artifact.json`

`lora_artifact.json` initially has status `trained_unqualified`; training completion is not a
scientific result.

## C. Development scale selection

Run all three frozen scales. Each run generates Base and LoRA once with the same semantic prompt
and seed, then applies the exact scorer to the decoded WAV files.

```bash
for scale in 0.5 0.75 1.0; do
  LORA_SCALE="$scale" bash scripts/run_topology_rerank_lora.sh validate-development
done
bash scripts/run_topology_rerank_lora.sh select-scale
```

If `scale_selection.json` says `not_supported`, stop and report LoRA as unsupported. Do not inspect
qualification to choose another scale.

## D. Fresh qualification

Read `selected_scale` from `runs/topology_rerank_lora_v1/scale_selection.json`, then run exactly
that value:

```bash
LORA_SCALE=0.75 bash scripts/run_topology_rerank_lora.sh validate-qualification
```

The final exact report is:

```text
runs/topology_rerank_lora_v1/validation/qualification/validation_report.json
```

Interpret `qualification_topology_confirmed=true` only as paired exact-topology support. The report
deliberately keeps `production_authorization=false` until the separate blind non-inferiority study
is attached.
