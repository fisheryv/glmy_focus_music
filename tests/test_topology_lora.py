from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from generation.ace_adapter import AceStepAdapter
from generation.experiment import (
    AceConfig,
    CandidateRecord,
    ExperimentConfig,
    ScoringConfig,
    load_experiment_config,
    write_candidate_manifest,
)
from generation.ltsn_contract import LTSNContractError, sha256_file
from generation.ltsn_pipeline import canonical_json_sha256
from generation.path_homology_exact_scorer import ExactPathHomologyScorer
from generation.topology_lora import (
    TOPOLOGY_LORA_EXPERIMENT_V2,
    build_reranking_prompt_splits,
    export_lora_teacher_dataset,
)
from generation.topology_lora_training import build_native_command
from generation.topology_lora_validation import (
    select_development_scale,
    summarize_paired_validation,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prompt_splits_are_family_disjoint_and_hashed(tmp_path: Path) -> None:
    source = tmp_path / "prompts.csv"
    rows = []
    for index, split in enumerate(("train", "development", "calibration", "qualification")):
        rows.append(
            {
                "prompt_id": f"family_{index}__v1",
                "caption": f"prompt {index}",
                "split": split,
                "bpm": 80,
                "keyscale": "",
                "timesignature": 4,
            }
        )
    _write_csv(source, rows)

    payload = build_reranking_prompt_splits(source, tmp_path / "out")

    assert payload["family_disjoint"] is True
    assert payload["outputs"]["train"]["prompts"] == 1
    assert sha256_file(tmp_path / "out" / "train.csv") == payload["outputs"]["train"]["sha256"]

    rows[1]["prompt_id"] = "family_0__v2"
    _write_csv(source, rows)
    with pytest.raises(LTSNContractError, match="family leakage"):
        build_reranking_prompt_splits(source, tmp_path / "leaked")


def test_prompt_override_is_explicit_and_does_not_change_default() -> None:
    path = Path("configs/ace_rerank_180s.toml")
    default = load_experiment_config(ROOT, path)
    overridden = load_experiment_config(
        ROOT,
        path,
        run_id="teacher_run",
        prompt_manifest="runs/topology/prompts/train.csv",
    )

    assert default.prompt_manifest == "generation/prompts/ace_rerank_formal.csv"
    assert overridden.prompt_manifest == "runs/topology/prompts/train.csv"
    assert overridden.run_id == "teacher_run"


def _teacher_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    manifest_path = run_dir / "manifests" / "candidates.csv"
    records = []
    scores = []
    for index, loss in enumerate((1.0, 0.1)):
        candidate_id = f"p1__c{index:02d}__s{10 + index}"
        audio_path = run_dir / "data_raw" / "candidates" / f"{candidate_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(f"audio-{index}".encode())
        records.append(
            CandidateRecord(
                experiment_id="teacher",
                prompt_id="p1",
                caption="steady instrumental",
                candidate_index=index,
                candidate_id=candidate_id,
                seed=10 + index,
                duration_seconds=180,
                status="scored",
                audio_relative_path=audio_path.relative_to(run_dir).as_posix(),
                audio_sha256=sha256_file(audio_path),
            )
        )
        scores.append(
            {
                "candidate_id": candidate_id,
                "focus_band_loss": loss,
                "technical_quality_eligible": 1,
            }
        )
    write_candidate_manifest(manifest_path, records)
    score_path = run_dir / "scores.csv"
    pool_path = run_dir / "pool_summary.csv"
    _write_csv(score_path, scores)
    _write_csv(
        pool_path,
        [
            {
                "prompt_id": "p1",
                "baseline_candidate_id": records[0].candidate_id,
                "selected_candidate_id": records[1].candidate_id,
            }
        ],
    )
    scorer = ExactPathHomologyScorer.from_json(
        ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "candidate_manifest_sha256": sha256_file(manifest_path),
                "selection_table_sha256": sha256_file(score_path),
                "pool_summary_sha256": sha256_file(pool_path),
            }
        ),
        encoding="utf-8",
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "gate": "exact_reranking_effect_v1",
                "status": "passed",
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "median_loss_improvement_fraction": 0.2,
                "bootstrap_ci95_low": 0.01,
                "target_band_hit_rate_improved": True,
                "quality_noninferior": True,
                "prompt_noninferior": True,
                "diversity_preserved": True,
            }
        ),
        encoding="utf-8",
    )
    prompts = tmp_path / "train.csv"
    _write_csv(
        prompts,
        [
            {
                "prompt_id": "p1",
                "caption": "steady instrumental",
                "split": "train",
                "bpm": 80,
                "keyscale": "",
                "timesignature": 4,
            }
        ],
    )
    return run_dir, gate_path, prompts


def _add_v2_selection_fixture(run_dir: Path, gate_path: Path) -> Path:
    constrained = run_dir / "constrained"
    constrained.mkdir()
    selector = constrained / "selector.json"
    selector.write_text('{"selector": "v2"}', encoding="utf-8")
    frozen = constrained / "frozen_selector.json"
    frozen.write_text('{"status": "frozen"}', encoding="utf-8")
    semantic = constrained / "semantic_audit.json"
    semantic.write_text('{"semantic": "bound"}', encoding="utf-8")
    selection = constrained / "constrained_selection.csv"
    _write_csv(
        selection,
        [
            {
                "prompt_id": "p1",
                "selected_candidate_id": "p1__c01__s11",
            }
        ],
    )
    descriptors = run_dir / "descriptors_18d.csv"
    descriptors.write_text("candidate_id\np1__c01__s11\n", encoding="utf-8")
    candidate_manifest = run_dir / "manifests" / "candidates.csv"
    selector_sha256 = sha256_file(selector)
    contract = {
        "schema_version": 1,
        "experiment": "exact_topology_constrained_reranker_v2",
        "run_id": run_dir.name,
        "selector_config_path": "constrained/selector.json",
        "selector_config_sha256": selector_sha256,
        "frozen_selector_path": "constrained/frozen_selector.json",
        "frozen_selector_sha256": sha256_file(frozen),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "descriptor_table_sha256": sha256_file(descriptors),
        "semantic_audit_sha256": sha256_file(semantic),
        "selection_table_sha256": sha256_file(selection),
        "feasibility_audit_sha256": None,
        "selected_candidates": {"p1": "p1__c01__s11"},
        "diagnostic": {"search_complete": True},
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    contract_path = constrained / "selection_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    policy = {
        "name": "exact_topology_constrained_reranker_v2",
        "selection_contract_path": "constrained/selection_contract.json",
        "selection_contract_sha256": sha256_file(contract_path),
        "selector_config_sha256": selector_sha256,
    }
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["descriptor_table_sha256"] = sha256_file(descriptors)
    summary["selection_policy"] = policy
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["selection_policy"] = policy
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return contract_path


def test_teacher_export_tags_only_exact_winner(tmp_path: Path) -> None:
    run_dir, gate_path, prompts = _teacher_fixture(tmp_path)
    output = tmp_path / "teacher"

    report = export_lora_teacher_dataset(
        reranking_run_dir=run_dir,
        prompt_manifest_path=prompts,
        fingerprint_path=ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json",
        reranking_gate_path=gate_path,
        output_dir=output,
    )
    dataset = json.loads((output / "ace_lora_dataset.json").read_text(encoding="utf-8"))

    assert report["winner_samples"] == 1
    assert report["baseline_replay_samples"] == 1
    assert [sample["custom_tag"] for sample in dataset["samples"]] == [
        "topology_focus",
        "",
    ]

    winner = run_dir / "data_raw" / "candidates" / "p1__c01__s11.wav"
    winner.write_bytes(b"changed")
    with pytest.raises(LTSNContractError, match="changed teacher audio"):
        export_lora_teacher_dataset(
            reranking_run_dir=run_dir,
            prompt_manifest_path=prompts,
            fingerprint_path=ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json",
            reranking_gate_path=gate_path,
            output_dir=tmp_path / "changed",
        )


def test_v2_teacher_export_requires_and_binds_constrained_selection(tmp_path: Path) -> None:
    run_dir, gate_path, prompts = _teacher_fixture(tmp_path)
    contract_path = _add_v2_selection_fixture(run_dir, gate_path)
    output = tmp_path / "teacher_v2"

    report = export_lora_teacher_dataset(
        reranking_run_dir=run_dir,
        prompt_manifest_path=prompts,
        fingerprint_path=ROOT / "metadata" / "focus_path_homology_fingerprint_v2.json",
        reranking_gate_path=gate_path,
        selection_contract_path=contract_path,
        output_dir=output,
        experiment=TOPOLOGY_LORA_EXPERIMENT_V2,
    )

    dataset = json.loads((output / "ace_lora_dataset.json").read_text(encoding="utf-8"))
    assert report["experiment"] == TOPOLOGY_LORA_EXPERIMENT_V2
    assert report["selection_contract_sha256"] == sha256_file(contract_path)
    assert report["selector_config_sha256"] is not None
    assert dataset["metadata"]["name"] == TOPOLOGY_LORA_EXPERIMENT_V2


def test_native_training_command_is_bound_to_teacher(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    dataset = teacher / "ace_lora_dataset.json"
    dataset.write_text('{"samples": [{"audio_path": "x.wav"}]}', encoding="utf-8")
    (teacher / "teacher_report.json").write_text(
        json.dumps(
            {
                "experiment": "exact_reranking_distilled_topology_lora_v1",
                "dataset_json_sha256": sha256_file(dataset),
                "winner_samples": 1,
            }
        ),
        encoding="utf-8",
    )
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    (tensors / "sample.pt").write_bytes(b"tensor")

    command, plan = build_native_command(
        stage="train",
        project_root=ROOT,
        config_path=ROOT / "configs" / "topology_lora_v1.json",
        teacher_dir=teacher,
        tensor_dir=tensors,
        output_dir=tmp_path / "lora",
    )

    assert "--target-modules" in command
    assert command[command.index("--base-model") + 1] == "xl_turbo"
    assert plan["dataset_json_sha256"] == sha256_file(dataset)


def test_v2_native_training_plan_retains_v2_teacher_binding(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher_v2"
    teacher.mkdir()
    dataset = teacher / "ace_lora_dataset.json"
    dataset.write_text('{"samples": [{"audio_path": "x.wav"}]}', encoding="utf-8")
    (teacher / "teacher_report.json").write_text(
        json.dumps(
            {
                "experiment": TOPOLOGY_LORA_EXPERIMENT_V2,
                "dataset_json_sha256": sha256_file(dataset),
                "winner_samples": 1,
                "teacher_source": "frozen_constrained_exact_18d_bestof16",
                "reranking_gate_sha256": "a" * 64,
                "selection_contract_sha256": "b" * 64,
                "selector_config_sha256": "c" * 64,
                "frozen_selector_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    tensors = tmp_path / "tensors_v2"
    tensors.mkdir()
    (tensors / "sample.pt").write_bytes(b"tensor")

    _, plan = build_native_command(
        stage="train",
        project_root=ROOT,
        config_path=ROOT / "configs" / "topology_lora_v2.json",
        teacher_dir=teacher,
        tensor_dir=tensors,
        output_dir=tmp_path / "lora_v2",
    )

    assert plan["experiment"] == TOPOLOGY_LORA_EXPERIMENT_V2
    assert plan["dataset_json_sha256"] == sha256_file(dataset)


class _FakeHandler:
    def __init__(self) -> None:
        self.enabled = False

    def load_lora(self, path: str) -> str:
        return f"✅ loaded {path}"

    def set_lora_scale(self, scale: float) -> str:
        return f"✅ scale {scale}"

    def set_use_lora(self, enabled: bool) -> str:
        self.enabled = enabled
        return "✅ toggled"


def test_adapter_rejects_hash_drift_and_records_lora_state(tmp_path: Path) -> None:
    checkout = tmp_path / "ACE"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter.safetensors").write_bytes(b"weights")
    adapter = AceStepAdapter(checkout, AceConfig())
    handler = _FakeHandler()
    adapter._handler = handler
    adapter.initialize = lambda: None  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="hash differs"):
        adapter.load_lora(lora, scale=0.75, expected_sha256="0" * 64)

    from generation.artifact_hash import sha256_directory

    adapter.load_lora(lora, scale=0.75, expected_sha256=sha256_directory(lora))
    adapter.set_lora_enabled(False)
    assert handler.enabled is False


def test_paired_validation_and_scale_selection_are_development_only(tmp_path: Path) -> None:
    rows = []
    for prompt_index in range(20):
        for candidate_index, loss in enumerate((1.0, 0.0)):
            rows.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "candidate_index": candidate_index,
                    "seed": prompt_index,
                    "focus_band_loss": loss,
                    "raw_clip_fraction": 0.0,
                    "raw_rms": 0.1,
                    "raw_dc_offset": 0.0,
                }
            )
    result = summarize_paired_validation(
        rows=rows,
        config=ExperimentConfig(
            run_id="validation",
            prompt_manifest="prompts.csv",
            scoring=ScoringConfig(),
        ),
        validation_config=json.loads(
            (ROOT / "configs" / "topology_lora_v1.json").read_text(encoding="utf-8")
        ),
        split="development",
    )
    assert result["summary"]["topology_supported"] is True
    assert result["summary"]["development_scale_selection_eligible"] is True
    assert result["summary"]["production_authorization"] is False

    validation_root = tmp_path / "validation"
    for scale, change in ((0.5, -0.1), (0.75, -0.3), (1.0, -0.2)):
        report_dir = validation_root / f"development_scale_{scale}"
        report_dir.mkdir(parents=True)
        (report_dir / "validation_report.json").write_text(
            json.dumps(
                {
                    "split": "development",
                    "lora_scale": scale,
                    "lora_bundle_sha256": "a" * 64,
                    "prompt_manifest_sha256": "b" * 64,
                    "development_scale_selection_eligible": True,
                    "median_lora_minus_base_exact_loss": change,
                    "paired_topology_win_rate": 0.8,
                }
            ),
            encoding="utf-8",
        )
    selection = select_development_scale(
        validation_root=validation_root,
        lora_config_path=ROOT / "configs" / "topology_lora_v1.json",
        output_path=tmp_path / "selection.json",
    )
    assert selection["selected_scale"] == 0.75
    assert selection["qualification_authorized"] is True
