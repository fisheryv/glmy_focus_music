"""Prompt- and cohort-diversity-constrained exact topology reranking."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import ExperimentConfig, read_candidate_manifest
from .ltsn_contract import LTSNContractError, sha256_file
from .ltsn_pipeline import canonical_json_sha256, write_csv_atomic, write_json_atomic
from .noninferiority_metrics import (
    TransformersClapBackend,
    embed_candidates_by_audio_sha256,
    nearest_neighbor_diversity,
    normalize_embeddings,
)
from .rerank_experiment import experiment_root, rank_and_summarize

CONSTRAINED_EXPERIMENT = "exact_topology_constrained_reranker_v1"
CONSTRAINED_EXPERIMENT_V2 = "exact_topology_constrained_reranker_v2"
CONSTRAINED_EXPERIMENTS = {CONSTRAINED_EXPERIMENT, CONSTRAINED_EXPERIMENT_V2}
GLOBAL_OBJECTIVE = (
    "effectful_pool_count_descending",
    "median_relative_exact_improvement_descending",
    "new_target_band_hits_descending",
    "changed_pool_count_descending",
    "sum_relative_exact_improvement_descending",
    "candidate_id_vector_ascending",
)


@dataclass(frozen=True, slots=True)
class _GlobalOption:
    candidate_id: str
    loss: float
    prompt_alignment: float
    relative_improvement: float
    effectful: int
    target_hit: int
    changed: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError(f"CSV is empty: {path}")
    return rows


def _family(prompt_id: str) -> str:
    return prompt_id.split("__v", 1)[0]


def build_constrained_prompt_manifests(
    source: Path,
    confirmation_source: Path,
    output_dir: Path,
    *,
    experiment: str = CONSTRAINED_EXPERIMENT,
) -> dict[str, Any]:
    """Freeze calibration and 32-prompt family-new confirmation manifests."""

    rows = _read_csv(source)
    columns = ("prompt_id", "caption", "split", "bpm", "keyscale", "timesignature")
    if not set(columns).issubset(rows[0]):
        raise LTSNContractError("constrained prompt source is missing required columns")
    calibration = [row for row in rows if row["split"] == "calibration"]
    confirmation = _read_csv(confirmation_source)
    known_families = {_family(row["prompt_id"]) for row in rows}
    confirmation_families = {_family(row["prompt_id"]) for row in confirmation}
    if len(calibration) != 64:
        raise LTSNContractError("expected 64 calibration prompts")
    if len(confirmation) != 32 or len(confirmation_families) != 32:
        raise LTSNContractError("confirmation must contain 32 unique prompt families")
    if known_families & confirmation_families:
        raise LTSNContractError("confirmation families were seen in the existing prompt source")
    if {row.get("split") for row in confirmation} != {"confirmation"}:
        raise LTSNContractError("new confirmation prompts require split=confirmation")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, selected in (("calibration", calibration), ("confirmation", confirmation)):
        path = output_dir / f"{name}.csv"
        write_csv_atomic(path, [{key: row.get(key, "") for key in columns} for row in selected])
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "prompts": len(selected),
            "families": sorted({_family(row["prompt_id"]) for row in selected}),
        }
    payload = {
        "schema_version": 1,
        "experiment": experiment,
        "source_sha256": sha256_file(source),
        "confirmation_source_sha256": sha256_file(confirmation_source),
        "family_disjoint": True,
        "confirmation_selection": "all_32_pre_frozen_unseen_families",
        "outputs": outputs,
    }
    write_json_atomic(output_dir / "manifest.json", payload)
    return payload


def load_selector_config(path: Path) -> dict[str, Any]:
    """Load the prospective zero-margin constrained-selector contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") not in CONSTRAINED_EXPERIMENTS
        or payload.get("status") != "frozen_before_calibration"
    ):
        raise LTSNContractError("constrained selector config is not prospectively frozen")
    constraints = payload.get("constraints", {})
    if (
        float(constraints.get("prompt_margin", math.nan)) != 0.0
        or float(constraints.get("diversity_margin", math.nan)) != 0.0
        or constraints.get("strict_topology_improvement") is not True
    ):
        raise LTSNContractError("constrained selector must retain the frozen zero margins")
    revision = str(payload.get("embedding", {}).get("model_revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise LTSNContractError("constrained selector requires an immutable CLAP commit")
    algorithm = payload.get("algorithm", {})
    if payload["experiment"] == CONSTRAINED_EXPERIMENT_V2:
        if algorithm.get("name") != "deterministic_global_assignment_v2":
            raise LTSNContractError("v2 constrained selector requires global assignment")
        if int(algorithm.get("candidate_count", 0)) < 2:
            raise LTSNContractError("v2 constrained selector candidate_count is invalid")
        effect_threshold = float(algorithm.get("effect_threshold", math.nan))
        if not 0.0 < effect_threshold <= 1.0:
            raise LTSNContractError("v2 constrained selector effect threshold is invalid")
        if int(algorithm.get("maximum_search_nodes", 0)) < 1:
            raise LTSNContractError("v2 constrained selector search limit is invalid")
        if tuple(algorithm.get("objective", ())) != GLOBAL_OBJECTIVE:
            raise LTSNContractError("v2 constrained selector objective changed")
        if algorithm.get("incomplete_search_policy") != "fail_closed":
            raise LTSNContractError("v2 constrained selector must fail closed")
        if constraints.get("allow_topology_neutral_support_moves") is not False:
            raise LTSNContractError("v2 constrained selector forbids topology-neutral moves")
        if int(algorithm["candidate_count"]) != int(
            payload.get("confirmation", {}).get("candidates_per_prompt", 0)
        ):
            raise LTSNContractError("v2 selector candidate counts disagree")
    return payload


def _validate_selector_experiment_contract(
    selector: dict[str, Any], config: ExperimentConfig
) -> None:
    expected_candidates = int(selector["confirmation"]["candidates_per_prompt"])
    if config.candidate_count != expected_candidates:
        raise LTSNContractError(
            "experiment candidate_count does not match the frozen selector"
        )
    expected_duration = float(selector["confirmation"]["duration_seconds"])
    if config.duration_seconds != expected_duration:
        raise LTSNContractError("experiment duration does not match the frozen selector")
    if selector["experiment"] == CONSTRAINED_EXPERIMENT_V2 and float(
        selector["algorithm"]["effect_threshold"]
    ) != float(config.scoring.success_min_median_improvement):
        raise LTSNContractError(
            "selector effect threshold does not match the frozen scoring gate"
        )


def _resolve_audio(run_root: Path, relative: str) -> Path:
    path = (run_root / relative).resolve()
    if not path.is_relative_to(run_root.resolve()):
        raise LTSNContractError("candidate audio path escapes the reranking run")
    return path


def build_candidate_semantics(
    *,
    project_root: Path,
    config: ExperimentConfig,
    prompt_manifest: Path,
    selector_config_path: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Embed every candidate and prompt under the frozen CLAP contract."""

    selector = load_selector_config(selector_config_path)
    _validate_selector_experiment_contract(selector, config)
    run_root = experiment_root(project_root, config)
    records = read_candidate_manifest(run_root / "manifests" / "candidates.csv")
    scores = {row["candidate_id"]: row for row in _read_csv(run_root / "scores.csv")}
    prompts = {row["prompt_id"]: row for row in _read_csv(prompt_manifest)}
    if set(scores) != {record.candidate_id for record in records}:
        raise LTSNContractError("exact scores do not cover the candidate manifest")
    if {record.prompt_id for record in records} != set(prompts):
        raise LTSNContractError("semantic prompt manifest does not match the reranking run")
    audio_paths = {}
    audio_hashes = {}
    for record in records:
        path = _resolve_audio(run_root, record.audio_relative_path)
        if not path.is_file() or sha256_file(path) != record.audio_sha256:
            raise LTSNContractError(f"candidate audio is missing or changed: {record.candidate_id}")
        audio_paths[record.candidate_id] = path
        audio_hashes[record.candidate_id] = record.audio_sha256
    embedding = selector["embedding"]
    backend = TransformersClapBackend(
        embedding["model_id"], embedding["model_revision"], device
    )
    candidate_embeddings, unique_audio = embed_candidates_by_audio_sha256(
        audio_paths,
        audio_hashes,
        backend,
        segment_seconds=float(embedding["segment_seconds"]),
        batch_size=batch_size,
    )
    prompt_ids = sorted(prompts)
    text_batches = []
    for start in range(0, len(prompt_ids), batch_size):
        batch_ids = prompt_ids[start : start + batch_size]
        captions = [prompts[prompt_id]["caption"] for prompt_id in batch_ids]
        text_batches.append(backend.embed_text(captions))
    text_embeddings = normalize_embeddings(np.concatenate(text_batches, axis=0))
    text_by_prompt = dict(zip(prompt_ids, text_embeddings, strict=True))
    ordered = sorted(records, key=lambda record: (record.prompt_id, record.candidate_index))
    matrix = normalize_embeddings(
        np.stack([candidate_embeddings[record.candidate_id] for record in ordered])
    )
    rows = []
    for record, values in zip(ordered, matrix, strict=True):
        score = scores[record.candidate_id]
        rows.append(
            {
                "candidate_id": record.candidate_id,
                "prompt_id": record.prompt_id,
                "candidate_index": record.candidate_index,
                "audio_sha256": record.audio_sha256,
                "focus_band_loss": score["focus_band_loss"],
                "technical_quality_eligible": score["technical_quality_eligible"],
                "prompt_alignment": format(
                    float(values @ text_by_prompt[record.prompt_id]), ".17g"
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "candidate_semantics.csv"
    embeddings_path = output_dir / "candidate_embeddings.npz"
    write_csv_atomic(metrics_path, rows)
    temporary = embeddings_path.with_suffix(".part.npz")
    np.savez_compressed(
        temporary,
        candidate_ids=np.asarray([record.candidate_id for record in ordered]),
        embeddings=matrix,
    )
    os.replace(temporary, embeddings_path)
    audit = {
        "schema_version": 1,
        "experiment": selector["experiment"],
        "run_id": config.run_id,
        "selector_config_path": os.path.relpath(selector_config_path, run_root).replace(
            "\\", "/"
        ),
        "selector_config_sha256": sha256_file(selector_config_path),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
        "candidate_manifest_sha256": sha256_file(run_root / "manifests" / "candidates.csv"),
        "descriptor_table_sha256": sha256_file(run_root / "descriptors_18d.csv"),
        "model_id": embedding["model_id"],
        "requested_revision": embedding["model_revision"],
        "resolved_revision": backend.model_commit,
        "candidate_count": len(rows),
        "unique_audio_sha256": unique_audio,
        "candidate_semantics_sha256": sha256_file(metrics_path),
        "candidate_embeddings_sha256": sha256_file(embeddings_path),
    }
    write_json_atomic(output_dir / "semantic_audit.json", audit)
    return audit


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        candidate_ids = archive["candidate_ids"].astype(str).tolist()
        embeddings = normalize_embeddings(archive["embeddings"])
    if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_ids) != len(embeddings):
        raise LTSNContractError("candidate embedding artifact has invalid IDs")
    return dict(zip(candidate_ids, embeddings, strict=True))


def _select_greedy_candidates(
    *,
    semantic_dir: Path,
    selector_config_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Greedily improve exact loss while preserving every frozen semantic guard."""

    selector = load_selector_config(selector_config_path)
    rows = _read_csv(semantic_dir / "candidate_semantics.csv")
    embeddings = _load_embeddings(semantic_dir / "candidate_embeddings.npz")
    pools: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        pools.setdefault(row["prompt_id"], []).append(row)
    candidate_count = int(selector["confirmation"]["candidates_per_prompt"])
    if any(len(pool) != candidate_count for pool in pools.values()):
        raise LTSNContractError(
            f"constrained selection requires complete best-of-{candidate_count} pools"
        )
    baseline = {
        prompt_id: next(row for row in pool if int(row["candidate_index"]) == 0)
        for prompt_id, pool in pools.items()
    }
    assignment = {prompt_id: row["candidate_id"] for prompt_id, row in baseline.items()}
    prompt_ids = sorted(assignment)
    baseline_matrix = np.stack([embeddings[assignment[prompt_id]] for prompt_id in prompt_ids])
    baseline_diversity = nearest_neighbor_diversity(baseline_matrix)
    baseline_diversity_by_prompt = dict(zip(prompt_ids, baseline_diversity, strict=True))
    prompt_margin = float(selector["constraints"]["prompt_margin"])
    diversity_margin = float(selector["constraints"]["diversity_margin"])
    proposals = []
    for prompt_id, pool in pools.items():
        base = baseline[prompt_id]
        base_loss = float(base["focus_band_loss"])
        base_prompt = float(base["prompt_alignment"])
        if base_loss <= 0.0:
            continue
        for row in pool:
            loss = float(row["focus_band_loss"])
            prompt = float(row["prompt_alignment"])
            if (
                row["candidate_id"] != base["candidate_id"]
                and row["technical_quality_eligible"] == "1"
                and loss < base_loss
                and prompt - base_prompt >= -prompt_margin
            ):
                proposals.append(
                    (
                        -(base_loss - loss) / base_loss,
                        -prompt,
                        prompt_id,
                        row["candidate_id"],
                        row,
                    )
                )
    for _, _, prompt_id, candidate_id, row in sorted(proposals):
        current = next(
            item
            for item in pools[prompt_id]
            if item["candidate_id"] == assignment[prompt_id]
        )
        if float(row["focus_band_loss"]) >= float(current["focus_band_loss"]):
            continue
        trial = dict(assignment)
        trial[prompt_id] = candidate_id
        trial_matrix = np.stack([embeddings[trial[item]] for item in prompt_ids])
        trial_diversity = nearest_neighbor_diversity(trial_matrix)
        if all(
            value - baseline_diversity_by_prompt[item] >= -diversity_margin
            for item, value in zip(prompt_ids, trial_diversity, strict=True)
        ):
            assignment = trial
    selected_matrix = np.stack([embeddings[assignment[prompt_id]] for prompt_id in prompt_ids])
    selected_diversity = nearest_neighbor_diversity(selected_matrix)
    selection_rows = []
    for prompt_id, diversity in zip(prompt_ids, selected_diversity, strict=True):
        base = baseline[prompt_id]
        chosen = next(
            row for row in pools[prompt_id] if row["candidate_id"] == assignment[prompt_id]
        )
        selection_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_candidate_id": base["candidate_id"],
                "selected_candidate_id": chosen["candidate_id"],
                "changed": int(base["candidate_id"] != chosen["candidate_id"]),
                "baseline_focus_band_loss": base["focus_band_loss"],
                "selected_focus_band_loss": chosen["focus_band_loss"],
                "prompt_difference": float(chosen["prompt_alignment"])
                - float(base["prompt_alignment"]),
                "diversity_difference": float(diversity) - baseline_diversity_by_prompt[prompt_id],
            }
        )
    changed = [row for row in selection_rows if row["changed"]]
    report = {
        "prompt_pools": len(prompt_ids),
        "changed_pools": len(changed),
        "all_prompt_constraints_passed": all(
            row["prompt_difference"] >= -prompt_margin for row in selection_rows
        ),
        "all_diversity_constraints_passed": all(
            row["diversity_difference"] >= -diversity_margin for row in selection_rows
        ),
        "minimum_prompt_difference": min(row["prompt_difference"] for row in selection_rows),
        "minimum_diversity_difference": min(
            row["diversity_difference"] for row in selection_rows
        ),
    }
    return assignment, selection_rows, report


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _global_assignment_candidates(
    *,
    rows: list[dict[str, str]],
    embeddings: dict[str, np.ndarray],
    selector: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        pools.setdefault(row["prompt_id"], []).append(row)
    candidate_count = int(selector["algorithm"]["candidate_count"])
    if any(len(pool) != candidate_count for pool in pools.values()):
        raise LTSNContractError(
            f"v2 constrained selection requires complete best-of-{candidate_count} pools"
        )
    try:
        baseline = {
            prompt_id: next(row for row in pool if int(row["candidate_index"]) == 0)
            for prompt_id, pool in pools.items()
        }
    except StopIteration as error:
        raise LTSNContractError("constrained pool has no candidate_index=0 baseline") from error
    prompt_ids = sorted(baseline)
    baseline_assignment = {
        prompt_id: baseline[prompt_id]["candidate_id"] for prompt_id in prompt_ids
    }
    if set(embeddings) != {
        row["candidate_id"] for pool in pools.values() for row in pool
    }:
        raise LTSNContractError("candidate embeddings do not match semantic candidates")
    baseline_matrix = np.stack(
        [embeddings[baseline_assignment[prompt_id]] for prompt_id in prompt_ids]
    )
    baseline_diversity = nearest_neighbor_diversity(baseline_matrix)
    baseline_diversity_by_prompt = dict(zip(prompt_ids, baseline_diversity, strict=True))
    prompt_margin = float(selector["constraints"]["prompt_margin"])
    diversity_margin = float(selector["constraints"]["diversity_margin"])
    effect_threshold = float(selector["algorithm"]["effect_threshold"])
    maximum_nodes = int(selector["algorithm"]["maximum_search_nodes"])
    positive_prompt_ids = [
        prompt_id
        for prompt_id in prompt_ids
        if float(baseline[prompt_id]["focus_band_loss"]) > 0.0
    ]
    fixed_prompt_ids = [
        prompt_id for prompt_id in prompt_ids if prompt_id not in positive_prompt_ids
    ]

    def compatible(
        left_prompt: str, left_candidate: str, right_prompt: str, right_candidate: str
    ) -> bool:
        similarity = float(embeddings[left_candidate] @ embeddings[right_candidate])
        left_limit = 1.0 - baseline_diversity_by_prompt[left_prompt] + diversity_margin
        right_limit = 1.0 - baseline_diversity_by_prompt[right_prompt] + diversity_margin
        return similarity <= min(left_limit, right_limit)

    options: dict[str, list[_GlobalOption]] = {}
    funnel_rows: list[dict[str, Any]] = []
    for prompt_id in positive_prompt_ids:
        base = baseline[prompt_id]
        base_loss = float(base["focus_band_loss"])
        base_prompt = float(base["prompt_alignment"])
        baseline_option = _GlobalOption(
            candidate_id=base["candidate_id"],
            loss=base_loss,
            prompt_alignment=base_prompt,
            relative_improvement=0.0,
            effectful=0,
            target_hit=0,
            changed=0,
        )
        prompt_options = [baseline_option]
        technical = topology = prompt_eligible = static_diversity = 0
        for row in pools[prompt_id]:
            if row["candidate_id"] == base["candidate_id"]:
                continue
            if row["technical_quality_eligible"] != "1":
                continue
            technical += 1
            loss = float(row["focus_band_loss"])
            if loss >= base_loss:
                continue
            topology += 1
            prompt_alignment = float(row["prompt_alignment"])
            if prompt_alignment - base_prompt < -prompt_margin:
                continue
            prompt_eligible += 1
            candidate_id = row["candidate_id"]
            if not all(
                compatible(
                    prompt_id,
                    candidate_id,
                    fixed_prompt,
                    baseline_assignment[fixed_prompt],
                )
                for fixed_prompt in fixed_prompt_ids
            ):
                continue
            static_diversity += 1
            improvement = (base_loss - loss) / base_loss
            prompt_options.append(
                _GlobalOption(
                    candidate_id=candidate_id,
                    loss=loss,
                    prompt_alignment=prompt_alignment,
                    relative_improvement=improvement,
                    effectful=int(improvement >= effect_threshold),
                    target_hit=int(loss <= 0.0),
                    changed=1,
                )
            )
        options[prompt_id] = prompt_options
        funnel_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_focus_band_loss": base_loss,
                "alternative_candidates": len(pools[prompt_id]) - 1,
                "technical_quality_eligible": technical,
                "strict_topology_improvements": topology,
                "prompt_eligible_improvements": prompt_eligible,
                "fixed_cohort_diversity_eligible": static_diversity,
                "effectful_fixed_cohort_eligible": sum(
                    option.effectful for option in prompt_options
                ),
            }
        )

    selected_options: dict[str, _GlobalOption] = {}
    best_options: dict[str, _GlobalOption] | None = None
    best_score: tuple[int, float, int, int, float] | None = None
    best_signature: tuple[str, ...] | None = None
    search_nodes = 0

    def viable_options(prompt_id: str) -> list[_GlobalOption]:
        return [
            option
            for option in options[prompt_id]
            if all(
                compatible(prompt_id, option.candidate_id, other_prompt, other.candidate_id)
                for other_prompt, other in selected_options.items()
            )
        ]

    def option_order(option: _GlobalOption) -> tuple[Any, ...]:
        return (
            -option.effectful,
            -option.relative_improvement,
            -option.target_hit,
            -option.changed,
            -option.prompt_alignment,
            option.candidate_id,
        )

    def score_of(chosen: dict[str, _GlobalOption]) -> tuple[int, float, int, int, float]:
        improvements = [
            chosen[prompt_id].relative_improvement
            for prompt_id in prompt_ids
            if prompt_id in chosen
        ]
        return (
            sum(option.effectful for option in chosen.values()),
            _median(improvements),
            sum(option.target_hit for option in chosen.values()),
            sum(option.changed for option in chosen.values()),
            sum(option.relative_improvement for option in chosen.values()),
        )

    def search(unassigned: tuple[str, ...]) -> None:
        nonlocal best_options, best_score, best_signature, search_nodes
        search_nodes += 1
        if search_nodes > maximum_nodes:
            raise LTSNContractError(
                f"global constrained search exceeded {maximum_nodes} nodes"
            )
        if not unassigned:
            score = score_of(selected_options)
            signature = tuple(
                selected_options[prompt_id].candidate_id
                for prompt_id in positive_prompt_ids
            )
            if best_score is None or score > best_score or (
                score == best_score and (best_signature is None or signature < best_signature)
            ):
                best_score = score
                best_signature = signature
                best_options = dict(selected_options)
            return
        viable_by_prompt = {
            prompt_id: viable_options(prompt_id) for prompt_id in unassigned
        }
        if any(not domain for domain in viable_by_prompt.values()):
            return
        optimistic_improvements = [
            option.relative_improvement for option in selected_options.values()
        ] + [
            max(option.relative_improvement for option in domain)
            for domain in viable_by_prompt.values()
        ]
        optimistic_score = (
            sum(option.effectful for option in selected_options.values())
            + sum(
                max(option.effectful for option in domain)
                for domain in viable_by_prompt.values()
            ),
            _median(optimistic_improvements),
            sum(option.target_hit for option in selected_options.values())
            + sum(
                max(option.target_hit for option in domain)
                for domain in viable_by_prompt.values()
            ),
            sum(option.changed for option in selected_options.values())
            + sum(max(option.changed for option in domain) for domain in viable_by_prompt.values()),
            sum(option.relative_improvement for option in selected_options.values())
            + sum(
                max(option.relative_improvement for option in domain)
                for domain in viable_by_prompt.values()
            ),
        )
        if best_score is not None and optimistic_score < best_score:
            return
        prompt_id = min(unassigned, key=lambda item: (len(viable_by_prompt[item]), item))
        remaining = tuple(item for item in unassigned if item != prompt_id)
        for option in sorted(viable_by_prompt[prompt_id], key=option_order):
            selected_options[prompt_id] = option
            search(remaining)
            del selected_options[prompt_id]

    search(tuple(positive_prompt_ids))
    if best_options is None or best_score is None:
        raise LTSNContractError("global constrained assignment has no feasible baseline fallback")
    assignment = dict(baseline_assignment)
    assignment.update(
        {prompt_id: option.candidate_id for prompt_id, option in best_options.items()}
    )
    selected_matrix = np.stack([embeddings[assignment[prompt_id]] for prompt_id in prompt_ids])
    selected_diversity = nearest_neighbor_diversity(selected_matrix)
    selection_rows = []
    for prompt_id, diversity in zip(prompt_ids, selected_diversity, strict=True):
        base = baseline[prompt_id]
        chosen = next(
            row for row in pools[prompt_id] if row["candidate_id"] == assignment[prompt_id]
        )
        selection_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_candidate_id": base["candidate_id"],
                "selected_candidate_id": chosen["candidate_id"],
                "changed": int(base["candidate_id"] != chosen["candidate_id"]),
                "baseline_focus_band_loss": base["focus_band_loss"],
                "selected_focus_band_loss": chosen["focus_band_loss"],
                "prompt_difference": float(chosen["prompt_alignment"])
                - float(base["prompt_alignment"]),
                "diversity_difference": float(diversity)
                - baseline_diversity_by_prompt[prompt_id],
            }
        )
    changed = [row for row in selection_rows if row["changed"]]
    required_effectful = len(positive_prompt_ids) // 2 + 1
    feasibility_audit = {
        "schema_version": 1,
        "experiment": selector["experiment"],
        "algorithm": selector["algorithm"]["name"],
        "candidate_count_per_prompt": candidate_count,
        "positive_baseline_loss_pools": len(positive_prompt_ids),
        "effect_threshold": effect_threshold,
        "conservative_effectful_pools_required_for_median": required_effectful,
        "maximum_feasible_effectful_pools": best_score[0],
        "maximum_feasible_median_under_lexicographic_objective": best_score[1],
        "candidate_pool_supports_effectful_majority": best_score[0] >= required_effectful,
        "search_complete": True,
        "search_nodes": search_nodes,
        "maximum_search_nodes": maximum_nodes,
        "objective": selector["algorithm"]["objective"],
        "pool_funnel": funnel_rows,
    }
    report = {
        "algorithm": selector["algorithm"]["name"],
        "prompt_pools": len(prompt_ids),
        "changed_pools": len(changed),
        "effectful_changed_pools": best_score[0],
        "search_complete": True,
        "all_prompt_constraints_passed": all(
            row["prompt_difference"] >= -prompt_margin for row in selection_rows
        ),
        "all_diversity_constraints_passed": all(
            row["diversity_difference"] >= -diversity_margin for row in selection_rows
        ),
        "minimum_prompt_difference": min(row["prompt_difference"] for row in selection_rows),
        "minimum_diversity_difference": min(
            row["diversity_difference"] for row in selection_rows
        ),
        "feasibility_audit": feasibility_audit,
    }
    return assignment, selection_rows, report


def select_constrained_candidates(
    *,
    semantic_dir: Path,
    selector_config_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Apply the prospectively frozen greedy-v1 or global-v2 selector."""

    selector = load_selector_config(selector_config_path)
    if selector["experiment"] == CONSTRAINED_EXPERIMENT:
        return _select_greedy_candidates(
            semantic_dir=semantic_dir, selector_config_path=selector_config_path
        )
    return _global_assignment_candidates(
        rows=_read_csv(semantic_dir / "candidate_semantics.csv"),
        embeddings=_load_embeddings(semantic_dir / "candidate_embeddings.npz"),
        selector=selector,
    )


def apply_constrained_selection(
    *,
    project_root: Path,
    config: ExperimentConfig,
    selector_config_path: Path,
    semantic_dir: Path,
    output_dir: Path,
    frozen_selector_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the frozen selector and rewrite standard reranking selection artifacts."""

    run_root = experiment_root(project_root, config)
    selector = load_selector_config(selector_config_path)
    _validate_selector_experiment_contract(selector, config)
    if frozen_selector_path is not None:
        frozen = json.loads(frozen_selector_path.read_text(encoding="utf-8"))
        if (
            frozen.get("status") != "frozen_after_calibration_before_confirmation"
            or frozen.get("experiment") != selector["experiment"]
            or frozen.get("selector_config_sha256") != sha256_file(selector_config_path)
        ):
            raise LTSNContractError("confirmation selector was not frozen on calibration")
    assignment, selection_rows, diagnostic = select_constrained_candidates(
        semantic_dir=semantic_dir,
        selector_config_path=selector_config_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "constrained_selection.csv"
    write_csv_atomic(selection_path, selection_rows)
    feasibility_path = output_dir / "feasibility_audit.json"
    feasibility_audit = diagnostic.get("feasibility_audit")
    if feasibility_audit is not None:
        write_json_atomic(feasibility_path, feasibility_audit)
    contract = {
        "schema_version": 1,
        "experiment": selector["experiment"],
        "run_id": config.run_id,
        "selector_config_path": os.path.relpath(selector_config_path, run_root).replace(
            "\\", "/"
        ),
        "selector_config_sha256": sha256_file(selector_config_path),
        "frozen_selector_path": (
            os.path.relpath(frozen_selector_path, run_root).replace("\\", "/")
            if frozen_selector_path is not None
            else None
        ),
        "frozen_selector_sha256": (
            sha256_file(frozen_selector_path) if frozen_selector_path is not None else None
        ),
        "candidate_manifest_sha256": sha256_file(run_root / "manifests" / "candidates.csv"),
        "descriptor_table_sha256": sha256_file(run_root / "descriptors_18d.csv"),
        "semantic_audit_sha256": sha256_file(semantic_dir / "semantic_audit.json"),
        "selection_table_sha256": sha256_file(selection_path),
        "feasibility_audit_sha256": (
            sha256_file(feasibility_path) if feasibility_audit is not None else None
        ),
        "selected_candidates": assignment,
        "diagnostic": diagnostic,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    contract_path = output_dir / "selection_contract.json"
    write_json_atomic(contract_path, contract)
    with (run_root / "descriptors_18d.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        descriptors = list(csv.DictReader(handle))
    records = read_candidate_manifest(run_root / "manifests" / "candidates.csv")
    summary = rank_and_summarize(
        project_root,
        config,
        records,
        descriptors,
        selection_override=assignment,
        selection_metadata={
            "name": selector["experiment"],
            "selection_contract_path": os.path.relpath(contract_path, run_root).replace("\\", "/"),
            "selection_contract_sha256": sha256_file(contract_path),
            "selector_config_sha256": sha256_file(selector_config_path),
        },
    )
    report = {
        "schema_version": 1,
        "experiment": selector["experiment"],
        "run_id": config.run_id,
        **diagnostic,
        "median_loss_improvement_fraction": summary["median_loss_improvement_fraction"],
        "target_band_hit_rate_improved": summary["target_band_hit_rate_improved"],
        "topology_passed": summary["topology_passed"],
        "calibration_supported": bool(
            frozen_selector_path is None
            and diagnostic["changed_pools"] > 0
            and diagnostic["all_prompt_constraints_passed"]
            and diagnostic["all_diversity_constraints_passed"]
            and diagnostic.get("search_complete", True)
            and summary["median_loss_improvement_fraction"]
            >= config.scoring.success_min_median_improvement
            and summary["target_band_hit_rate_improved"]
            and summary["all_selected_technical_quality_eligible"]
        ),
        "confirmation_ready_for_noninferiority": bool(
            frozen_selector_path is not None and summary["topology_passed"]
        ),
        "selection_contract_sha256": sha256_file(contract_path),
        "summary_sha256": sha256_file(run_root / "summary.json"),
    }
    write_json_atomic(output_dir / "constrained_report.json", report)
    return report


def freeze_calibrated_selector(
    *,
    selector_config_path: Path,
    calibration_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze a supported calibration result before any confirmation generation."""

    report_path = calibration_dir / "constrained_report.json"
    contract_path = calibration_dir / "selection_contract.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    selector = load_selector_config(selector_config_path)
    if (
        report.get("calibration_supported") is not True
        or report.get("selection_contract_sha256") != sha256_file(contract_path)
        or report.get("experiment") != selector["experiment"]
        or contract.get("experiment") != selector["experiment"]
        or contract.get("selector_config_sha256") != sha256_file(selector_config_path)
    ):
        raise LTSNContractError("calibration does not support freezing this selector")
    payload = {
        "schema_version": 1,
        "experiment": selector["experiment"],
        "status": "frozen_after_calibration_before_confirmation",
        "selector_config_sha256": sha256_file(selector_config_path),
        "calibration_report_sha256": sha256_file(report_path),
        "calibration_selection_contract_sha256": sha256_file(contract_path),
        "constraints": selector["constraints"],
        "confirmation_authorized": True,
    }
    write_json_atomic(output_path, payload)
    return payload


def load_selection_contract(
    run_root: Path, path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate a run-bound constrained selection for gate re-evaluation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") not in CONSTRAINED_EXPERIMENTS
        or payload.get("run_id") != run_root.name
    ):
        raise LTSNContractError("selection contract has the wrong experiment")
    if payload.get("candidate_manifest_sha256") != sha256_file(
        run_root / "manifests" / "candidates.csv"
    ) or payload.get("descriptor_table_sha256") != sha256_file(
        run_root / "descriptors_18d.csv"
    ):
        raise LTSNContractError("selection contract input binding changed")
    expected_contract_hash = payload.get("contract_sha256")
    canonical_payload = {key: value for key, value in payload.items() if key != "contract_sha256"}
    if expected_contract_hash != canonical_json_sha256(canonical_payload):
        raise LTSNContractError("selection contract canonical hash changed")
    bound_paths = {
        "selector_config_sha256": (run_root / payload["selector_config_path"]).resolve(),
        "semantic_audit_sha256": path.parent / "semantic_audit.json",
        "selection_table_sha256": path.parent / "constrained_selection.csv",
    }
    if payload.get("feasibility_audit_sha256"):
        bound_paths["feasibility_audit_sha256"] = path.parent / "feasibility_audit.json"
    frozen_relative = payload.get("frozen_selector_path")
    if not frozen_relative or not payload.get("frozen_selector_sha256"):
        raise LTSNContractError("formal selection contract has no frozen calibration selector")
    bound_paths["frozen_selector_sha256"] = (run_root / frozen_relative).resolve()
    for hash_name, artifact_path in bound_paths.items():
        if not artifact_path.is_file() or sha256_file(artifact_path) != payload.get(hash_name):
            raise LTSNContractError(f"selection contract artifact binding changed: {hash_name}")
    selected = payload.get("selected_candidates")
    if not isinstance(selected, dict) or not selected:
        raise LTSNContractError("selection contract has no selected candidates")
    metadata = {
        "name": payload["experiment"],
        "selection_contract_path": os.path.relpath(path, run_root).replace("\\", "/"),
        "selection_contract_sha256": sha256_file(path),
        "selector_config_sha256": payload["selector_config_sha256"],
    }
    return {str(key): str(value) for key, value in selected.items()}, metadata
