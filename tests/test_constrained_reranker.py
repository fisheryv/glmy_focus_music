from __future__ import annotations

from pathlib import Path

import numpy as np

from generation.constrained_reranker import (
    build_constrained_prompt_manifests,
    select_constrained_candidates,
)
from generation.ltsn_pipeline import write_csv_atomic

ROOT = Path(__file__).resolve().parents[1]


def _prompt_rows() -> list[dict[str, object]]:
    rows = []
    for split, prefix in (("calibration", "cal"), ("qualification", "qual")):
        for family in range(4):
            for variant in range(1, 17):
                rows.append(
                    {
                        "prompt_id": f"{prefix}_{family}__v{variant:02d}",
                        "caption": f"{split} family {family} variant {variant}",
                        "split": split,
                        "bpm": 80,
                        "keyscale": "",
                        "timesignature": 4,
                    }
                )
    return rows


def test_constrained_prompt_manifests_use_new_confirmation_families(tmp_path: Path) -> None:
    source = tmp_path / "prompts.csv"
    write_csv_atomic(source, _prompt_rows())
    confirmation_source = tmp_path / "confirmation.csv"
    write_csv_atomic(
        confirmation_source,
        [
            {
                "prompt_id": f"new_family_{index:02d}",
                "caption": f"new confirmation family {index}",
                "split": "confirmation",
                "bpm": 80,
                "keyscale": "",
                "timesignature": 4,
            }
            for index in range(32)
        ],
    )

    payload = build_constrained_prompt_manifests(
        source, confirmation_source, tmp_path / "out"
    )

    assert payload["family_disjoint"] is True
    assert payload["outputs"]["calibration"]["prompts"] == 64
    assert payload["outputs"]["confirmation"]["prompts"] == 32
    assert set(payload["outputs"]["calibration"]["families"]).isdisjoint(
        payload["outputs"]["confirmation"]["families"]
    )
    assert len(payload["outputs"]["confirmation"]["families"]) == 32


def test_selector_accepts_only_topology_improvements_preserving_both_guards(
    tmp_path: Path,
) -> None:
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    rows = []
    candidate_ids = []
    embeddings = []
    for prompt_index, baseline_embedding in enumerate(
        (np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))
    ):
        prompt_id = f"p{prompt_index}"
        for candidate_index in range(8):
            candidate_id = f"{prompt_id}__c{candidate_index:02d}"
            candidate_ids.append(candidate_id)
            embedding = baseline_embedding.copy()
            loss = 1.0
            prompt = 0.8
            eligible = 1
            if prompt_index == 0 and candidate_index == 1:
                embedding = np.asarray([-1.0, 0.0])
                loss = 0.0
                prompt = 0.9
            elif prompt_index == 1 and candidate_index == 1:
                loss = 0.0
                prompt = 0.7
            elif candidate_index > 1:
                eligible = 0
            embeddings.append(embedding)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "prompt_id": prompt_id,
                    "candidate_index": candidate_index,
                    "audio_sha256": str(prompt_index) * 64,
                    "focus_band_loss": loss,
                    "technical_quality_eligible": eligible,
                    "prompt_alignment": prompt,
                }
            )
    write_csv_atomic(semantic_dir / "candidate_semantics.csv", rows)
    np.savez_compressed(
        semantic_dir / "candidate_embeddings.npz",
        candidate_ids=np.asarray(candidate_ids),
        embeddings=np.asarray(embeddings),
    )

    selected, selection_rows, report = select_constrained_candidates(
        semantic_dir=semantic_dir,
        selector_config_path=ROOT / "configs" / "constrained_reranker_v1.json",
    )

    assert selected["p0"] == "p0__c01"
    assert selected["p1"] == "p1__c00"
    assert report["changed_pools"] == 1
    assert report["all_prompt_constraints_passed"] is True
    assert report["all_diversity_constraints_passed"] is True
    assert all(row["diversity_difference"] >= 0.0 for row in selection_rows)


def test_v2_global_assignment_can_accept_jointly_feasible_substitutions(
    tmp_path: Path,
) -> None:
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    baselines = {
        "p0": np.asarray([1.0, 0.0, 0.0]),
        "p1": np.asarray([0.0, 1.0, 0.0]),
        "p2": np.asarray([0.0, 0.0, 1.0]),
    }
    joint_candidates = {
        "p0": np.asarray([-2.0, 1.0, -1.0]),
        "p1": np.asarray([1.0, -2.0, -1.0]),
    }
    rows = []
    candidate_ids = []
    embeddings = []
    for prompt_id, baseline_embedding in baselines.items():
        for candidate_index in range(16):
            candidate_id = f"{prompt_id}__c{candidate_index:02d}"
            candidate_ids.append(candidate_id)
            embedding = baseline_embedding.copy()
            loss = 1.0
            prompt = 0.8
            eligible = int(candidate_index == 0)
            if candidate_index == 1 and prompt_id in joint_candidates:
                embedding = joint_candidates[prompt_id]
                loss = 0.0
                prompt = 0.9
                eligible = 1
            embeddings.append(embedding)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "prompt_id": prompt_id,
                    "candidate_index": candidate_index,
                    "audio_sha256": prompt_id * 32,
                    "focus_band_loss": loss,
                    "technical_quality_eligible": eligible,
                    "prompt_alignment": prompt,
                }
            )
    write_csv_atomic(semantic_dir / "candidate_semantics.csv", rows)
    np.savez_compressed(
        semantic_dir / "candidate_embeddings.npz",
        candidate_ids=np.asarray(candidate_ids),
        embeddings=np.asarray(embeddings),
    )

    selected, selection_rows, report = select_constrained_candidates(
        semantic_dir=semantic_dir,
        selector_config_path=ROOT / "configs" / "constrained_reranker_v2.json",
    )

    assert selected["p0"] == "p0__c01"
    assert selected["p1"] == "p1__c01"
    assert selected["p2"] == "p2__c00"
    assert report["changed_pools"] == 2
    assert report["effectful_changed_pools"] == 2
    assert report["search_complete"] is True
    assert report["feasibility_audit"]["maximum_feasible_effectful_pools"] == 2
    assert report["feasibility_audit"]["candidate_pool_supports_effectful_majority"] is True
    assert all(row["diversity_difference"] >= 0.0 for row in selection_rows)
