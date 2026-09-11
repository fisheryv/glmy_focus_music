from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generation.path_homology_exact_scorer import ExactPathHomologyScorer
from generation.tac_target import fit_target_payload, sha256_file
from topology.metrics import TOPOLOGY_METRICS

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
PHASE_VIEWS = ("path_acoustic_phase", "path_chroma_phase")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mapping_sha256(values: dict[str, str]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_target(*, root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    target_config = config["target"]
    geometry = config["geometry"]
    bands = config["reference_bands"]
    if (
        int(target_config["dimensions"]) != 18
        or list(geometry["pitch_active_indices"]) != list(range(3, 16))
        or list(geometry["distance_weights"]) != [0.5, 0.25, 0.25]
        or float(geometry["pitch_shrinkage"]) != 0.1
        or list(bands["quantiles"]) != [0.5, 0.9, 0.95]
    ):
        raise RuntimeError("TAC Stage-0 frozen geometry or reference bands changed")

    fingerprint_path = root / target_config["fingerprint_path"]
    scores_path = root / target_config["scores_path"]
    pitch_path = root / target_config["pitch_path"]
    phase_path = root / target_config["phase_path"]
    scorer = ExactPathHomologyScorer.from_json(fingerprint_path)

    pitch = pd.read_csv(pitch_path).set_index(IDENTITY).sort_index()
    phase = pd.read_csv(phase_path)
    phase = phase.pivot(index=IDENTITY, columns="representation", values="loop_score").reindex(
        pitch.index
    )
    published = pd.read_csv(scores_path).set_index(IDENTITY).sort_index()
    if not pitch.index.equals(phase.index) or not pitch.index.equals(published.index):
        raise RuntimeError("Pitch, phase and published score identities differ")
    score = scorer.score(
        pitch.loc[:, TOPOLOGY_METRICS].to_numpy(float),
        phase.loc[:, [PHASE_VIEWS[0]]].to_numpy(float),
        phase.loc[:, [PHASE_VIEWS[1]]].to_numpy(float),
    )
    feature_order = list(scorer.contract.feature_order)
    np.testing.assert_allclose(
        score.coordinates,
        published.loc[:, feature_order].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
        err_msg="exact scorer no longer reproduces the published 18-D coordinates",
    )
    identity = pitch.index.to_frame(index=False)
    reference = (
        (identity["split"].astype(str).to_numpy() == target_config["reference_split"])
        & (identity["group"].astype(str).to_numpy() == target_config["reference_group"])
        & np.isclose(
            identity["scale_seconds"].to_numpy(float),
            float(target_config["reference_scale_seconds"]),
        )
    )
    if int(reference.sum()) != 195:
        raise RuntimeError(f"expected 195 Focus references, observed {int(reference.sum())}")

    module_path = root / "src" / "generation" / "tac_target.py"
    script_path = Path(__file__).resolve()
    source_paths = [
        fingerprint_path,
        scores_path,
        pitch_path,
        phase_path,
        config_path,
        module_path,
        script_path,
    ]
    source_sha256 = {_relative(path, root): sha256_file(path) for path in source_paths}
    payload = fit_target_payload(
        score.coordinates[reference],
        active_pitch_indices=tuple(int(value) for value in geometry["pitch_active_indices"]),
        acoustic_index=int(geometry["acoustic_phase_index"]),
        chroma_index=int(geometry["chroma_phase_index"]),
        distance_weights=tuple(float(value) for value in geometry["distance_weights"]),
        pitch_shrinkage=float(geometry["pitch_shrinkage"]),
        reference_quantiles=tuple(float(value) for value in bands["quantiles"]),
        metadata={
            "target_id": target_config["target_id"],
            "spec_revision": target_config["spec_revision"],
            "fingerprint_id": scorer.contract.fingerprint_id,
            "fingerprint_sha256": sha256_file(fingerprint_path),
            "feature_order": feature_order,
            "reference_split": target_config["reference_split"],
            "reference_group": target_config["reference_group"],
            "reference_scale_seconds": float(target_config["reference_scale_seconds"]),
            "source_sha256": source_sha256,
            "source_bundle_sha256": _mapping_sha256(source_sha256),
            "evidence": dict(config["evidence"]),
        },
    )
    _write_json(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "tac_topology_target_v1.toml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "metadata" / "tac_topology_target_v1.json"
    )
    args = parser.parse_args(argv)
    payload = build_target(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "target_id": payload["target_id"],
                "reference_count": payload["reference_count"],
                "output": str(args.output.resolve()),
                "source_bundle_sha256": payload["source_bundle_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
