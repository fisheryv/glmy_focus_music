from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from topology.multiview_fusion import (
    DiscoveryMahalanobisBlock,
    equal_block_fusion,
    hierarchical_fusion,
    paired_incremental_permutation,
    permutation_pseudo_f,
)
from topology.statistics import TOPOLOGY_METRICS, benjamini_hochberg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata"
GATE = METADATA / "holdout_gate.json"
SUMMARY = METADATA / "holdout_confirmation_summary.json"
EXECUTION = METADATA / "holdout_confirmation_execution.json"
PERMANOVA = METADATA / "holdout_confirmation_permanova.csv"
INCREMENTAL = METADATA / "holdout_confirmation_incremental.csv"
DIRECTIONAL = METADATA / "holdout_confirmation_directional_metrics.csv"
OUTPUT = ROOT / "runs" / "holdout_confirmation"
FIGURES = OUTPUT / "figures"
VIEW_FILES = {
    "pitch": METADATA / "pitch_v2_topology_segments.csv",
    "rhythm": METADATA / "rhythm_topology_segments.csv",
    "modulation": METADATA / "modulation_tertile_topology_segments.csv",
    "structure": METADATA / "structure_topology_segments.csv",
}
IDENTITY = ["segment_id", "track_id", "group", "split", "scale_seconds"]
FEATURE_SETS = ("local", "pitch", "rhythm", "modulation", "structure", "hierarchical")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _verify_gate(gate: dict[str, Any]) -> None:
    if gate.get("status") != "frozen_before_holdout_opening":
        raise RuntimeError("holdout gate is not in the frozen state")
    for category in (
        "config_sha256",
        "input_sha256",
        "model_sha256",
        "validation_artifact_sha256",
    ):
        for relative, expected in gate[category].items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"gate hash mismatch: {relative}")


def _load_aligned() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    canonical: pd.MultiIndex | None = None
    for view, path in VIEW_FILES.items():
        frame = pd.read_csv(path)
        required = set(IDENTITY) | set(TOPOLOGY_METRICS) | {"status"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{view} missing columns: {sorted(missing)}")
        if frame.duplicated(IDENTITY).any() or (frame["status"] == "failed").any():
            raise RuntimeError(f"{view} topology manifest is not complete and unique")
        indexed = frame.set_index(IDENTITY).sort_index()
        if canonical is None:
            canonical = indexed.index
        elif not indexed.index.equals(canonical):
            raise RuntimeError(f"{view} identities do not align")
        frames[view] = indexed
    assert canonical is not None
    return canonical.to_frame(index=False), frames


def _mask(identity: pd.DataFrame, split: str, scale: float) -> np.ndarray:
    return (identity["split"].astype(str).to_numpy() == split) & np.isclose(
        identity["scale_seconds"].to_numpy(float), scale
    )


def _blocks(
    identity: pd.DataFrame, frames: dict[str, pd.DataFrame], scale: float
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    discovery = _mask(identity, "discovery", scale)
    holdout = _mask(identity, "holdout", scale)
    if discovery.sum() != 390 or holdout.sum() != 90:
        raise RuntimeError(
            f"unexpected {scale:g}s discovery/holdout counts: "
            f"{discovery.sum()}/{holdout.sum()}"
        )
    holdout_groups = identity.loc[holdout, "group"].value_counts().to_dict()
    if holdout_groups != {"classical": 45, "focus": 45}:
        raise RuntimeError(f"unexpected holdout group counts: {holdout_groups}")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view, frame in frames.items():
        values = frame.loc[:, TOPOLOGY_METRICS].to_numpy(float)
        transformer = DiscoveryMahalanobisBlock().fit(values[discovery])
        result[view] = (
            transformer.transform(values[discovery]),
            transformer.transform(values[holdout]),
        )
    return discovery, holdout, result


def _feature_sets(
    blocks: dict[str, tuple[np.ndarray, np.ndarray]], position: int
) -> dict[str, np.ndarray]:
    values = {view: payload[position] for view, payload in blocks.items()}
    local = equal_block_fusion([values[view] for view in ("pitch", "rhythm", "modulation")])
    return {
        "local": local,
        "pitch": values["pitch"],
        "rhythm": values["rhythm"],
        "modulation": values["modulation"],
        "structure": values["structure"],
        "hierarchical": hierarchical_fusion(
            local, values["structure"], structure_weight=0.5
        ),
    }


def _run_fusion_tests(
    identity: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    gate: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = gate["analysis_specification"]
    permutations = int(spec["permutations"])
    seed = int(spec["seed"])
    permanova_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    for scale_index, scale in enumerate((180.0, 300.0)):
        _, holdout, blocks = _blocks(identity, frames, scale)
        matrices = _feature_sets(blocks, 1)
        labels = identity.loc[holdout, "group"].astype(str).to_numpy()
        for feature_index, name in enumerate(FEATURE_SETS):
            result = permutation_pseudo_f(
                matrices[name],
                labels,
                permutations=permutations,
                seed=seed + scale_index * 100 + feature_index,
            )
            permanova_rows.append(
                {
                    "analysis_set": "primary_holdout_180"
                    if scale == 180.0
                    else "sensitivity_holdout_300",
                    "scale_seconds": scale,
                    "feature_set": name,
                    "role": "primary" if scale == 180.0 and name == "local" else "secondary",
                    "n_holdout": int(holdout.sum()),
                    "permutations": permutations,
                    **result,
                }
            )
        comparisons = (
            ("local_vs_pitch", "local", "pitch"),
            ("add_structure", "hierarchical", "local"),
        )
        for comparison_index, (name, candidate, baseline) in enumerate(comparisons):
            result = paired_incremental_permutation(
                matrices[candidate],
                matrices[baseline],
                labels,
                permutations=permutations,
                seed=seed + 300 + scale_index * 100 + comparison_index,
            )
            incremental_rows.append(
                {
                    "analysis_set": "primary_holdout_180"
                    if scale == 180.0
                    else "sensitivity_holdout_300",
                    "scale_seconds": scale,
                    "comparison": name,
                    "candidate": candidate,
                    "baseline": baseline,
                    "alternative": "candidate pseudo-F > baseline pseudo-F",
                    "permutations": permutations,
                    **result,
                }
            )
    permanova = pd.DataFrame(permanova_rows)
    permanova["p_fdr_bh"] = np.nan
    for scale, indices in permanova.groupby("scale_seconds").groups.items():
        secondary = [index for index in indices if permanova.loc[index, "role"] == "secondary"]
        permanova.loc[secondary, "p_fdr_bh"] = benjamini_hochberg(
            permanova.loc[secondary, "p_value"].to_numpy(float)
        )
        if scale == 180.0:
            primary = [index for index in indices if permanova.loc[index, "role"] == "primary"]
            permanova.loc[primary, "p_fdr_bh"] = permanova.loc[primary, "p_value"]
    incremental = pd.DataFrame(incremental_rows)
    for _, indices in incremental.groupby("scale_seconds").groups.items():
        incremental.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            incremental.loc[indices, "p_value_one_sided"].to_numpy(float)
        )
    return permanova, incremental


def _run_directional_tests(
    identity: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    gate: dict[str, Any],
) -> pd.DataFrame:
    locked = gate["analysis_specification"]["directional_metrics"]
    rows: list[dict[str, Any]] = []
    for scale in (180.0, 300.0):
        holdout = _mask(identity, "holdout", scale)
        labels = identity.loc[holdout, "group"].astype(str).to_numpy()
        for endpoint in locked:
            values = frames[endpoint["view"]].loc[:, endpoint["metric"]].to_numpy(float)[holdout]
            classical = values[labels == "classical"]
            focus = values[labels == "focus"]
            direction = endpoint["expected_focus_direction"]
            result = mannwhitneyu(focus, classical, alternative=direction, method="asymptotic")
            difference = float(np.median(focus) - np.median(classical))
            observed_direction = (
                "greater" if difference > 0 else "less" if difference < 0 else "equal"
            )
            rows.append(
                {
                    "analysis_set": "primary_holdout_180"
                    if scale == 180.0
                    else "sensitivity_holdout_300",
                    "scale_seconds": scale,
                    "view": endpoint["view"],
                    "metric": endpoint["metric"],
                    "expected_focus_direction": direction,
                    "observed_focus_direction": observed_direction,
                    "direction_matched": observed_direction == direction,
                    "classical_median": float(np.median(classical)),
                    "focus_median": float(np.median(focus)),
                    "median_difference_focus_minus_classical": difference,
                    "u_statistic": float(result.statistic),
                    "p_value_one_sided": float(result.pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    for _, indices in frame.groupby("scale_seconds").groups.items():
        frame.loc[indices, "p_fdr_bh"] = benjamini_hochberg(
            frame.loc[indices, "p_value_one_sided"].to_numpy(float)
        )
    frame["replicated_q_0_10"] = frame["direction_matched"] & (frame["p_fdr_bh"] <= 0.10)
    return frame


def _plot_permanova(frame: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = list(FEATURE_SETS)
    x = np.arange(len(order))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    series = (
        (-width / 2, 180.0, "180 s primary"),
        (width / 2, 300.0, "300 s sensitivity"),
    )
    for offset, scale, label in series:
        subset = frame[frame["scale_seconds"] == scale].set_index("feature_set").loc[order]
        bars = axis.bar(x + offset, subset["pseudo_f"], width=width, label=label)
        for bar, p_value in zip(bars, subset["p_value"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"p={p_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axis.set_xticks(x, order, rotation=20, ha="right")
    axis.set_ylabel("Permutation pseudo-F")
    axis.set_title("Frozen holdout Path Homology endpoints")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"holdout_permanova.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_replication(frame: pd.DataFrame) -> None:
    primary = frame[frame["scale_seconds"] == 180.0]
    summary = (
        primary.groupby("view")["replicated_q_0_10"]
        .agg(["sum", "count"])
        .reindex(["pitch", "rhythm", "modulation", "structure"])
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar(summary.index, summary["sum"], color="C0")
    for bar, replicated, total in zip(bars, summary["sum"], summary["count"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(replicated)}/{int(total)}",
            ha="center",
            va="bottom",
        )
    axis.set_ylabel("Directional metrics replicated (BH q ≤ 0.10)")
    axis.set_title("Validation-selected metric replication in 180 s holdout")
    axis.set_ylim(0, max(1.0, float(summary["count"].max()) * 1.15))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURES / f"holdout_directional_replication.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    if SUMMARY.exists() or EXECUTION.exists():
        raise RuntimeError("holdout confirmation has already been executed")
    if not GATE.is_file():
        raise RuntimeError("holdout gate does not exist")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    _verify_gate(gate)
    gate_sha256 = _sha256(GATE)

    identity, frames = _load_aligned()
    permanova, incremental = _run_fusion_tests(identity, frames, gate)
    directional = _run_directional_tests(identity, frames, gate)
    permanova.to_csv(PERMANOVA, index=False, encoding="utf-8", lineterminator="\n")
    incremental.to_csv(INCREMENTAL, index=False, encoding="utf-8", lineterminator="\n")
    directional.to_csv(DIRECTIONAL, index=False, encoding="utf-8", lineterminator="\n")
    _plot_permanova(permanova)
    _plot_replication(directional)

    primary = permanova[
        (permanova["scale_seconds"] == 180.0) & (permanova["feature_set"] == "local")
    ].iloc[0]
    hierarchical = permanova[
        (permanova["scale_seconds"] == 180.0)
        & (permanova["feature_set"] == "hierarchical")
    ].iloc[0]
    primary_directional = directional[directional["scale_seconds"] == 180.0]
    payload = {
        "generated_at": date.today().isoformat(),
        "status": "completed_once",
        "gate_sha256": gate_sha256,
        "scientific_scope": gate["scientific_scope"],
        "holdout_counts": {"classical": 45, "focus": 45},
        "primary_180": {
            "feature_set": "local",
            "pseudo_f": float(primary["pseudo_f"]),
            "p_value": float(primary["p_value"]),
            "confirmed_at_alpha_0_05": bool(primary["p_value"] <= 0.05),
        },
        "secondary_hierarchical_180": {
            "pseudo_f": float(hierarchical["pseudo_f"]),
            "p_value": float(hierarchical["p_value"]),
            "p_fdr_bh": float(hierarchical["p_fdr_bh"]),
        },
        "directional_metric_replication_180": {
            "locked_metrics": int(len(primary_directional)),
            "direction_matched": int(primary_directional["direction_matched"].sum()),
            "replicated_q_0_10": int(primary_directional["replicated_q_0_10"].sum()),
            "fdr_family": "all validation-selected directional metrics across four views",
        },
        "adaptation_audit": {
            "parameters_refit": False,
            "metrics_reselected": False,
            "directions_changed": False,
            "fusion_weights_changed": False,
            "thresholds_changed": False,
            "fdr_family_changed": False,
        },
        "artifacts": {
            "permanova": PERMANOVA.relative_to(ROOT).as_posix(),
            "incremental": INCREMENTAL.relative_to(ROOT).as_posix(),
            "directional": DIRECTIONAL.relative_to(ROOT).as_posix(),
            "figures": [path.relative_to(ROOT).as_posix() for path in sorted(FIGURES.glob("*"))],
        },
    }
    _write_json(SUMMARY, payload)
    outputs = (SUMMARY, PERMANOVA, INCREMENTAL, DIRECTIONAL, *sorted(FIGURES.glob("*")))
    execution = {
        "executed_at": date.today().isoformat(),
        "status": "completed_once",
        "gate_sha256": gate_sha256,
        "output_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in outputs},
    }
    _write_json(EXECUTION, execution)
    print(json.dumps({**payload, "execution": execution}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
