"""Auditable data-plane utilities for the frozen 18-D LTSN pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ltsn_contract import FingerprintContract, LTSNContractError, sha256_file
from .path_homology_exact_scorer import ExactPathHomologyScorer

TRAJECTORY_SCHEMA_VERSION = 1
RERANKING_GATE_NAME = "exact_reranking_effect_v1"
SURROGATE_TRAINING_GATE_NAME = "ltsn_surrogate_training_v1"
LABEL_SCOPE = "per_snapshot_exact"
SPLITS = ("train", "development", "calibration", "qualification")
FORMAL_SNAPSHOT_STEPS = (4, 5, 6, 8)


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload using a stable serialization."""

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without exposing a partially written audit artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty, rectangular CSV atomically."""

    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    columns = tuple(rows[0])
    if any(tuple(row) != columns for row in rows):
        raise ValueError("CSV rows do not share one field order")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LTSNContractError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RerankingGate:
    """Validated evidence that exact same-prompt reranking cleared Stage 1."""

    artifact_path: Path
    artifact_sha256: str
    median_loss_improvement_fraction: float
    bootstrap_ci95_low: float


@dataclass(frozen=True, slots=True)
class SurrogateTrainingGate:
    """Validated evidence permitting exact labeling and surrogate training only."""

    artifact_path: Path
    artifact_sha256: str
    median_loss_improvement_fraction: float
    bootstrap_ci95_low: float
    prompt_noninferior: bool
    diversity_preserved: bool


def load_reranking_gate(path: Path, contract: FingerprintContract) -> RerankingGate:
    """Load the separate exact-reranking effect gate or reject LTSN promotion."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != RERANKING_GATE_NAME or payload.get("status") != "passed":
        raise LTSNContractError("exact reranking effect gate has not passed")
    if payload.get("fingerprint_json_sha256") != contract.artifact_sha256:
        raise LTSNContractError("reranking gate uses a different exact scorer")
    improvement = _finite(
        payload.get("median_loss_improvement_fraction"),
        "median_loss_improvement_fraction",
    )
    ci_low = _finite(payload.get("bootstrap_ci95_low"), "bootstrap_ci95_low")
    if improvement < 0.10 or ci_low <= 0:
        raise LTSNContractError("reranking effect is below the frozen improvement/CI gate")
    required = (
        "target_band_hit_rate_improved",
        "quality_noninferior",
        "prompt_noninferior",
        "diversity_preserved",
    )
    if not all(payload.get(name) is True for name in required):
        raise LTSNContractError("reranking quality, prompt, hit-rate, or diversity gate failed")
    return RerankingGate(
        artifact_path=path.resolve(),
        artifact_sha256=sha256_file(path),
        median_loss_improvement_fraction=improvement,
        bootstrap_ci95_low=ci_low,
    )


def load_surrogate_training_gate(
    path: Path, contract: FingerprintContract
) -> SurrogateTrainingGate:
    """Load a scope-limited gate for labels/training without promoting guidance."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 2
        or payload.get("gate") != SURROGATE_TRAINING_GATE_NAME
        or payload.get("status") != "passed"
        or payload.get("scope") != "exact_labeling_and_surrogate_training_only"
    ):
        raise LTSNContractError("LTSN surrogate training gate has not passed")
    if payload.get("fingerprint_json_sha256") != contract.artifact_sha256:
        raise LTSNContractError("surrogate training gate uses a different exact scorer")
    improvement = _finite(
        payload.get("median_loss_improvement_fraction"),
        "median_loss_improvement_fraction",
    )
    ci_low = _finite(payload.get("bootstrap_ci95_low"), "bootstrap_ci95_low")
    if improvement < 0.10 or ci_low <= 0:
        raise LTSNContractError("surrogate training topology effect is below the frozen gate")
    if payload.get("target_band_hit_rate_improved") is not True:
        raise LTSNContractError("surrogate training requires improved target-band hit rate")
    if payload.get("all_selected_technical_quality_eligible") is not True:
        raise LTSNContractError("surrogate training selected candidates failed technical quality")
    if payload.get("quality_noninferior") is not True:
        raise LTSNContractError("surrogate training requires blind-quality non-inferiority")
    prompt_noninferior = payload.get("prompt_noninferior")
    diversity_preserved = payload.get("diversity_preserved")
    if not isinstance(prompt_noninferior, bool) or not isinstance(diversity_preserved, bool):
        raise LTSNContractError("surrogate training gate must record prompt and diversity results")
    if payload.get("guidance_promotion_eligible") is not False:
        raise LTSNContractError("surrogate training gate must not promote latent guidance")
    return SurrogateTrainingGate(
        artifact_path=path.resolve(),
        artifact_sha256=sha256_file(path),
        median_loss_improvement_fraction=improvement,
        bootstrap_ci95_low=ci_low,
        prompt_noninferior=prompt_noninferior,
        diversity_preserved=diversity_preserved,
    )


def require_surrogate_training_gate(
    gate_path: Path | None,
    contract: FingerprintContract,
    *,
    engineering_smoke: bool,
) -> SurrogateTrainingGate | None:
    """Enforce the scope-limited training gate, with a non-qualifying smoke bypass."""

    if engineering_smoke:
        return None
    if gate_path is None:
        raise LTSNContractError(
            "LTSN labeling/training is blocked until --surrogate-training-gate points to a "
            "passed gate; use --engineering-smoke only for non-qualifying pipeline tests"
        )
    return load_surrogate_training_gate(gate_path, contract)


@dataclass(frozen=True, slots=True)
class TrajectorySnapshotRecord:
    """One no-op-recorded ACE clean-latent estimate."""

    sample_id: str
    prompt_id: str
    trajectory_id: str
    split: str
    model_family: str
    step_number: int
    timestep: float
    is_final: bool
    latent_path: str
    latent_sha256: str
    audio_path: str
    audio_sha256: str
    ace_model_sha256: str
    vae_sha256: str
    engineering_smoke: bool
    schema_version: int = TRAJECTORY_SCHEMA_VERSION


class TrajectoryRecorder:
    """ACE sampler hook that records per-step ``x0_hat`` and is bitwise no-op."""

    def __init__(
        self,
        output_dir: Path,
        *,
        model_family: str,
        ace_model_sha256: str,
        vae_sha256: str,
        inference_steps: int = 8,
        selected_steps: Sequence[int] = (4, 5, 6),
        engineering_smoke: bool = False,
    ) -> None:
        if model_family not in {"acestep-v15-xl-turbo", "acestep-v15-xl-sft"}:
            raise LTSNContractError(
                "XL-Turbo and XL-SFT trajectories require an explicit model family"
            )
        for name, value in (("ace_model_sha256", ace_model_sha256), ("vae_sha256", vae_sha256)):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise LTSNContractError(f"{name} must be a lowercase SHA-256 digest")
        self.output_dir = output_dir.resolve()
        self.model_family = model_family
        self.ace_model_sha256 = ace_model_sha256
        self.vae_sha256 = vae_sha256
        self.inference_steps = int(inference_steps)
        self.selected_steps = frozenset((*selected_steps, self.inference_steps))
        self.engineering_smoke = engineering_smoke
        self.records: list[TrajectorySnapshotRecord] = []
        self._context: tuple[str, str, str] | None = None
        self._lock = threading.Lock()

    def begin(self, *, prompt_id: str, trajectory_id: str, split: str) -> None:
        """Set identity for the next single-trajectory generation call."""

        if split not in SPLITS:
            raise LTSNContractError(f"unsupported LTSN split: {split}")
        with self._lock:
            self._context = (prompt_id, trajectory_id, split)

    def end(self) -> None:
        """Clear trajectory identity after generation."""

        with self._lock:
            self._context = None

    @staticmethod
    def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part.npy")
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, path)

    def __call__(
        self,
        *,
        xt_next: Any,
        xt_before_step: Any,
        velocity: Any,
        timestep: float,
        next_timestep: float,
        step_index: int,
        attention_mask: Any,
        repaint_mask: Any | None = None,
    ) -> Any:
        """Record selected snapshots while returning ``xt_next`` unchanged."""

        del next_timestep, repaint_mask
        step_number = int(step_index) + 1
        if step_number not in self.selected_steps:
            return xt_next
        with self._lock:
            context = self._context
        if context is None:
            raise RuntimeError("TrajectoryRecorder.begin() was not called")
        prompt_id, trajectory_id, split = context
        clean = (xt_before_step.float() - float(timestep) * velocity.float()).detach().cpu()
        mask = attention_mask.detach().to(device="cpu", dtype=bool)
        if clean.ndim != 3 or clean.shape[-1] != 64 or mask.shape != clean.shape[:2]:
            raise LTSNContractError(
                "ACE trajectory hook must provide [B,T,64] latents and [B,T] mask"
            )
        for batch_index in range(clean.shape[0]):
            valid = mask[batch_index]
            if not bool(valid.any()):
                raise LTSNContractError("trajectory snapshot has no valid latent frames")
            latent = clean[batch_index, valid].numpy().astype(np.float32, copy=False)
            sample_id = f"{trajectory_id}__step{step_number:02d}__b{batch_index:02d}"
            latent_path = self.output_dir / "latents" / f"{sample_id}.npy"
            self._save_npy_atomic(latent_path, latent)
            self.records.append(
                TrajectorySnapshotRecord(
                    sample_id=sample_id,
                    prompt_id=prompt_id,
                    trajectory_id=trajectory_id,
                    split=split,
                    model_family=self.model_family,
                    step_number=step_number,
                    timestep=float(timestep),
                    is_final=step_number == self.inference_steps,
                    latent_path=latent_path.relative_to(self.output_dir).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path="",
                    audio_sha256="",
                    ace_model_sha256=self.ace_model_sha256,
                    vae_sha256=self.vae_sha256,
                    engineering_smoke=self.engineering_smoke,
                )
            )
        return xt_next

    def record_final_latent(self, final_latent: Any) -> tuple[str, ...]:
        """Persist ACE's decoder-input latent as the final sampler-step record."""

        with self._lock:
            context = self._context
        if context is None:
            raise RuntimeError("TrajectoryRecorder.begin() was not called")
        prompt_id, trajectory_id, split = context
        if final_latent is None:
            raise LTSNContractError(
                "ACE final pred_latents must have finite shape [B,T,64]"
            )
        value = final_latent
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "numpy"):
            value = value.numpy()
        batch = np.asarray(value, dtype=np.float32)
        if batch.ndim == 2:
            batch = batch[None, ...]
        if (
            batch.ndim != 3
            or batch.shape[0] < 1
            or batch.shape[1] < 1
            or batch.shape[2] != 64
            or not np.isfinite(batch).all()
        ):
            raise LTSNContractError(
                "ACE final pred_latents must have finite shape [B,T,64]"
            )
        if any(
            record.trajectory_id == trajectory_id and record.is_final
            for record in self.records
        ):
            raise LTSNContractError(
                f"trajectory already contains a final latent: {trajectory_id}"
            )
        sample_ids: list[str] = []
        for batch_index, latent in enumerate(batch):
            sample_id = (
                f"{trajectory_id}__step{self.inference_steps:02d}__b{batch_index:02d}"
            )
            latent_path = self.output_dir / "latents" / f"{sample_id}.npy"
            self._save_npy_atomic(latent_path, latent)
            self.records.append(
                TrajectorySnapshotRecord(
                    sample_id=sample_id,
                    prompt_id=prompt_id,
                    trajectory_id=trajectory_id,
                    split=split,
                    model_family=self.model_family,
                    step_number=self.inference_steps,
                    timestep=0.0,
                    is_final=True,
                    latent_path=latent_path.relative_to(self.output_dir).as_posix(),
                    latent_sha256=sha256_file(latent_path),
                    audio_path="",
                    audio_sha256="",
                    ace_model_sha256=self.ace_model_sha256,
                    vae_sha256=self.vae_sha256,
                    engineering_smoke=self.engineering_smoke,
                )
            )
            sample_ids.append(sample_id)
        return tuple(sample_ids)

    def attach_audio(self, sample_id: str, audio_path: Path) -> None:
        """Attach a separately VAE-decoded snapshot audio artifact."""

        resolved = audio_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        relative = resolved.relative_to(self.output_dir).as_posix()
        for index, record in enumerate(self.records):
            if record.sample_id == sample_id:
                payload = asdict(record)
                payload["audio_path"] = relative
                payload["audio_sha256"] = sha256_file(resolved)
                self.records[index] = TrajectorySnapshotRecord(**payload)
                return
        raise KeyError(f"unknown trajectory sample: {sample_id}")

    def write_manifest(self, path: Path) -> None:
        """Persist all recorded snapshot identities and upstream hashes."""

        rows = [asdict(record) for record in sorted(self.records, key=lambda item: item.sample_id)]
        write_csv_atomic(path, rows)


def validate_snapshot_coverage(
    rows: Sequence[Mapping[str, Any] | TrajectorySnapshotRecord],
    *,
    expected_steps: Sequence[int] = FORMAL_SNAPSHOT_STEPS,
) -> None:
    """Require exactly one selected-step snapshot and one final row per trajectory."""

    expected = {int(value) for value in expected_steps}
    if not expected:
        raise LTSNContractError("expected snapshot steps cannot be empty")
    final_step = max(expected)
    grouped: dict[str, list[tuple[int, bool]]] = {}
    for row in rows:
        if isinstance(row, TrajectorySnapshotRecord):
            trajectory_id = row.trajectory_id
            step_number = row.step_number
            is_final = row.is_final
        else:
            trajectory_id = str(row.get("trajectory_id", ""))
            step_number = int(row.get("step_number", 0))
            raw_final = row.get("is_final", False)
            is_final = (
                raw_final
                if isinstance(raw_final, bool)
                else str(raw_final).strip().lower() == "true"
            )
        if not trajectory_id:
            raise LTSNContractError("snapshot row is missing trajectory_id")
        grouped.setdefault(trajectory_id, []).append((step_number, is_final))
    if not grouped:
        raise LTSNContractError("snapshot coverage cannot be checked on an empty collection")
    for trajectory_id, items in grouped.items():
        steps = [step for step, _ in items]
        if len(steps) != len(expected) or set(steps) != expected:
            raise LTSNContractError(
                f"trajectory {trajectory_id} must contain exactly steps {sorted(expected)}; "
                f"found {sorted(steps)}"
            )
        for step, is_final in items:
            if is_final is not (step == final_step):
                raise LTSNContractError(
                    f"trajectory {trajectory_id} has inconsistent final-step metadata"
                )


def read_trajectory_manifest(path: Path) -> list[dict[str, str]]:
    """Read and minimally validate a trajectory collection manifest."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LTSNContractError("trajectory manifest is empty")
    prompt_splits: dict[str, set[str]] = {}
    model_families: set[str] = set()
    for row in rows:
        if row.get("split") not in SPLITS:
            raise LTSNContractError("trajectory manifest contains an unsupported split")
        prompt_splits.setdefault(row["prompt_id"], set()).add(row["split"])
        model_families.add(row["model_family"])
        latent_path = (path.parent / row["latent_path"]).resolve()
        if not latent_path.is_file() or sha256_file(latent_path) != row["latent_sha256"]:
            raise LTSNContractError(f"trajectory latent missing or hash-mismatched: {latent_path}")
    if any(len(values) != 1 for values in prompt_splits.values()):
        raise LTSNContractError("prompt leakage detected in trajectory manifest")
    if len(model_families) != 1:
        raise LTSNContractError("Turbo and SFT datasets must be collected separately")
    return rows


def build_exact_label_tables(
    *,
    trajectory_manifest: Path,
    descriptor_table: Path,
    output_manifest: Path,
    exact_label_table: Path,
    split_manifest: Path,
    scorer: ExactPathHomologyScorer,
    gate: SurrogateTrainingGate | None,
    engineering_smoke: bool,
) -> dict[str, Any]:
    """Join per-snapshot exact descriptors, score them, and issue hashed manifests."""

    trajectories = read_trajectory_manifest(trajectory_manifest)
    if not engineering_smoke:
        validate_snapshot_coverage(trajectories)
    with descriptor_table.open("r", encoding="utf-8-sig", newline="") as handle:
        descriptors = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if len(descriptors) != len(trajectories):
        raise LTSNContractError("descriptor table must contain exactly one row per snapshot")
    label_rows: list[dict[str, Any]] = []
    manifest_base: list[dict[str, Any]] = []
    for trajectory in trajectories:
        sample_id = trajectory["sample_id"]
        if sample_id not in descriptors:
            raise LTSNContractError(f"missing exact descriptor row for {sample_id}")
        descriptor = descriptors[sample_id]
        if not engineering_smoke:
            if descriptor.get("label_source") != "decoded_snapshot_exact_v1":
                raise LTSNContractError(
                    "qualification-eligible labels require decoded_snapshot_exact_v1 descriptors"
                )
            expected_audio_sha256 = trajectory.get("audio_sha256")
            if not expected_audio_sha256 or descriptor.get(
                "audio_sha256"
            ) != expected_audio_sha256:
                raise LTSNContractError("exact descriptor audio hash differs from the snapshot")
        pitch = json.loads(descriptor["pitch_descriptors_json"])
        score = scorer.score(
            pitch,
            [_finite(descriptor["acoustic_loop_score"], "acoustic_loop_score")],
            [_finite(descriptor["chroma_loop_score"], "chroma_loop_score")],
        )
        coordinates = score.coordinates[0].tolist()
        ood_label = _finite(descriptor.get("ood_label", 0.0), "ood_label")
        if ood_label not in {0.0, 1.0}:
            raise LTSNContractError("ood_label must be 0 or 1")
        label_rows.append(
            {
                "sample_id": sample_id,
                "pitch_descriptors_json": json.dumps(pitch, separators=(",", ":")),
                "acoustic_loop_score": descriptor["acoustic_loop_score"],
                "chroma_loop_score": descriptor["chroma_loop_score"],
                "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                "focus_logit": float(score.focus_logit[0]),
                "focus_probability": float(score.focus_probability[0]),
                "focus_band_loss": float(score.focus_band_loss[0]),
                "pitch_block_l2_norm": float(score.pitch_block_l2_norm[0]),
                "phase_block_l2_norm": float(score.phase_block_l2_norm[0]),
                "ood_label": ood_label,
                "label_scope": LABEL_SCOPE,
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
            }
        )
        manifest_base.append(
            {
                "sample_id": sample_id,
                "prompt_id": trajectory["prompt_id"],
                "trajectory_id": trajectory["trajectory_id"],
                "split": trajectory["split"],
                "model_family": trajectory["model_family"],
                "step_number": int(trajectory["step_number"]),
                "timestep": float(trajectory["timestep"]),
                "latent_path": os.path.relpath(
                    (trajectory_manifest.parent / trajectory["latent_path"]).resolve(),
                    output_manifest.parent.resolve(),
                ).replace("\\", "/"),
                "latent_sha256": trajectory["latent_sha256"],
                "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
                "focus_logit": float(score.focus_logit[0]),
                "ood_label": ood_label,
                "fingerprint_json_sha256": scorer.contract.artifact_sha256,
                "feature_order_json": json.dumps(list(scorer.contract.feature_order)),
                "label_scope": LABEL_SCOPE,
                "exact_label_table_sha256": "",
                "exact_label_table_path": os.path.relpath(
                    exact_label_table.resolve(), output_manifest.parent.resolve()
                ).replace("\\", "/"),
                "is_final": trajectory["is_final"],
                "ace_model_sha256": trajectory["ace_model_sha256"],
                "vae_sha256": trajectory["vae_sha256"],
                "qualification_eligible": str(not engineering_smoke).lower(),
                "surrogate_training_gate_sha256": (
                    "" if gate is None else gate.artifact_sha256
                ),
                "guidance_promotion_eligible": "false",
            }
        )
    write_csv_atomic(exact_label_table, label_rows)
    label_sha256 = sha256_file(exact_label_table)
    for row in manifest_base:
        row["exact_label_table_sha256"] = label_sha256
    split_payload = {
        "schema_version": 1,
        "grouping": "prompt_id+trajectory_id",
        "assignments": dict(
            sorted({row["prompt_id"]: row["split"] for row in manifest_base}.items())
        ),
        "qualification_eligible": not engineering_smoke,
    }
    write_json_atomic(split_manifest, split_payload)
    write_csv_atomic(output_manifest, manifest_base)
    return {
        "samples": len(manifest_base),
        "prompts": len({row["prompt_id"] for row in manifest_base}),
        "model_family": manifest_base[0]["model_family"],
        "exact_label_table_sha256": label_sha256,
        "training_manifest_sha256": sha256_file(output_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "qualification_eligible": not engineering_smoke,
        "surrogate_training_gate_sha256": "" if gate is None else gate.artifact_sha256,
        "guidance_promotion_eligible": False,
    }


def synthetic_descriptor_rows(
    trajectory_manifest: Path,
    *,
    pitch_dimensions: int,
) -> list[dict[str, Any]]:
    """Create deterministic fake descriptors for non-qualifying CI/smoke runs only."""

    rows = read_trajectory_manifest(trajectory_manifest)
    output: list[dict[str, Any]] = []
    for row in rows:
        latent = np.load(trajectory_manifest.parent / row["latent_path"], allow_pickle=False)
        mean = float(np.mean(latent))
        scale = float(np.std(latent))
        pitch = [mean + scale * math.sin(index + 1.0) for index in range(pitch_dimensions)]
        output.append(
            {
                "sample_id": row["sample_id"],
                "pitch_descriptors_json": json.dumps(pitch, separators=(",", ":")),
                "acoustic_loop_score": 1.0 / (1.0 + math.exp(-mean)),
                "chroma_loop_score": 1.0 / (1.0 + math.exp(-scale)),
                "ood_label": 0,
                "label_source": "engineering_smoke_synthetic",
                "audio_sha256": "",
            }
        )
    return output


def model_data_identity(rows: Iterable[Mapping[str, str]]) -> tuple[str, str, str]:
    """Return the one ACE model hash, VAE hash, and model family in a manifest."""

    rows = list(rows)
    identities = {
        (row["ace_model_sha256"], row["vae_sha256"], row["model_family"]) for row in rows
    }
    if len(identities) != 1:
        raise LTSNContractError("manifest mixes ACE/VAE/model-family identities")
    return identities.pop()
