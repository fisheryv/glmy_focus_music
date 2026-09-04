from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .experiment import CandidateRecord, read_candidate_manifest
from .ltsn_contract import sha256_file

OUTPUT_COLUMNS = (
    "prompt_id",
    "baseline_candidate_id",
    "selected_candidate_id",
    "quality_baseline",
    "quality_selected",
    "prompt_baseline",
    "prompt_selected",
    "diversity_baseline",
    "diversity_selected",
)

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PromptPair:
    prompt_id: str
    caption: str
    baseline_candidate_id: str
    selected_candidate_id: str


@dataclass(frozen=True, slots=True)
class EvidenceInputs:
    pairs: tuple[PromptPair, ...]
    candidates: dict[str, CandidateRecord]
    audio_paths: dict[str, Path]
    pool_summary_path: Path
    candidate_manifest_path: Path
    prompt_manifest_path: Path


class EmbeddingBackend(Protocol):
    sampling_rate: int
    maximum_audio_samples: int
    model_commit: str

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray: ...

    def embed_text(self, captions: Sequence[str]) -> np.ndarray: ...


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _unique_by(
    rows: Sequence[dict[str, str]], key: str, *, source: Path
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = (row.get(key) or "").strip()
        if not value or value in result:
            raise ValueError(f"{source} must contain one non-empty row per {key}")
        result[value] = row
    return result


def _resolve_audio_path(run_root: Path, relative_path: str) -> Path:
    if not relative_path.strip():
        raise ValueError("candidate audio_relative_path is empty")
    root = run_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"candidate audio path escapes run root: {relative_path}")
    return candidate


def load_evidence_inputs(
    run_root: Path,
    prompt_manifest_path: Path,
    *,
    expected_prompt_count: int = 32,
) -> EvidenceInputs:
    """Load and hash-verify the exact baseline/selected candidate bindings."""

    run_root = run_root.resolve()
    pool_path = run_root / "pool_summary.csv"
    candidate_path = run_root / "manifests" / "candidates.csv"
    pool_by_prompt = _unique_by(_read_csv(pool_path), "prompt_id", source=pool_path)
    prompt_by_id = _unique_by(
        _read_csv(prompt_manifest_path), "prompt_id", source=prompt_manifest_path
    )
    if len(pool_by_prompt) != expected_prompt_count:
        raise ValueError(
            f"expected {expected_prompt_count} complete prompt pools, found {len(pool_by_prompt)}"
        )
    if set(pool_by_prompt) != set(prompt_by_id):
        missing = sorted(set(pool_by_prompt) - set(prompt_by_id))
        extra = sorted(set(prompt_by_id) - set(pool_by_prompt))
        raise ValueError(
            "prompt manifest IDs do not exactly match pool_summary.csv: "
            f"missing={missing}, extra={extra}"
        )

    candidate_records = read_candidate_manifest(candidate_path)
    candidates: dict[str, CandidateRecord] = {}
    for record in candidate_records:
        if record.candidate_id in candidates:
            raise ValueError(f"duplicate candidate_id in manifest: {record.candidate_id}")
        candidates[record.candidate_id] = record

    pairs: list[PromptPair] = []
    required_ids: set[str] = set()
    for prompt_id in sorted(pool_by_prompt):
        pool = pool_by_prompt[prompt_id]
        prompt = prompt_by_id[prompt_id]
        caption = (prompt.get("caption") or "").strip()
        if not caption:
            raise ValueError(f"prompt caption is empty: {prompt_id}")
        baseline_id = (pool.get("baseline_candidate_id") or "").strip()
        selected_id = (pool.get("selected_candidate_id") or "").strip()
        if not baseline_id or not selected_id:
            raise ValueError(f"pool candidate binding is incomplete: {prompt_id}")
        for candidate_id in (baseline_id, selected_id):
            record = candidates.get(candidate_id)
            if record is None:
                raise ValueError(f"candidate is absent from manifest: {candidate_id}")
            if record.prompt_id != prompt_id or record.caption != caption:
                raise ValueError(f"candidate prompt binding mismatch: {candidate_id}")
            if record.status != "scored":
                raise ValueError(f"candidate has not completed exact scoring: {candidate_id}")
            required_ids.add(candidate_id)
        pairs.append(PromptPair(prompt_id, caption, baseline_id, selected_id))

    audio_paths: dict[str, Path] = {}
    for candidate_id in sorted(required_ids):
        record = candidates[candidate_id]
        path = _resolve_audio_path(run_root, record.audio_relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"candidate audio is missing: {path}")
        expected_hash = record.audio_sha256.strip().lower()
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"candidate audio SHA-256 is malformed: {candidate_id}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"candidate audio SHA-256 mismatch: {candidate_id}")
        audio_paths[candidate_id] = path

    return EvidenceInputs(
        pairs=tuple(pairs),
        candidates=candidates,
        audio_paths=audio_paths,
        pool_summary_path=pool_path,
        candidate_manifest_path=candidate_path,
        prompt_manifest_path=prompt_manifest_path.resolve(),
    )


def normalize_embeddings(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero-norm row")
    return matrix / norms


def nearest_neighbor_diversity(embeddings: np.ndarray) -> np.ndarray:
    """Return 1 - maximum off-diagonal cosine similarity for each item."""

    normalized = normalize_embeddings(embeddings)
    if normalized.shape[0] < 2:
        raise ValueError("diversity requires at least two prompt pools")
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return 1.0 - np.max(similarities, axis=1)


def split_audio(audio: np.ndarray, segment_samples: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Split a mono track into fixed windows, zero-padding only its final window."""

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("decoded audio must be a finite, non-empty mono waveform")
    if segment_samples < 1:
        raise ValueError("segment_samples must be positive")
    segments: list[np.ndarray] = []
    weights: list[float] = []
    for start in range(0, waveform.size, segment_samples):
        chunk = waveform[start : start + segment_samples]
        valid = chunk.size
        if valid < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - valid))
        segments.append(np.asarray(chunk, dtype=np.float32))
        weights.append(float(valid))
    return segments, np.asarray(weights, dtype=np.float64)


def _load_resampled_mono(path: Path, target_rate: int) -> np.ndarray:
    try:
        import soundfile
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("audio metrics require soundfile and scipy") from exc

    audio, source_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or source_rate <= 0:
        raise ValueError(f"decoded audio is empty or invalid: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), int(target_rate))
        mono = resample_poly(
            mono,
            target_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)
    if not np.isfinite(mono).all():
        raise ValueError(f"decoded audio contains non-finite samples: {path}")
    return mono


def embed_track(
    path: Path,
    backend: EmbeddingBackend,
    *,
    segment_seconds: float,
    batch_size: int,
) -> np.ndarray:
    if segment_seconds <= 0 or batch_size < 1:
        raise ValueError("segment_seconds and batch_size must be positive")
    segment_samples = int(round(segment_seconds * backend.sampling_rate))
    if segment_samples > backend.maximum_audio_samples:
        maximum_seconds = backend.maximum_audio_samples / backend.sampling_rate
        raise ValueError(
            f"segment_seconds exceeds the CLAP feature window ({maximum_seconds:g}s)"
        )
    audio = _load_resampled_mono(path, backend.sampling_rate)
    segments, weights = split_audio(audio, segment_samples)
    batches: list[np.ndarray] = []
    for start in range(0, len(segments), batch_size):
        values = backend.embed_audio(segments[start : start + batch_size])
        batches.append(normalize_embeddings(values))
    segment_embeddings = np.concatenate(batches, axis=0)
    if segment_embeddings.shape[0] != len(segments):
        raise ValueError("audio backend returned the wrong number of embeddings")
    track = np.average(segment_embeddings, axis=0, weights=weights)
    return normalize_embeddings(track[None, :])[0]


def _load_quality_table(
    path: Path | None, pairs: Sequence[PromptPair]
) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    rows = _unique_by(_read_csv(path), "prompt_id", source=path)
    if set(rows) != {pair.prompt_id for pair in pairs}:
        raise ValueError("quality table must contain exactly one row per reranking prompt")
    result: dict[str, tuple[str, str]] = {}
    for pair in pairs:
        row = rows[pair.prompt_id]
        if row.get("baseline_candidate_id") != pair.baseline_candidate_id or row.get(
            "selected_candidate_id"
        ) != pair.selected_candidate_id:
            raise ValueError(f"quality candidate binding mismatch: {pair.prompt_id}")
        baseline = (row.get("quality_baseline") or "").strip()
        selected = (row.get("quality_selected") or "").strip()
        try:
            values = (float(baseline), float(selected))
        except ValueError as exc:
            raise ValueError(f"quality scores are not numeric: {pair.prompt_id}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"quality scores are not finite: {pair.prompt_id}")
        result[pair.prompt_id] = (baseline, selected)
    return result


def build_metric_rows(
    pairs: Sequence[PromptPair],
    candidate_embeddings: dict[str, np.ndarray],
    text_embeddings: np.ndarray,
    quality: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    if not pairs:
        raise ValueError("no prompt pairs were supplied")
    text = normalize_embeddings(text_embeddings)
    if text.shape[0] != len(pairs):
        raise ValueError("text backend returned the wrong number of embeddings")
    baseline = normalize_embeddings(
        np.stack([candidate_embeddings[pair.baseline_candidate_id] for pair in pairs])
    )
    selected = normalize_embeddings(
        np.stack([candidate_embeddings[pair.selected_candidate_id] for pair in pairs])
    )
    if baseline.shape[1] != text.shape[1] or selected.shape[1] != text.shape[1]:
        raise ValueError("audio and text embedding dimensions differ")
    baseline_diversity = nearest_neighbor_diversity(baseline)
    selected_diversity = nearest_neighbor_diversity(selected)
    quality = quality or {}
    rows: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        quality_values = quality.get(pair.prompt_id, ("", ""))
        rows.append(
            {
                "prompt_id": pair.prompt_id,
                "baseline_candidate_id": pair.baseline_candidate_id,
                "selected_candidate_id": pair.selected_candidate_id,
                "quality_baseline": quality_values[0],
                "quality_selected": quality_values[1],
                "prompt_baseline": format(float(baseline[index] @ text[index]), ".17g"),
                "prompt_selected": format(float(selected[index] @ text[index]), ".17g"),
                "diversity_baseline": format(float(baseline_diversity[index]), ".17g"),
                "diversity_selected": format(float(selected_diversity[index]), ".17g"),
            }
        )
    return rows


def _write_csv_atomic(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class TransformersClapBackend:
    def __init__(self, model_id: str, revision: str, device: str) -> None:
        if not _COMMIT_SHA.fullmatch(revision):
            raise ValueError("--model-revision must be an immutable 40-hex commit SHA")
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError("CLAP metrics require torch and transformers") from exc

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        self._processor = ClapProcessor.from_pretrained(model_id, revision=revision)
        self._model = ClapModel.from_pretrained(model_id, revision=revision).to(self._device)
        self._model.eval()
        feature_extractor = self._processor.feature_extractor
        self.sampling_rate = int(feature_extractor.sampling_rate)
        self.maximum_audio_samples = int(
            getattr(
                feature_extractor,
                "nb_max_samples",
                round(float(feature_extractor.max_length_s) * self.sampling_rate),
            )
        )
        resolved = getattr(self._model.config, "_commit_hash", None)
        self.model_commit = str(resolved or revision).lower()
        if self.model_commit != revision.lower():
            raise ValueError(
                f"resolved model commit differs from requested revision: {self.model_commit}"
            )

    def _to_device(self, inputs: Any) -> dict[str, Any]:
        return {name: value.to(self._device) for name, value in inputs.items()}

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        inputs = self._processor(
            audios=list(waveforms),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            output = self._model.get_audio_features(**self._to_device(inputs))
        return output.float().cpu().numpy()

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        inputs = self._processor(text=list(captions), padding=True, return_tensors="pt")
        with self._torch.inference_mode():
            output = self._model.get_text_features(**self._to_device(inputs))
        return output.float().cpu().numpy()


def generate_noninferiority_metrics(
    *,
    run_root: Path,
    prompt_manifest_path: Path,
    output_path: Path,
    audit_path: Path,
    backend: EmbeddingBackend,
    model_id: str,
    model_revision: str,
    device: str,
    batch_size: int,
    segment_seconds: float,
    quality_table_path: Path | None = None,
    expected_prompt_count: int = 32,
) -> dict[str, Any]:
    inputs = load_evidence_inputs(
        run_root, prompt_manifest_path, expected_prompt_count=expected_prompt_count
    )
    quality = _load_quality_table(quality_table_path, inputs.pairs)
    candidate_embeddings: dict[str, np.ndarray] = {}
    for candidate_id, path in inputs.audio_paths.items():
        candidate_embeddings[candidate_id] = embed_track(
            path,
            backend,
            segment_seconds=segment_seconds,
            batch_size=batch_size,
        )
    captions = [pair.caption for pair in inputs.pairs]
    text_batches: list[np.ndarray] = []
    for start in range(0, len(captions), batch_size):
        text_batches.append(backend.embed_text(captions[start : start + batch_size]))
    text_embeddings = np.concatenate(text_batches, axis=0)
    rows = build_metric_rows(inputs.pairs, candidate_embeddings, text_embeddings, quality)
    _write_csv_atomic(output_path, rows)

    versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "soundfile", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "UNAVAILABLE"
    audit = {
        "schema_version": 1,
        "metric_contract": "clap_prompt_and_cohort_diversity_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "id": model_id,
            "requested_revision": model_revision.lower(),
            "resolved_commit": backend.model_commit,
        },
        "runtime": {
            "device": device,
            "batch_size": batch_size,
            "package_versions": versions,
        },
        "audio": {
            "sampling_rate": backend.sampling_rate,
            "segment_seconds": segment_seconds,
            "segmentation": (
                "contiguous non-overlapping windows; zero-pad final window; "
                "valid-sample-weighted mean of L2-normalized window embeddings; "
                "L2-normalize track embedding"
            ),
        },
        "formulas": {
            "prompt": "cosine(normalized_track_audio_embedding, normalized_prompt_text_embedding)",
            "diversity": "1 - max_{j != i} cosine(normalized_track_i, normalized_track_j)",
            "diversity_cohorts": "baseline and selected cohorts are evaluated separately",
            "direction": "higher_is_better",
        },
        "inputs": {
            "pool_summary": {
                "path": str(inputs.pool_summary_path),
                "sha256": sha256_file(inputs.pool_summary_path),
            },
            "candidate_manifest": {
                "path": str(inputs.candidate_manifest_path),
                "sha256": sha256_file(inputs.candidate_manifest_path),
            },
            "prompt_manifest": {
                "path": str(inputs.prompt_manifest_path),
                "sha256": sha256_file(inputs.prompt_manifest_path),
            },
            "quality_table": (
                {
                    "path": str(quality_table_path.resolve()),
                    "sha256": sha256_file(quality_table_path),
                }
                if quality_table_path is not None
                else None
            ),
            "audio_sha256": {
                candidate_id: inputs.candidates[candidate_id].audio_sha256
                for candidate_id in sorted(inputs.audio_paths)
            },
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "rows": len(rows),
            "quality_columns_complete": quality_table_path is not None,
        },
    }
    _write_json_atomic(audit_path, audit)
    return audit
