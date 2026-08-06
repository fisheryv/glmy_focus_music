from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import focus_topology
from focus_topology import AnalysisConfig, TopologyAnalyzer, analyze_states
from focus_topology.cli import main


def test_public_api_analyzes_cycle_and_normalizes_configuration() -> None:
    config = AnalysisConfig(thresholds=(0.5, 0.9, 0.5), top_k=None)

    result = analyze_states([0, 1, 2, 0], config=config)

    assert config.thresholds == (0.9, 0.5)
    assert result.betti_curve(0) == (1, 1)
    assert result.betti_curve(1) == (1, 1)
    assert result.metrics["h1_betti_max"] == 1
    assert result.graph.edge_pairs == ((0, 1), (1, 2), (2, 0))


def test_result_json_is_self_contained_and_can_omit_states(tmp_path: Path) -> None:
    analyzer = TopologyAnalyzer(AnalysisConfig(thresholds=(0.9, 0.5)))
    result = analyzer.analyze(
        [("pitch", 0), ("pitch", 1), None, ("pitch", 0)],
        metadata={"track_id": "example"},
    )

    payload = json.loads(result.to_json(include_states=False))
    output = result.write_json(tmp_path / "nested" / "result.json")

    assert "states" not in payload
    assert payload["metadata"]["track_id"] == "example"
    assert payload["graph"]["vertices"] == [["pitch", 0], ["pitch", 1]]
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_public_api_is_exported() -> None:
    assert focus_topology.AnalysisConfig is AnalysisConfig
    assert callable(focus_topology.analyze_audio)
    assert callable(focus_topology.persistent_path_homology)


def test_invalid_public_inputs_have_clear_errors() -> None:
    with pytest.raises(ValueError, match="at least one threshold"):
        AnalysisConfig(thresholds=())
    with pytest.raises(ValueError, match="not hashable"):
        analyze_states([[0], [1]])  # type: ignore[list-item]


def test_states_cli_writes_full_result(tmp_path: Path) -> None:
    states = tmp_path / "states.json"
    output = tmp_path / "result.json"
    states.write_text("[0, 1, 2, 0]\n", encoding="utf-8")

    return_code = main(
        [
            "states",
            str(states),
            "--thresholds",
            "0.9,0.5",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert return_code == 0
    assert payload["persistence"]["thresholds"] == [0.9, 0.5]
    assert payload["metrics"]["h1_betti_max"] == 1


def test_audio_api_uses_optional_extractor_without_leaking_feature_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from focus_topology import audio as audio_module

    path = tmp_path / "track.wav"
    path.write_bytes(b"placeholder")
    bundle = SimpleNamespace(
        duration_seconds=3.0,
        pitch=SimpleNamespace(
            states=np.asarray([0, 1, 2, 0]),
            valid=np.asarray([True, True, True, True]),
        ),
    )

    class FakeConfig:
        sample_rate = 22_050

    class FakeExtractor:
        def extract(self, *_: object, **__: object) -> SimpleNamespace:
            return bundle

    monkeypatch.setattr(
        audio_module,
        "_audio_dependencies",
        lambda: (FakeConfig, FakeExtractor, object(), object()),
    )
    monkeypatch.setattr(
        audio_module,
        "_extract_bundle",
        lambda *_args, **_kwargs: bundle,
    )

    result = audio_module.analyze_audio(
        path,
        config=AnalysisConfig(thresholds=(0.9, 0.5)),
    )

    assert result.metadata["view"] == "pitch"
    assert result.metadata["duration_seconds"] == 3.0
    assert result.betti_curve(1) == (1, 1)


def test_audio_api_canonicalizes_stereo_sample_rate(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    sample_rate = 44_100
    time = np.arange(sample_rate, dtype=float) / sample_rate
    mono = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    stereo = np.column_stack([mono, mono])
    path = tmp_path / "stereo-44k.wav"
    soundfile.write(path, stereo.astype(np.float32), sample_rate, subtype="FLOAT")

    result = focus_topology.analyze_audio(
        path,
        config=AnalysisConfig(thresholds=(0.9, 0.5)),
    )

    assert result.metadata["analysis_sample_rate"] == 22_050
    assert result.metadata["duration_seconds"] == pytest.approx(1.0)
