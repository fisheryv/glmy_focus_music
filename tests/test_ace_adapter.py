from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from generation.ace_adapter import AceStepAdapter


def test_snapshot_decode_uses_worker_device(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeTensor:
        def unsqueeze(self, _dimension: int):
            return self

        def to(self, device):
            captured["device"] = device
            return self

    class FakeWaveform:
        def __getitem__(self, _index: int):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((2, 8), dtype=np.float32)

    class FakeHandler:
        sample_rate = 48_000

        def _decode_generate_music_pred_latents(self, **_kwargs):
            return FakeWaveform(), None, None

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda value: f"device:{value}"  # type: ignore[attr-defined]
    fake_torch.from_numpy = lambda _values: FakeTensor()  # type: ignore[attr-defined]
    fake_soundfile = ModuleType("soundfile")

    def write(path: str, _audio, _sample_rate: int, *, subtype: str) -> None:
        assert subtype == "FLOAT"
        Path(path).write_bytes(b"fake-wave")

    fake_soundfile.write = write  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    adapter = object.__new__(AceStepAdapter)
    adapter._handler = FakeHandler()
    adapter._device = "cuda:3"
    adapter.initialize = lambda: None

    output = adapter.decode_latent_to_audio(
        np.zeros((16, 64), dtype=np.float32),
        tmp_path / "snapshot.wav",
    )

    assert captured["device"] == "device:cuda:3"
    assert output.is_file()
