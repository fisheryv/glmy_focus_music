from __future__ import annotations

import math

import numpy as np

from .ace_adapter import GenerationRequest, GenerationResult


class FakeMusicBackend:
    """Deterministic CPU backend used to exercise the cloud pipeline locally."""

    sample_rate = 48_000

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            import soundfile
        except ImportError as exc:
            raise RuntimeError("fake generation requires soundfile; install .[audio]") from exc

        rng = np.random.default_rng(request.seed)
        sample_count = int(round(request.duration_seconds * self.sample_rate))
        time = np.arange(sample_count, dtype=np.float64) / self.sample_rate
        bpm = float(request.bpm or rng.integers(72, 101))
        beat_period = 60.0 / bpm
        base_frequency = float(rng.choice([110.0, 130.81, 146.83, 164.81, 196.0]))
        phase = np.mod(time, beat_period) / beat_period
        pulse = np.exp(-8.0 * phase)
        slow = 0.75 + 0.25 * np.sin(2.0 * math.pi * time / float(rng.integers(8, 17)))
        tone = (
            0.10 * np.sin(2.0 * math.pi * base_frequency * time)
            + 0.04 * np.sin(2.0 * math.pi * base_frequency * 1.5 * time + 0.2)
            + 0.025 * np.sin(2.0 * math.pi * base_frequency * 2.0 * time + 0.7)
        )
        noise = rng.normal(0.0, 0.002, sample_count)
        mono = np.asarray((tone * (0.55 + 0.45 * pulse) * slow) + noise, dtype=np.float32)
        stereo = np.stack([mono, np.roll(mono, 19)], axis=1)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        path = request.output_dir / f"fake_{request.seed}.wav"
        soundfile.write(path, stereo, self.sample_rate, subtype="FLOAT")
        return GenerationResult(
            audio_path=path.resolve(),
            seed=request.seed,
            final_latent=None,
            metadata={"backend": "fake", "sample_rate": self.sample_rate},
        )
