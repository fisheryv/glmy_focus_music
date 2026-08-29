from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .experiment import AceConfig


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    seed: int
    duration_seconds: float
    output_dir: Path
    inference_steps: int = 8
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    audio_path: Path
    seed: int
    final_latent: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class AceStepAdapter:
    """Version-bound adapter for the local ACE-Step 1.5 Python inference API."""

    def __init__(self, checkout: Path, config: AceConfig):
        self.checkout = checkout.resolve()
        self.config = config
        if not (self.checkout / "pyproject.toml").is_file():
            raise FileNotFoundError(f"ACE-Step checkout not found at {self.checkout}")
        self._handler: Any | None = None
        self._api: tuple[Any, Any, Any] | None = None
        self._topology_corrector: Any | None = None

    def set_topology_corrector(self, corrector: Any | None) -> None:
        """Install a qualified experimental corrector on the PyTorch ACE backend."""

        if corrector is not None and not callable(corrector):
            raise TypeError("topology corrector must be callable or None")
        self._topology_corrector = corrector
        if self._handler is not None:
            self._handler.set_topology_corrector(corrector)

    def _import_api(self) -> tuple[Any, Any, Any, Any]:
        checkout_text = str(self.checkout)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
        handler_module = importlib.import_module("acestep.handler")
        inference_module = importlib.import_module("acestep.inference")
        module_path = Path(handler_module.__file__).resolve()
        if self.checkout not in module_path.parents:
            raise RuntimeError(
                f"imported ACE-Step from {module_path}, expected checkout {self.checkout}"
            )
        return (
            handler_module.AceStepHandler,
            inference_module.GenerationParams,
            inference_module.GenerationConfig,
            inference_module.generate_music,
        )

    def initialize(self) -> None:
        if self._handler is not None:
            return
        handler_cls, params_cls, generation_config_cls, generate_music = self._import_api()
        handler = handler_cls()
        status, success = handler.initialize_service(
            project_root=str(self.checkout),
            config_path=self.config.model,
            device=self.config.device,
            compile_model=self.config.compile_model,
            offload_to_cpu=self.config.offload_to_cpu,
            offload_dit_to_cpu=self.config.offload_dit_to_cpu,
            quantization=self.config.quantization,
            prefer_source=self.config.prefer_source,
            use_mlx_dit=False,
            vae_checkpoint=self.config.vae_checkpoint,
        )
        if not success:
            raise RuntimeError(f"ACE-Step initialization failed: {status}")
        handler.set_topology_corrector(self._topology_corrector)
        self._handler = handler
        self._api = (params_cls, generation_config_cls, generate_music)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.initialize()
        assert self._handler is not None and self._api is not None
        params_cls, generation_config_cls, generate_music = self._api
        request.output_dir.mkdir(parents=True, exist_ok=True)
        params = params_cls(
            caption=request.prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            bpm=request.bpm,
            keyscale=request.keyscale,
            timesignature=request.timesignature,
            duration=request.duration_seconds,
            inference_steps=request.inference_steps,
            seed=request.seed,
            guidance_scale=self.config.guidance_scale,
            shift=self.config.shift,
            infer_method=self.config.infer_method,
            sampler_mode=self.config.sampler_mode,
            dcw_enabled=self.config.dcw_enabled,
            dcw_mode=self.config.dcw_mode,
            dcw_scaler=self.config.dcw_scaler,
            dcw_high_scaler=self.config.dcw_high_scaler,
            dcw_wavelet=self.config.dcw_wavelet,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_lyrics=False,
            use_cot_language=False,
            enable_normalization=False,
        )
        generation_config = generation_config_cls(
            batch_size=1,
            use_random_seed=False,
            seeds=[request.seed],
            audio_format="wav32",
        )
        result = generate_music(
            self._handler,
            None,
            params,
            generation_config,
            save_dir=str(request.output_dir),
        )
        if not result.success or len(result.audios) != 1:
            message = result.error or result.status_message or "ACE-Step generation failed"
            raise RuntimeError(message)
        audio = result.audios[0]
        audio_path = Path(audio.get("path") or "")
        if not audio_path.is_file():
            raise RuntimeError("ACE-Step returned no saved audio file")
        extra = result.extra_outputs or {}
        metadata = {
            "audio_key": audio.get("key", ""),
            "sample_rate": audio.get("sample_rate"),
            "time_costs": extra.get("time_costs", {}),
            "model": self.config.model,
            "device": self.config.device,
        }
        return GenerationResult(
            audio_path=audio_path.resolve(),
            seed=request.seed,
            final_latent=extra.get("pred_latents"),
            metadata=metadata,
        )

    def decode_latent_to_audio(self, latent: Any, output_path: Path) -> Path:
        """Decode one recorded ``[T,64]`` x0 estimate with the initialized ACE VAE."""

        self.initialize()
        assert self._handler is not None
        try:
            import numpy as np
            import soundfile
            import torch
        except ImportError as exc:
            raise RuntimeError("snapshot decoding requires torch, numpy, and soundfile") from exc
        values = np.asarray(latent, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 64 or not np.isfinite(values).all():
            raise ValueError("recorded snapshot latent must have finite shape [T,64]")
        device = torch.device(self.config.device)
        tensor = torch.from_numpy(values).unsqueeze(0).to(device)
        waveforms, _, _ = self._handler._decode_generate_music_pred_latents(
            pred_latents=tensor,
            progress=None,
            use_tiled_decode=True,
            time_costs={"total_time_cost": 0.0},
        )
        audio = waveforms[0].detach().float().cpu().numpy().T
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".part.wav")
        soundfile.write(str(temporary), audio, int(self._handler.sample_rate), subtype="FLOAT")
        temporary.replace(output_path)
        return output_path.resolve()
