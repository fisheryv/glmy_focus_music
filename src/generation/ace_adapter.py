from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .artifact_hash import sha256_directory
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
        self._device = self.config.device
        self._lora_path: Path | None = None
        self._lora_sha256: str | None = None
        self._lora_scale: float | None = None
        self._lora_enabled = False

    def set_topology_corrector(self, corrector: Any | None) -> None:
        """Install a qualified experimental corrector on the PyTorch ACE backend."""

        if corrector is not None and not callable(corrector):
            raise TypeError("topology corrector must be callable or None")
        self._topology_corrector = corrector
        if self._handler is not None:
            self._handler.set_topology_corrector(corrector)

    @staticmethod
    def _require_lora_success(message: Any, action: str) -> None:
        if not isinstance(message, str) or not message.startswith("✅"):
            raise RuntimeError(f"ACE-Step LoRA {action} failed: {message}")

    def load_lora(
        self,
        lora_path: Path,
        *,
        scale: float,
        expected_sha256: str,
    ) -> None:
        """Load one hash-bound LoRA adapter and enable it for inference."""

        resolved = lora_path.resolve()
        if self.config.quantization is not None:
            raise ValueError("topology LoRA inference forbids quantized ACE models")
        if not resolved.is_dir():
            raise FileNotFoundError(f"LoRA directory not found: {resolved}")
        if not 0.0 <= scale <= 1.0:
            raise ValueError("LoRA scale must lie in [0, 1]")
        actual_sha256 = sha256_directory(resolved)
        if actual_sha256 != expected_sha256:
            raise ValueError("LoRA directory hash differs from the expected artifact")
        self.initialize()
        assert self._handler is not None
        self._require_lora_success(self._handler.load_lora(str(resolved)), "load")
        self._require_lora_success(self._handler.set_lora_scale(scale), "scale")
        self._require_lora_success(self._handler.set_use_lora(True), "enable")
        self._lora_path = resolved
        self._lora_sha256 = actual_sha256
        self._lora_scale = float(scale)
        self._lora_enabled = True

    def set_lora_enabled(self, enabled: bool) -> None:
        """Toggle an already loaded LoRA without changing its frozen scale."""

        if self._handler is None or self._lora_path is None:
            raise RuntimeError("no LoRA adapter has been loaded")
        self._require_lora_success(
            self._handler.set_use_lora(bool(enabled)),
            "enable" if enabled else "disable",
        )
        self._lora_enabled = bool(enabled)

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
        device = os.environ.get("ACESTEP_DEVICE", self.config.device)
        self._device = device
        status, success = handler.initialize_service(
            project_root=str(self.checkout),
            config_path=self.config.model,
            device=device,
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
            "model_repository": self.config.model_repository,
            "device": self._device,
            "lora_enabled": self._lora_enabled,
            "lora_path": str(self._lora_path) if self._lora_path else None,
            "lora_sha256": self._lora_sha256,
            "lora_scale": self._lora_scale,
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
        device = torch.device(self._device)
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
