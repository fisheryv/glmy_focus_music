"""Safety wrapper for ACE-Step's native LoRA preprocessing/training entrypoint."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

RUNTIME_POLICY = {
    "schema_version": 1,
    "attention_backend": "sdpa",
    "flash_attention_disabled": True,
    "preprocess_cuda_launch_blocking": True,
    "preprocess_fail_fast": True,
    "cleanup_preserves_primary_cuda_error": True,
    "variant_alias_contract": "ace_step_1_5_all_official_variants_v1",
}

VARIANT_DIRECTORY_ALIASES = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
    "xl_turbo": "acestep-v15-xl-turbo",
    "xl_base": "acestep-v15-xl-base",
    "xl_sft": "acestep-v15-xl-sft",
}


def _install_variant_aliases() -> None:
    from acestep.training_v2.cli.args import VARIANT_DIR_MAP

    VARIANT_DIR_MAP.update(VARIANT_DIRECTORY_ALIASES)


def _install_safe_runtime(*, preprocess_mode: bool) -> Any:
    if preprocess_mode:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    _install_variant_aliases()

    import torch
    from acestep.training_v2 import model_loader
    from acestep.training_v2.cli import train_fixed

    model_loader._is_flash_attention_available = lambda _device: False

    def load_text_encoder(
        checkpoint_dir: str | Path,
        device: str = "cpu",
        precision: str = "bf16",
    ) -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        text_path = Path(checkpoint_dir) / "Qwen3-Embedding-0.6B"
        if not text_path.is_dir():
            raise FileNotFoundError(f"Text encoder directory not found: {text_path}")
        dtype = model_loader._resolve_dtype(precision)
        tokenizer = AutoTokenizer.from_pretrained(str(text_path))
        encoder = AutoModel.from_pretrained(
            str(text_path),
            attn_implementation="sdpa",
        )
        encoder = encoder.to(device=device, dtype=dtype)
        encoder.eval()
        return tokenizer, encoder

    model_loader.load_text_encoder = load_text_encoder

    original_unload = model_loader.unload_models

    def safe_unload(*models: Any) -> None:
        try:
            original_unload(*models)
        except torch.AcceleratorError as exc:
            gc.collect()
            print(
                f"[WARN] CUDA cleanup skipped after primary accelerator failure: {exc}",
                file=sys.stderr,
            )

    model_loader.unload_models = safe_unload

    def safe_cleanup_gpu() -> None:
        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except torch.AcceleratorError as exc:
            print(
                f"[WARN] CUDA cache cleanup skipped after primary accelerator failure: {exc}",
                file=sys.stderr,
            )

    train_fixed._cleanup_gpu = safe_cleanup_gpu

    if preprocess_mode:
        from acestep.training_v2 import preprocess

        original_error = preprocess.logger.error

        def fail_fast(message: str, *args: Any, **kwargs: Any) -> None:
            original_error(message, *args, **kwargs)
            try:
                rendered = message % args
            except (TypeError, ValueError):
                rendered = str(message)
            raise RuntimeError(rendered)

        preprocess.logger.error = fail_fast

    return train_fixed


def main() -> int:
    preprocess_mode = "--preprocess" in sys.argv[1:]
    train_fixed = _install_safe_runtime(preprocess_mode=preprocess_mode)
    return int(train_fixed.main())


if __name__ == "__main__":
    raise SystemExit(main())
