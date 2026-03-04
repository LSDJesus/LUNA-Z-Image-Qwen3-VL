"""
LunaVLMLoader — Load a GGUF model via llama-cpp-python for text encoding or VLM.

Outputs a LLM_MODEL dict containing the loaded Llama instance and metadata,
consumed by LunaTextConditioner and LunaVLMChat nodes.
"""
from __future__ import annotations

import os
import logging
import folder_paths

logger = logging.getLogger("LUNA-VLM")

# Register custom folder for GGUF LLM models and mmproj files
for key, targets in [
    ("luna_llm_gguf", ["diffusion_models", "unet"]),
    ("luna_mmproj", ["clip", "text_encoders"]),
    ("luna_adapter", ["clip", "text_encoders"]),
]:
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base_dirs = base[0] if isinstance(base[0], (list, set, tuple)) else []
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base_dirs, {".gguf"})


def _get_gguf_files(key: str) -> list[str]:
    """Get .gguf files from a registered folder path."""
    try:
        return [f for f in folder_paths.get_filename_list(key) if f.endswith(".gguf")]
    except Exception:
        return []


def _get_adapter_files() -> list[str]:
    """Get adapter files (.safetensors / .pt) from adapter folder paths."""
    results = ["none"]
    for key in ["luna_adapter", "clip", "text_encoders"]:
        try:
            for f in folder_paths.get_filename_list(key):
                if f.endswith((".safetensors", ".pt")) and "adapter" in f.lower():
                    if f not in results:
                        results.append(f)
        except Exception:
            pass
    return results


class LunaVLMLoader:
    """Load a GGUF language model via llama-cpp-python with optional mmproj.

    Outputs a LLM_MODEL dict that can be consumed by:
      - LunaTextConditioner (penultimate hidden state extraction)
      - LunaVLMChat (multimodal text generation)
    """

    @classmethod
    def INPUT_TYPES(cls):
        gguf_files = _get_gguf_files("luna_llm_gguf")
        if not gguf_files:
            # Fallback: scan all registered GGUF paths
            gguf_files = _get_gguf_files("clip_gguf") + _get_gguf_files("unet_gguf")
        gguf_files = sorted(set(gguf_files)) or ["(no .gguf files found)"]

        mmproj_files = ["none"] + sorted(set(
            f for f in _get_gguf_files("luna_mmproj")
            if "mmproj" in f.lower()
        ))

        return {
            "required": {
                "model_path": (gguf_files, {"tooltip": "GGUF model file (e.g. Qwen3-4B.i1-Q4_K_M.gguf)"}),
                "gpu_index": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1,
                                      "tooltip": "CUDA device index for the LLM"}),
                "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256,
                                  "tooltip": "Context window size in tokens"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1,
                                         "tooltip": "-1 = offload all layers to GPU"}),
            },
            "optional": {
                "mmproj_path": (mmproj_files, {"tooltip": "Optional mmproj file for vision (VLM Chat only)"}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "load_model"
    CATEGORY = "LUNA/VLM"
    TITLE = "LUNA VLM Loader (GGUF)"

    def load_model(self, model_path: str, gpu_index: int = 0,
                   n_ctx: int = 2048, n_gpu_layers: int = -1,
                   mmproj_path: str = "none"):
        try:
            from llama_cpp import Llama
            import llama_cpp
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the LUNA fork with "
                "penultimate layer support: pip install llama-cpp-python from "
                "https://github.com/JamePeng/llama-cpp-python"
            )

        # Resolve full paths
        full_model_path = folder_paths.get_full_path("luna_llm_gguf", model_path)
        if full_model_path is None:
            # Try other registered paths
            for key in ["clip", "clip_gguf", "unet_gguf", "diffusion_models"]:
                full_model_path = folder_paths.get_full_path(key, model_path)
                if full_model_path:
                    break
        if not full_model_path or not os.path.isfile(full_model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        full_mmproj_path = None
        if mmproj_path and mmproj_path != "none":
            full_mmproj_path = folder_paths.get_full_path("luna_mmproj", mmproj_path)
            if full_mmproj_path is None:
                for key in ["clip", "clip_gguf", "text_encoders"]:
                    full_mmproj_path = folder_paths.get_full_path(key, mmproj_path)
                    if full_mmproj_path:
                        break
            if not full_mmproj_path or not os.path.isfile(full_mmproj_path):
                raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")

        logger.info(f"Loading GGUF model: {model_path} on cuda:{gpu_index}")
        if full_mmproj_path:
            logger.info(f"  mmproj: {mmproj_path}")

        # Build tensor_split for multi-GPU: all weight on target GPU
        tensor_split = [0.0] * 8
        tensor_split[gpu_index] = 1.0

        llm = Llama(
            model_path=full_model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            main_gpu=gpu_index,
            tensor_split=tensor_split,
            embeddings=True,  # Required for penultimate layer extraction
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE,  # Per-token embeddings
            verbose=False,
        )

        n_embd = llm.n_embd()
        n_vocab = llm._n_vocab
        model_name = os.path.basename(model_path)

        logger.info(f"  Loaded: n_embd={n_embd}, n_vocab={n_vocab}, n_ctx={n_ctx}")

        result = {
            "llm": llm,
            "model_path": full_model_path,
            "mmproj_path": full_mmproj_path,
            "model_name": model_name,
            "n_embd": n_embd,
            "gpu_index": gpu_index,
            "has_mmproj": full_mmproj_path is not None,
        }

        return (result,)
