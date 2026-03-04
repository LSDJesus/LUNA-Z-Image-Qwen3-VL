"""
LunaVLMLoader — Load a GGUF model via llama-cpp-python for text encoding or VLM.

All model files (GGUF, mmproj, adapter) live in a single directory:
    models/LLM/LUNA-Qwen3-VL/

Files are auto-downloaded from HuggingFace on first use if not present locally.

Outputs a LLM_MODEL dict containing the loaded Llama instance and metadata,
consumed by LunaTextConditioner and LunaVLMChat nodes.
"""
from __future__ import annotations

import os
import logging
import folder_paths

logger = logging.getLogger("LUNA-VLM")

# ── Configuration ─────────────────────────────────────────────────────────────

HF_REPO_ID = "LSDJesus/LUNA-Qwen3-VL"

# Known model files — always shown in dropdowns, auto-downloaded on first use
DEFAULT_MODEL = "LUNA-Qwen3-VL.i1-Q4_K_M.gguf"
DEFAULT_MMPROJ = "LUNA-Qwen3-VL.mmproj-Q8_0.gguf"
DEFAULT_ADAPTER = "LUNA-Qwen3-VL_adapter.safetensors"

# ── Register folder path ─────────────────────────────────────────────────────

_luna_model_dir = os.path.join(folder_paths.models_dir, "LLM", "LUNA-Qwen3-VL")
os.makedirs(_luna_model_dir, exist_ok=True)

folder_paths.folder_names_and_paths["luna_qwen3_vl"] = (
    [_luna_model_dir],
    {".gguf", ".safetensors", ".pt"},
)


# ── HuggingFace Auto-Download ─────────────────────────────────────────────────

def _ensure_file(filename: str) -> str:
    """Return full path to filename, downloading from HuggingFace if missing."""
    # Check primary directory first
    full_path = os.path.join(_luna_model_dir, filename)
    if os.path.isfile(full_path):
        return full_path

    # Check all registered paths for this key
    try:
        found = folder_paths.get_full_path("luna_qwen3_vl", filename)
        if found and os.path.isfile(found):
            return found
    except Exception:
        pass

    # Auto-download from HuggingFace
    logger.info(f"Downloading {filename} from {HF_REPO_ID}...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=_luna_model_dir,
        )
        logger.info(f"  Downloaded to: {downloaded}")
        return downloaded
    except ImportError:
        raise RuntimeError(
            f"'{filename}' not found in {_luna_model_dir} and "
            f"huggingface_hub is not installed for auto-download.\n"
            f"Install with: pip install huggingface_hub\n"
            f"Or manually download from: https://huggingface.co/{HF_REPO_ID}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{filename}' from {HF_REPO_ID}: {e}\n"
            f"Please download manually to: {_luna_model_dir}"
        )


# ── File Discovery ────────────────────────────────────────────────────────────

def _get_luna_files(ext_filter: set[str] | None = None) -> list[str]:
    """List files in the LUNA model directory, optionally filtered by extension."""
    try:
        files = folder_paths.get_filename_list("luna_qwen3_vl")
        if ext_filter:
            files = [f for f in files if any(f.endswith(e) for e in ext_filter)]
        return sorted(set(files))
    except Exception:
        return []


def _get_gguf_models() -> list[str]:
    """Get GGUF model files (excluding mmproj)."""
    found = [f for f in _get_luna_files({".gguf"}) if "mmproj" not in f.lower()]
    if DEFAULT_MODEL not in found:
        found.insert(0, DEFAULT_MODEL)
    return found


def _get_mmproj_files() -> list[str]:
    """Get mmproj GGUF files."""
    found = [f for f in _get_luna_files({".gguf"}) if "mmproj" in f.lower()]
    result = ["none"]
    if DEFAULT_MMPROJ not in found:
        result.append(DEFAULT_MMPROJ)
    result.extend(found)
    return result


def _get_adapter_files() -> list[str]:
    """Get adapter checkpoint files (.safetensors / .pt)."""
    found = [f for f in _get_luna_files({".safetensors", ".pt"})
             if "adapter" in f.lower()]
    result = ["none"]
    if DEFAULT_ADAPTER not in found:
        result.append(DEFAULT_ADAPTER)
    result.extend(found)
    return result


class LunaVLMLoader:
    """Load a GGUF language model via llama-cpp-python with optional mmproj.

    Outputs a LLM_MODEL dict that can be consumed by:
      - LunaTextConditioner (penultimate hidden state extraction)
      - LunaVLMChat (multimodal text generation)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (_get_gguf_models(),
                               {"tooltip": "GGUF model file — auto-downloaded from HuggingFace if missing"}),
                "gpu_index": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1,
                                      "tooltip": "CUDA device index for the LLM"}),
                "n_ctx": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256,
                                  "tooltip": "Context window size in tokens"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1,
                                         "tooltip": "-1 = offload all layers to GPU"}),
            },
            "optional": {
                "mmproj_path": (_get_mmproj_files(),
                                {"tooltip": "mmproj file for vision (VLM Chat). Auto-downloaded if missing."}),
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
                "penultimate layer support from: "
                "https://github.com/LSDJesus/llama-cpp-python"
            )

        # Resolve full paths (auto-downloads from HuggingFace if missing)
        full_model_path = _ensure_file(model_path)

        full_mmproj_path = None
        if mmproj_path and mmproj_path != "none":
            full_mmproj_path = _ensure_file(mmproj_path)

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
