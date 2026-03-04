"""
LunaTextConditioner — Extract penultimate hidden states from a GGUF LLM
for Z-Image conditioning, with optional VL-to-Base adapter alignment.

Takes text input, runs it through the loaded GGUF model via llama-cpp-python,
extracts hidden_states[-2] (penultimate layer), optionally applies an adapter
to align variant models to the base distribution, and outputs standard
ComfyUI CONDITIONING format compatible with Z-Image / Lumina2.
"""
from __future__ import annotations

import os
import logging
import ctypes

import torch
import torch.nn as nn
import numpy as np

from .nodes_loader import _get_adapter_files, _ensure_file

logger = logging.getLogger("LUNA-VLM")

# ── Adapter Architecture ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Single residual MLP block: x + MLP(norm(x))."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        # Init output projection near zero → starts as identity
        nn.init.zeros_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class VLtoBaseAdapter(nn.Module):
    """Maps VL/Instruct hidden states to Base hidden state space."""

    def __init__(self, dim: int = 2560, hidden: int = 4096, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, hidden) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


# ── Adapter Cache ─────────────────────────────────────────────────────────────

_adapter_cache: dict[str, VLtoBaseAdapter] = {}


def _load_adapter(adapter_path: str, device: torch.device) -> VLtoBaseAdapter:
    """Load adapter weights from .safetensors or .pt checkpoint."""
    if adapter_path in _adapter_cache:
        return _adapter_cache[adapter_path]

    if adapter_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(adapter_path)
        config = {"dim": 2560, "hidden": 4096, "n_blocks": 2}  # default
    elif adapter_path.endswith(".pt"):
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        config = ckpt.get("config", {"dim": 2560, "hidden": 4096, "n_blocks": 2})
    else:
        raise ValueError(f"Unsupported adapter format: {adapter_path}")

    adapter = VLtoBaseAdapter(
        dim=config.get("dim", 2560),
        hidden=config.get("hidden", 4096),
        n_blocks=config.get("n_blocks", 2),
    )
    adapter.load_state_dict(state_dict)
    adapter = adapter.to(device).eval()

    _adapter_cache[adapter_path] = adapter
    logger.info(f"Loaded adapter: {os.path.basename(adapter_path)} "
                f"({sum(p.numel() for p in adapter.parameters()):,} params)")
    return adapter


# ── Node ──────────────────────────────────────────────────────────────────────

class LunaTextConditioner:
    """Extract penultimate layer hidden states from a GGUF LLM for Z-Image.

    Wraps text in the Qwen3 chat template, runs a forward pass through
    llama-cpp-python, extracts per-token hidden_states[-2], optionally
    applies a VL-to-Base adapter, and outputs CONDITIONING.
    """

    CHAT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", {"tooltip": "From LUNA VLM Loader"}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                    "tooltip": "Text prompt to encode"}),
            },
            "optional": {
                "adapter_path": (_get_adapter_files(),
                                 {"tooltip": "Optional VL-to-Base alignment adapter. "
                                  "Required for instruct/VL/abliterated model variants. "
                                  "Not needed for base Qwen3-4B GGUF."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "LUNA/VLM"
    TITLE = "LUNA Text Conditioner"

    def encode(self, llm_model: dict, text: str,
               adapter_path: str = "none"):
        import llama_cpp

        llm = llm_model["llm"]
        n_embd = llm_model["n_embd"]

        # ── Wrap in chat template ─────────────────────────────────────
        prompt = self.CHAT_TEMPLATE.format(text)

        # ── Tokenize ──────────────────────────────────────────────────
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        n_tokens = len(tokens)

        if n_tokens == 0:
            raise ValueError("Empty prompt after tokenization")

        logger.info(f"Encoding {n_tokens} tokens for conditioning")

        # ── Reset context and evaluate ────────────────────────────────
        llm.reset()
        llm._ctx.memory_clear(True)

        # Build batch with all tokens marked as output for embedding extraction
        batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
        try:
            for i, tok in enumerate(tokens):
                batch.token[i] = tok
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1  # Mark ALL positions as output
            batch.n_tokens = n_tokens

            ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed with code {ret}")
        finally:
            llama_cpp.llama_batch_free(batch)

        llm.n_tokens = n_tokens

        # ── Extract penultimate hidden states ─────────────────────────
        embeddings = []
        for i in range(n_tokens):
            ptr = llama_cpp.llama_get_embeddings_penultimate_ith(
                llm._ctx.ctx, ctypes.c_int32(i)
            )
            if ptr is None:
                raise RuntimeError(
                    f"llama_get_embeddings_penultimate_ith returned NULL for "
                    f"token {i}. Ensure the GGUF model was built with "
                    f"penultimate layer support (patched qwen3.cpp/qwen3vl.cpp)."
                )
            vec = np.ctypeslib.as_array(ptr, shape=(n_embd,)).copy()
            embeddings.append(vec)

        # [n_tokens, n_embd] float32
        cond_np = np.stack(embeddings, axis=0)
        cond = torch.from_numpy(cond_np).unsqueeze(0)  # [1, n_tokens, n_embd]

        # ── Apply adapter if requested ────────────────────────────────
        if adapter_path and adapter_path != "none":
            full_adapter_path = _ensure_file(adapter_path)

            device = torch.device(f"cuda:{llm_model['gpu_index']}")
            adapter = _load_adapter(full_adapter_path, device)

            with torch.no_grad():
                cond = adapter(cond.to(device)).cpu()

            logger.info(f"Applied adapter: {os.path.basename(adapter_path)}")

        # ── Build CONDITIONING output ─────────────────────────────────
        # Z-Image / Lumina2 conditioning format:
        #   cond tensor: [1, seq_len, n_embd]
        #   dict with attention_mask (all 1s for text-only)
        attention_mask = torch.ones(1, n_tokens, dtype=torch.long)

        conditioning = [[
            cond.to(torch.bfloat16),
            {
                "pooled_output": None,
                "attention_mask": attention_mask,
            }
        ]]

        return (conditioning,)
