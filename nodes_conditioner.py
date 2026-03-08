"""
LunaTextConditioner — Extract penultimate hidden states from a GGUF LLM
for Z-Image conditioning, with automatic VL-to-Base adapter alignment.

Takes text input, runs it through the loaded GGUF model via llama-cpp-python,
extracts hidden_states[-2] (penultimate layer), automatically applies the
matched adapter (loaded by the VLM Loader), and outputs standard ComfyUI
CONDITIONING format compatible with Z-Image / Lumina2.
"""
from __future__ import annotations

import os
import sys
import logging
import ctypes
import contextlib

import torch
import numpy as np

logger = logging.getLogger("LUNA-VLM")


@contextlib.contextmanager
def _suppress_c_output():
    """Suppress C-level stdout/stderr (llama.cpp decode spam)."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


# ── Node ──────────────────────────────────────────────────────────────────────

class LunaTextConditioner:
    """Extract penultimate layer hidden states from a GGUF LLM for Z-Image.

    Wraps text in the Qwen3 chat template, runs a forward pass through
    llama-cpp-python, extracts per-token hidden_states[-2], automatically
    applies the matched adapter (from the loader), and outputs CONDITIONING.
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
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "LUNA/VLM"
    TITLE = "LUNA Text Conditioner"

    def encode(self, llm_model: dict, text: str):
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

            with _suppress_c_output():
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

        # ── Apply adapter if loaded by the VLM Loader ────────────────
        adapter = llm_model.get("adapter")
        if adapter is not None:
            device = torch.device(f"cuda:{llm_model['gpu_index']}")
            with torch.no_grad():
                cond = adapter(cond.to(device)).cpu()
            logger.info(f"Applied adapter: {llm_model.get('adapter_name', 'unknown')}")

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
