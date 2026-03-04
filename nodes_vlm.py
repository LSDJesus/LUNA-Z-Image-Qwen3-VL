"""
LunaVLMChat — Vision-language inference using a GGUF model with mmproj.

Takes an image and text prompt, runs multimodal inference through
llama-cpp-python's mtmd (multimodal) pipeline, and outputs generated text.

Use case: Look at a reference image, generate a descriptive prompt,
then feed that text (optionally combined with manual prompt) into
LunaTextConditioner or standard CLIPTextEncode for conditioning.
"""
from __future__ import annotations

import io
import os
import ctypes
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("LUNA-VLM")


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 to PIL."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]  # Take first in batch
    img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np, "RGB")


class LunaVLMChat:
    """Vision-language chat: feed an image + prompt to a GGUF VLM,
    get descriptive text back.

    Requires a model loaded with mmproj via LunaVLMLoader.
    Uses llama-cpp-python's mtmd multimodal pipeline for image tokenization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", {"tooltip": "From LUNA VLM Loader (must have mmproj)"}),
                "image": ("IMAGE", {"tooltip": "Input image for the VLM to describe"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail for use as an image generation prompt.",
                    "tooltip": "Instruction for the VLM. What should it say about the image?",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant that describes images concisely and accurately for use in AI image generation prompts.",
                    "tooltip": "System prompt for the VLM conversation",
                }),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32,
                                       "tooltip": "Maximum tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                                          "tooltip": "Sampling temperature (0 = greedy)"}),
                "prepend_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to prepend before VLM output in the final result",
                }),
                "append_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to append after VLM output in the final result",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat"
    CATEGORY = "LUNA/VLM"
    TITLE = "LUNA VLM Chat"
    OUTPUT_NODE = False

    def chat(self, llm_model: dict, image: torch.Tensor, prompt: str,
             system_prompt: str = "", max_tokens: int = 256,
             temperature: float = 0.7, prepend_text: str = "",
             append_text: str = ""):

        if not llm_model.get("has_mmproj"):
            raise RuntimeError(
                "VLM Chat requires an mmproj file. Reload the model with "
                "LunaVLMLoader and provide an mmproj_path."
            )

        import llama_cpp
        from llama_cpp import mtmd_cpp

        llm = llm_model["llm"]
        mmproj_path = llm_model["mmproj_path"]

        # ── Initialize mtmd context if needed ─────────────────────────
        # The mtmd context handles image tokenization via the mmproj
        mtmd_params = mtmd_cpp.mtmd_context_params_default()
        mtmd_params.use_gpu = True
        mtmd_params.n_threads = max(1, os.cpu_count() // 2)
        mtmd_params.verbosity = 0

        mtmd_ctx = mtmd_cpp.mtmd_init_from_file(
            llm._model.model,
            llm._ctx.ctx,
            mmproj_path.encode("utf-8"),
            mtmd_params,
        )
        if mtmd_ctx is None:
            raise RuntimeError(f"Failed to initialize mtmd context from {mmproj_path}")

        try:
            # ── Convert image to bytes ────────────────────────────────
            pil_image = _tensor_to_pil(image)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # ── Create bitmap from image bytes ────────────────────────
            bitmap = mtmd_cpp.mtmd_bitmap_init_from_memory(
                img_bytes, len(img_bytes)
            )
            if bitmap is None:
                raise RuntimeError("Failed to create mtmd bitmap from image")

            try:
                # ── Build chat prompt with image marker ───────────────
                media_marker = mtmd_cpp.mtmd_default_marker().decode("utf-8")

                if system_prompt:
                    chat_text = (
                        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                        f"<|im_start|>user\n{media_marker}\n{prompt}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                else:
                    chat_text = (
                        f"<|im_start|>user\n{media_marker}\n{prompt}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )

                # ── Tokenize text + image ─────────────────────────────
                input_text = mtmd_cpp.mtmd_input_text()
                input_text.text = chat_text.encode("utf-8")
                input_text.add_special = True
                input_text.parse_special = True

                bitmap_array = (mtmd_cpp.mtmd_bitmap_p_ctypes * 1)(bitmap)
                chunks = mtmd_cpp.mtmd_input_chunks_init()
                if chunks is None:
                    raise RuntimeError("Failed to create mtmd input chunks")

                try:
                    ret = mtmd_cpp.mtmd_tokenize(
                        mtmd_ctx, chunks,
                        ctypes.byref(input_text),
                        bitmap_array, 1,
                    )
                    if ret != 0:
                        raise RuntimeError(f"mtmd_tokenize failed: {ret}")

                    # ── Reset context and decode chunks ───────────────
                    llm.reset()
                    llm._ctx.memory_clear(True)
                    n_past = 0
                    n_chunks = mtmd_cpp.mtmd_input_chunks_size(chunks)

                    for i in range(n_chunks):
                        chunk = mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                        if chunk is None:
                            continue

                        ctype = mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                        if ctype == mtmd_cpp.mtmd_input_chunk_type.MTMD_INPUT_CHUNK_TYPE_TEXT:
                            n_tok_out = ctypes.c_size_t(0)
                            tok_ptr = mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                                chunk, ctypes.byref(n_tok_out)
                            )
                            if tok_ptr and n_tok_out.value > 0:
                                tokens = [tok_ptr[j] for j in range(n_tok_out.value)]
                                batch = llama_cpp.llama_batch_init(len(tokens), 0, 1)
                                for k, tok in enumerate(tokens):
                                    batch.token[k] = tok
                                    batch.pos[k] = n_past + k
                                    batch.n_seq_id[k] = 1
                                    batch.seq_id[k][0] = 0
                                    batch.logits[k] = 1 if k == len(tokens) - 1 else 0
                                batch.n_tokens = len(tokens)

                                ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
                                llama_cpp.llama_batch_free(batch)
                                if ret != 0:
                                    raise RuntimeError(f"llama_decode failed: {ret}")
                                n_past += len(tokens)

                        else:  # IMAGE or AUDIO chunk
                            new_n_past = llama_cpp.llama_pos(0)
                            ret = mtmd_cpp.mtmd_helper_eval_chunk_single(
                                mtmd_ctx, llm._ctx.ctx, chunk,
                                llama_cpp.llama_pos(n_past),
                                llama_cpp.llama_seq_id(0),
                                llm.n_batch,
                                True,  # logits_last
                                ctypes.byref(new_n_past),
                            )
                            if ret != 0:
                                raise RuntimeError(f"mtmd_helper_eval_chunk_single failed: {ret}")
                            n_past = new_n_past.value

                    llm.n_tokens = n_past

                    # ── Generate text tokens ──────────────────────────
                    output_tokens = []
                    eos_token = llm.token_eos()

                    for _ in range(max_tokens):
                        logits_ptr = llm._ctx.get_logits()
                        logits = np.ctypeslib.as_array(
                            logits_ptr, shape=(llm._n_vocab,)
                        ).copy()

                        if temperature <= 0.01:
                            # Greedy
                            token = int(np.argmax(logits))
                        else:
                            # Temperature sampling
                            logits = logits / temperature
                            logits -= np.max(logits)  # numerical stability
                            probs = np.exp(logits)
                            probs /= probs.sum()
                            token = int(np.random.choice(len(probs), p=probs))

                        if token == eos_token:
                            break

                        output_tokens.append(token)
                        llm.eval([token])

                    generated_text = llm.detokenize(output_tokens).decode(
                        "utf-8", errors="replace"
                    ).strip()

                finally:
                    mtmd_cpp.mtmd_input_chunks_free(chunks)

            finally:
                mtmd_cpp.mtmd_bitmap_free(bitmap)

        finally:
            mtmd_cpp.mtmd_free(mtmd_ctx)

        # ── Combine with prepend/append ───────────────────────────────
        parts = []
        if prepend_text.strip():
            parts.append(prepend_text.strip())
        parts.append(generated_text)
        if append_text.strip():
            parts.append(append_text.strip())

        result = " ".join(parts)
        logger.info(f"VLM generated {len(output_tokens)} tokens: {result[:100]}...")

        return (result,)
