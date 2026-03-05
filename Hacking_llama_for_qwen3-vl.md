# The Qwen3-VL GGUF Encoder — How We Got Here

> A technical deep-dive into extracting penultimate hidden states from GGUF models via a custom llama.cpp fork, building a VL-to-Base alignment adapter, and achieving a 3.3× VRAM reduction with vision capabilities.

---

## The Starting Point

[Z-Image](https://github.com/city96/ComfyUI-GGUF) uses Qwen3-4B as its text encoder — but not the way most people use LLMs. We don't care about token predictions. We extract the **penultimate hidden states** (`hidden_states[-2]`) after the full 36-layer transformer stack processes the prompt. The denoiser receives `[B, seq, 2560]` — the model's deep semantic understanding of the text, not raw token features or logits. This is what Z-Image was trained against.

The standard encoder is a 7.67 GB bf16 safetensors file loaded through HuggingFace Transformers. It works perfectly but consumes ~8.2 GB of VRAM.

## The Goal

The original hypothesis was that **Qwen3-VL** (the Vision-Language variant) could give us face conditioning — feed a reference face image alongside the text prompt, and the hidden states would encode a unified representation of both. Character consistency without LoRA or IP-Adapter. But Qwen3-VL in HF bf16 would be *even larger* than the base model.

The dream: a **2.4 GB Q4_K_M GGUF quantized** Qwen3-VL model that gives us both text encoding AND vision conditioning, replacing the 8.2 GB safetensors.

## The Problem: llama.cpp Doesn't Expose Hidden Layers

llama.cpp is built for text generation — it gives you logits and final-layer embeddings. That's it. But Z-Image needs the *penultimate* layer (the second-to-last transformer block output, before the final LayerNorm and LM head projection).

Why specifically the penultimate layer? It carries richer, less-collapsed representations that the denoiser was trained on. The final layer has been optimized for next-token prediction and loses spatial/stylistic nuance.

No existing API function in llama.cpp gave access to intermediate layer hidden states.

## The Solution: A Custom llama-cpp-python Fork

We didn't hack around it — we went deep. A proper C++ implementation across the full llama.cpp stack:

| Layer | Change |
|---|---|
| `llama-graph.h` | Added `t_embd_penultimate` field to `llm_graph_result` |
| `llama-graph.cpp` | Marked it as output in `set_outputs()` |
| `qwen3vl.cpp` | Stored `inpL` (the residual stream) into `t_embd_penultimate` *before* the final transformer block |
| `llama-context.h` | Added `embd_penultimate` buffer + getter declaration |
| `llama-context.cpp` | Allocated GPU→CPU copy buffer, implemented the getter |
| `llama.h` | Declared `llama_get_embeddings_penultimate_ith()` — the new C API function |
| `llama_cpp/llama_cpp.py` | ctypes binding for the new function |
| `llama_cpp/_internals.py` | `LlamaContext.get_embeddings_penultimate_ith()` wrapper |
| `llama_cpp/llama.py` | High-level `get_penultimate_embeddings()` helper on the `Llama` class |

The key insight in the C++ change: in `qwen3vl.cpp`'s graph builder, right before the final transformer block processes the residual stream, we capture `inpL` into `t_embd_penultimate`. This is the exact equivalent of HuggingFace's `hidden_states[-2]` — same data, same position in the computation graph.

Three critical flags unlock per-token extraction from the Python side:
1. `embeddings=True` — tells `llama_decode` to store hidden states in the output buffer
2. `pooling_type=LLAMA_POOLING_TYPE_NONE` — per-token storage, no mean/CLS pooling
3. Manual batch construction marking all token positions as output

The fork is available at [LSDJesus/llama-cpp-python](https://github.com/LSDJesus/llama-cpp-python) with pre-built CUDA wheels for Windows and Linux.

## Verification

The first proof-of-life test loaded the GGUF model, ran a prompt, and extracted both final and penultimate layer embeddings for every token position. It confirmed:
- Penultimate embeddings are non-null, non-zero
- They're distinctly *different* from the final layer (max abs diff >> 0, cosine similarity < 1.0)
- They have the correct shape: `[n_tokens, 2560]`

A 5-way comparison test then mapped the distribution drift across model variants:

1. **HF Qwen3-4B base** (bf16) — the gold standard Z-Image was trained against
2. **GGUF Qwen3-4B base** (Q4_K_M) — quantization error only
3. **GGUF Qwen3-4B-Instruct** (Q4_K_M) — instruct tuning shift
4. **HF Qwen3-VL-4B** (bf16) — VL fine-tuning shift
5. **GGUF Qwen3-VL-4B** (Q4_K_M) — full chain: quantization + instruct + VL

This revealed the cosine similarity gap between the GGUF VL embeddings and the HF Base embeddings that the denoiser expects. Close, but not close enough for direct substitution — which led to the adapter.

## The VL-to-Base Alignment Adapter

### The Problem

The VL/Instruct variants of Qwen3 have drifted from the base model's hidden state distribution during fine-tuning. Feeding raw VL hidden states into a denoiser trained on base model outputs produces degraded or garbled images.

### The Architecture

A lightweight residual MLP stack (~42M parameters, ~160 MB on disk):

```
VLtoBaseAdapter:
  2× ResidualBlock:
    LayerNorm(2560) → Linear(2560→4096) → GELU → Dropout → Linear(4096→2560) → Dropout
    + residual connection: x + MLP(norm(x))
  Final LayerNorm(2560)
```

Key design choices:
- **Residual initialization** — output projection initialized to zeros, so the untrained adapter starts as an identity function. Training refines from there.
- **Loss function**: MSE + 0.5× cosine distance — pushes both absolute values and directional alignment toward the target
- **AdamW** with OneCycleLR cosine schedule, gradient clipping at 1.0

### Training

We generated paired training data by encoding ~5,000 diverse prompts through *both* encoders:
- Qwen3-4B HF base (safetensors, bf16) → ground truth embeddings
- Qwen3-VL GGUF (Q4_K_M) → input embeddings

Each prompt produces a sequence of `[seq, 2560]` token embeddings. Non-padding tokens are extracted and paired token-by-token, yielding hundreds of thousands of `(vl_embedding, base_embedding)` training pairs.

### Result

The trained adapter achieved **0.979 cosine similarity** to the base encoder on the validation set (best checkpoint at epoch 4). Visual evaluation confirmed that images generated with adapter-aligned VL embeddings closely match those from the original base encoder.

## What Didn't Work: Direct Vision Conditioning

Here's the honest part: **the original face-ID hypothesis didn't pan out.**

Feeding vision embeddings directly into Z-Image's denoiser — either alone or interleaved with text embeddings — produces garbage. The denoiser was trained exclusively on text-derived hidden states from the base Qwen3-4B encoder. It has no concept of visual tokens. The vision embeddings from Qwen3-VL's mmproj occupy a completely different region of the representation space, and the denoiser simply doesn't know what to do with them.

What *does* work is using the VLM as a **description engine**: feed it a reference face image, have it generate a detailed text description of the person's features, then combine that description with the rest of your prompt and run it through the text encoding path. It's not true face-ID — it's more like automated prompt engineering from a reference image. But it produces surprisingly good results for character consistency, and the entire pipeline (VLM chat → text conditioning) is what the LUNA VLM Chat node is designed for.

## The Numbers

| | HF Qwen3-4B (safetensors) | GGUF Qwen3-VL (Q4_K_M + adapter) |
|---|---|---|
| **Model size** | 7.67 GB | 2.4 GB + 160 MB = **2.56 GB** |
| **VRAM** | ~8.2 GB | ~2.5 GB |
| **Cosine to base** | 1.000 (it IS the base) | 0.979 (after adapter) |
| **Vision capability** | None | Text-based (VLM describe → re-encode) |
| **Size reduction** | — | **3.3× smaller** |

## Summary

From "llama.cpp doesn't expose hidden layers" to a custom C API, a penultimate layer extraction function, a 42M-parameter alignment adapter, and a 3.3× VRAM reduction. The original vision-embedding dream didn't survive contact with reality, but the VLM describe-then-encode workflow turned out to be a practical and effective alternative for character consistency.

The entire stack — custom llama.cpp fork, pre-built wheels, adapter weights, and these ComfyUI nodes — is open source under MIT.

## Future Work

Some things we haven't tried yet:
- **Lower quantizations** — Q3_K or Q2_K GGUF models could push the size down even further. The adapter was only trained against Q4_K_M outputs, but lower quants might still land close enough (or the adapter could be retrained).
- **Quantized mmproj** — The mmproj is currently Q8_0 (~600 MB). A Q4 mmproj would cut that significantly for the VLM chat path, with unknown impact on description quality.
- **Adapter-free base model** — If you're using the base Qwen3-4B GGUF (not VL), the adapter isn't needed at all. We haven't benchmarked lower quants of the base model for conditioning quality.