# LUNA Z-Image Qwen3-VL

ComfyUI custom nodes for using **GGUF quantized Qwen3 / Qwen3-VL** models as text encoders for Z-Image, and for vision-language (VLM) inference.

## What This Does

- **LUNA VLM Loader** — Load a GGUF model (+ optional mmproj for vision) via llama-cpp-python
- **LUNA Text Conditioner** — Extract penultimate hidden states (`hidden_states[-2]`) from the GGUF model for Z-Image conditioning, with optional adapter alignment for non-base model variants
- **LUNA VLM Chat** — Feed an image to the VLM, get descriptive text back for prompt engineering

### Why Not Just Use CLIPLoaderGGUF?

**You can.** For most users, `CLIPLoaderGGUF` → `Qwen3-4B.i1-Q4_K_M.gguf` → type `lumina2` → `CLIPTextEncode` works perfectly.

These nodes add value in two cases:
1. **Adapter alignment** — If you're using an instruct, VL, or abliterated model variant instead of the base Qwen3-4B, the weight distribution has drifted. The built-in adapter realigns the hidden states to match what Z-Image's denoiser expects.
2. **VLM inference** — Use the same GGUF model (with mmproj) to *look at* images and generate text descriptions, which you can then feed into conditioning.

### VRAM Savings

| Encoder | Size | VRAM |
|---------|------|------|
| Qwen3-4B safetensors (bf16) | 7.67 GB | ~8.2 GB |
| Qwen3-4B GGUF (Q4_K_M) | 2.4 GB | ~2.5 GB |
| Qwen3-VL-4B GGUF + adapter | 2.4 GB + 160 MB | ~2.7 GB |

## Requirements

**llama-cpp-python** with penultimate layer support. Install the pre-built wheel:

### Windows (CUDA)
```bash
pip install llama-cpp-python --extra-index-url https://github.com/JamePeng/llama-cpp-python-luna/releases/latest
```

### Linux (CUDA)
```bash
pip install llama-cpp-python --extra-index-url https://github.com/JamePeng/llama-cpp-python-luna/releases/latest
```

### From Source (Advanced)
```bash
# Requires CUDA Toolkit + Visual Studio Build Tools (Windows) or gcc (Linux)
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-binary llama-cpp-python
```

## Models

| File | Purpose | Where to Put It |
|------|---------|-----------------|
| `Qwen3-4B.i1-Q4_K_M.gguf` | Text encoder (base) | `ComfyUI/models/text_encoders/` |
| `Qwen3-VL-4B-*.gguf` | Text encoder (VL variant) | `ComfyUI/models/text_encoders/` |
| `*.mmproj-Q8_0.gguf` | Vision projector (for VLM Chat) | `ComfyUI/models/text_encoders/` |
| `vl_to_base_adapter_best.safetensors` | Adapter (for VL variants) | `ComfyUI/models/text_encoders/` |

## Workflow Examples

### Basic Text Encoding (GGUF replaces safetensors)
```
[LUNA VLM Loader] → [LUNA Text Conditioner] → CONDITIONING → [KSampler]
  ├─ model: Qwen3-4B.i1-Q4_K_M.gguf
  └─ adapter: none (not needed for base model)
```

### VL Variant with Adapter
```
[LUNA VLM Loader] → [LUNA Text Conditioner] → CONDITIONING → [KSampler]
  ├─ model: Qwen3-VL-4B-abliterated-Q4_K_M.gguf
  └─ adapter: vl_to_base_adapter_best.safetensors
```

### VLM Image Description → Conditioning
```
[Load Image] ──────────────┐
                            ▼
[LUNA VLM Loader] → [LUNA VLM Chat] → TEXT ─┐
  ├─ model: Qwen3-VL-4B-*.gguf              │
  └─ mmproj: *.mmproj-Q8_0.gguf             ▼
                              [String Concatenate] → [CLIPTextEncode / LUNA Text Conditioner] → CONDITIONING
                                      ▲
                    [manual prompt] ───┘
```

## License

MIT
