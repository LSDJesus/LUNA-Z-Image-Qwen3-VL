# LUNA Z-Image Qwen3-VL V0.1.2

ComfyUI custom nodes for using **GGUF quantized Qwen3-VL** as a text encoder for [Z-Image](https://github.com/city96/ComfyUI-GGUF), with built-in vision-language (VLM) chat for image-to-prompt generation.

**All models, adapters, and projectors are auto-downloaded from [HuggingFace](https://huggingface.co/LSDJesus/LUNA-Qwen3-VL) on first use.**

> **[How We Got Here](Hacking_llama_for_qwen3-vl.md)** — Technical deep-dive into hacking llama.cpp for penultimate hidden state extraction, building the VL-to-Base adapter, and achieving 3.3× VRAM reduction with vision capabilities.

---

## Nodes

| Node | Description |
|------|-------------|
| **LUNA VLM Loader** | Load a GGUF model + optional mmproj for vision. Automatically detects and loads the matching adapter. |
| **LUNA Text Conditioner** | Extract penultimate hidden states for Z-Image conditioning. Adapter is applied automatically if loaded. |
| **LUNA VLM Chat** | Feed an image to the VLM and get descriptive text back. Includes repetition penalty for clean output. |

---

## Installation

### Via ComfyUI Manager (Recommended)
Search for **LUNA Z-Image Qwen3-VL** in ComfyUI Manager and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LSDJesus/LUNA-Z-Image-Qwen3-VL.git
cd LUNA-Z-Image-Qwen3-VL
pip install -r requirements.txt
```

### Dependencies

Requires a custom fork of **llama-cpp-python** with penultimate hidden state extraction support. The correct wheel is installed automatically via `requirements.txt`:

> **Note:** These wheels are built from the [LUNA fork](https://github.com/LSDJesus/llama-cpp-python) with penultimate hidden state support. Standard llama-cpp-python will **not** work for text conditioning.

| Platform | Wheel |
|----------|-------|
| Windows (CUDA) | [llama_cpp_python-0.3.27 (win_amd64)](https://github.com/LSDJesus/llama-cpp-python/releases/download/v0.3.27-cuda-6396cf3/llama_cpp_python-0.3.27-cp39-abi3-win_amd64.whl) |
| Linux (CUDA) | [llama_cpp_python-0.3.27 (manylinux)](https://github.com/LSDJesus/llama-cpp-python/releases/download/v0.3.27-cuda-6396cf3/llama_cpp_python-0.3.27-cp39-abi3-manylinux_2_28_x86_64.whl) |

---

## Models

All files are stored in `ComfyUI/models/LLM/LUNA-Qwen3-VL/` and auto-downloaded from [HuggingFace](https://huggingface.co/LSDJesus/LUNA-Qwen3-VL) on first use.

### LLM Quantizations

| Model | Size | Conditioning | VLM Chat | Notes |
|-------|------|:---:|:---:|-------|
| `LUNA-Qwen3-VL.i1-Q4_K_M.gguf` | 2.4 GB | ✅ | ✅ | **Recommended.** Best quality-to-size ratio. |
| `LUNA-Qwen3-VL.i1-Q3_K_S.gguf` | 1.8 GB | ✅ | ✅ | Good balance for 6 GB cards. |
| `LUNA-Qwen3-VL.i1-Q2_K_S.gguf` | 1.5 GB | ✅ | ✅ | Minimum for reliable VLM chat. |
| `LUNA-Qwen3-VL.i1-IQ2_XXS.gguf` | 1.2 GB | ✅ | ✅ | Lowest quant with working VLM chat. |
| `LUNA-Qwen3-VL.i1-IQ1_S.gguf` | 1.0 GB | ✅ | ❌ | **Conditioning only.** VLM chat produces gibberish at ~1 bit/weight. |

### Vision Projectors (mmproj)

Required only for VLM Chat. Not needed for text conditioning.

| Projector | Size | Notes |
|-----------|------|-------|
| `LUNA-Qwen3-VL.mmproj-Q4_K_M.gguf` | 268 MB | **Recommended.** Tested working with all LLM quants. |
| `LUNA-Qwen3-VL.mmproj-Q8_0.gguf` | 433 MB | Higher precision projector. |
| `LUNA-Qwen3-VL.mmproj-f16.gguf` | 797 MB | Full precision. |

### Adapters (Automatic)

Each LLM quant has a matched adapter that the loader detects and downloads automatically. **You do not need to select adapters manually** — the loader matches them by quant tag.

| Adapter | Size | Matches |
|---------|------|---------|
| `LUNA-Qwen3-VL_Q4_K_M_adapter.safetensors` | ~160 MB | Q4_K_M |
| `LUNA-Qwen3-VL_Q3_K_S_adapter.safetensors` | ~160 MB | Q3_K_S |
| `LUNA-Qwen3-VL_Q2_K_S_adapter.safetensors` | ~160 MB | Q2_K_S |
| `LUNA-Qwen3-VL_IQ2_XXS_adapter.safetensors` | ~160 MB | IQ2_XXS |
| `LUNA-Qwen3-VL_IQ1_S_adapter.safetensors` | ~160 MB | IQ1_S |

### Why Adapters?

The VL/Instruct variants of Qwen3 have drifted from the base model's hidden state distribution. The adapters are small residual MLPs trained per-quant to realign the penultimate layer outputs to match what Z-Image's denoiser expects. Each adapter is trained specifically for its quant level, compensating for quantization-specific distortion — this is why even IQ1_S (~1 bit/weight) produces excellent conditioning.

---

## VRAM Usage

Total VRAM ≈ LLM + adapter (~160 MB) + mmproj (if using VLM chat).

| Configuration | Disk | VRAM (approx) | Use Case |
|---------------|------|---------------|----------|
| IQ1_S + adapter | 1.0 GB + ~160 MB | ~1.3 GB | Conditioning only (ultra-low VRAM) |
| IQ2_XXS + adapter + Q4 mmproj | 1.2 GB + ~160 MB + 268 MB | ~1.8 GB | Full stack — smallest working VLM chat |
| Q2_K_S + adapter + Q4 mmproj | 1.5 GB + ~160 MB + 268 MB | ~2.1 GB | Reliable VLM chat + conditioning |
| Q4_K_M + adapter + Q8 mmproj | 2.4 GB + ~160 MB + 433 MB | ~3.2 GB | **Recommended** — best quality |
| Qwen3-4B bf16 (safetensors) | 7.7 GB | ~8.2 GB | *For comparison — 3.3× larger* |

---

## Workflow Examples

### Text Conditioning Only
```
[LUNA VLM Loader] → [LUNA Text Conditioner] → CONDITIONING → [KSampler]
  └─ model: LUNA-Qwen3-VL.i1-Q4_K_M.gguf
     (adapter auto-loaded)
```

### VLM Image Description → Conditioning
```
[Load Image] ──────────────┐
                            ▼
[LUNA VLM Loader] → [LUNA VLM Chat] → TEXT → [LUNA Text Conditioner] → CONDITIONING
  ├─ model:  LUNA-Qwen3-VL.i1-Q4_K_M.gguf
  └─ mmproj: LUNA-Qwen3-VL.mmproj-Q4_K_M.gguf
     (adapter auto-loaded)
```

### VLM Chat + Manual Prompt Combination
```
[Load Image] ──────────────┐
                            ▼
[LUNA VLM Loader] → [LUNA VLM Chat] → TEXT ─┐
  ├─ model:  LUNA-Qwen3-VL.i1-Q4_K_M.gguf   │
  └─ mmproj: LUNA-Qwen3-VL.mmproj-Q4_K_M.gguf│
                                              ▼
                               [String Concatenate] → [LUNA Text Conditioner] → CONDITIONING
                                       ▲
                     [manual prompt] ───┘
```

---

## VLM Chat Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 0.7 | 0.0 – 2.0 | Sampling temperature. 0 = greedy/deterministic. |
| `max_tokens` | 256 | 32 – 2048 | Maximum tokens to generate. |
| `repetition_penalty` | 1.15 | 1.0 – 2.0 | Penalizes repeated tokens. Prevents looping. 1.0 = off. |
| `system_prompt` | *(image description prompt)* | — | System instruction for the VLM. |
| `prepend_text` / `append_text` | *(empty)* | — | Text added before/after VLM output in the final result. |

---

## Credits

- **llama-cpp-python** fork with penultimate layer API: [LSDJesus/llama-cpp-python](https://github.com/LSDJesus/llama-cpp-python)
- Based on work from [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)

## License

MIT
