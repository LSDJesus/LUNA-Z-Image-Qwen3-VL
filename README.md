# LUNA Z-Image Qwen3-VL

ComfyUI custom nodes for using **GGUF quantized Qwen3-VL** as a text encoder for [Z-Image](https://github.com/city96/ComfyUI-GGUF), with built-in vision-language (VLM) chat for image-to-prompt generation.

**Models are auto-downloaded from [HuggingFace](https://huggingface.co/LSDJesus/LUNA-Qwen3-VL) on first use.**

> **[How We Got Here](Hacking_llama_for_qwen3-vl.md)** — Technical deep-dive into hacking llama.cpp for penultimate hidden state extraction, building the VL-to-Base adapter, and achieving 3.3× VRAM reduction with vision capabilities.

## Nodes

| Node | Description |
|------|-------------|
| **LUNA VLM Loader** | Load a GGUF model (+ optional mmproj for vision) via llama-cpp-python |
| **LUNA Text Conditioner** | Extract penultimate hidden states for Z-Image conditioning, with optional adapter alignment |
| **LUNA VLM Chat** | Feed an image to the VLM, get descriptive text back for prompt engineering |

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

## Dependencies

This node requires a custom fork of **llama-cpp-python** with penultimate hidden state extraction support. The correct wheel is installed automatically via `requirements.txt`:

- **Windows (CUDA):** [llama_cpp_python-0.3.27 (win_amd64)](https://github.com/LSDJesus/llama-cpp-python/releases/download/v0.3.27-cuda-6396cf3/llama_cpp_python-0.3.27-cp39-abi3-win_amd64.whl)
- **Linux (CUDA):** [llama_cpp_python-0.3.27 (manylinux)](https://github.com/LSDJesus/llama-cpp-python/releases/download/v0.3.27-cuda-6396cf3/llama_cpp_python-0.3.27-cp39-abi3-manylinux_2_28_x86_64.whl)

## Models

All model files are stored in `ComfyUI/models/LLM/LUNA-Qwen3-VL/` and **downloaded automatically** from [HuggingFace](https://huggingface.co/LSDJesus/LUNA-Qwen3-VL) when you first run a node.

| File | Size | Purpose |
|------|------|---------|
| `LUNA-Qwen3-VL.i1-Q4_K_M.gguf` | ~2.4 GB | GGUF text encoder model |
| `LUNA-Qwen3-VL.mmproj-Q8_0.gguf` | ~600 MB | Vision projector (needed for VLM Chat) |
| `LUNA-Qwen3-VL_adapter.safetensors` | ~160 MB | VL-to-Base adapter for conditioning alignment |

To download manually instead, place the files in `ComfyUI/models/LLM/LUNA-Qwen3-VL/`.

### Why an Adapter?

The VL/Instruct variants of Qwen3 have drifted from the base model's hidden state distribution. The adapter realigns the penultimate layer outputs to match what Z-Image's denoiser expects, so you get proper conditioning from a vision-capable model.

### VRAM Usage

| Configuration | Size on Disk | VRAM |
|---------------|-------------|------|
| Qwen3-4B safetensors (bf16) | 7.67 GB | ~8.2 GB |
| LUNA-Qwen3-VL GGUF (Q4_K_M) | 2.4 GB | ~2.5 GB |
| LUNA-Qwen3-VL GGUF + adapter | 2.4 GB + 160 MB | ~2.7 GB |

## Workflow Examples

### Text Encoding with Adapter
```
[LUNA VLM Loader] → [LUNA Text Conditioner] → CONDITIONING → [KSampler]
  ├─ model:   LUNA-Qwen3-VL.i1-Q4_K_M.gguf
  └─ adapter: LUNA-Qwen3-VL_adapter.safetensors
```

### VLM Image Description → Conditioning
```
[Load Image] ──────────────┐
                            ▼
[LUNA VLM Loader] → [LUNA VLM Chat] → TEXT → [LUNA Text Conditioner] → CONDITIONING
  ├─ model:   LUNA-Qwen3-VL.i1-Q4_K_M.gguf
  ├─ mmproj:  LUNA-Qwen3-VL.mmproj-Q8_0.gguf
  └─ adapter: LUNA-Qwen3-VL_adapter.safetensors
```

### VLM Chat + Manual Prompt Combination
```
[Load Image] ──────────────┐
                            ▼
[LUNA VLM Loader] → [LUNA VLM Chat] → TEXT ─┐
  ├─ model:   LUNA-Qwen3-VL.i1-Q4_K_M.gguf  │
  └─ mmproj:  LUNA-Qwen3-VL.mmproj-Q8_0.gguf ▼
                               [String Concatenate] → [LUNA Text Conditioner] → CONDITIONING
                                       ▲
                     [manual prompt] ───┘
```

## Credits

- **llama-cpp-python** fork with penultimate layer API: [LSDJesus/llama-cpp-python](https://github.com/LSDJesus/llama-cpp-python)
- Based on work from [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)

## License

MIT
