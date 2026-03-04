"""
LUNA-Z-Image-Qwen3-VL — ComfyUI Custom Nodes
==============================================
GGUF-based Qwen3/Qwen3-VL text encoding and VLM inference for Z-Image.

Nodes:
  - LunaVLMLoader: Load a GGUF model + optional mmproj via llama-cpp-python
  - LunaTextConditioner: Extract penultimate hidden states for Z-Image conditioning
  - LunaVLMChat: Vision-language inference — describe images, generate prompts

Requires: llama-cpp-python (Luna fork with penultimate layer API)
"""

from .nodes_loader import LunaVLMLoader
from .nodes_conditioner import LunaTextConditioner
from .nodes_vlm import LunaVLMChat

NODE_CLASS_MAPPINGS = {
    "LunaVLMLoader": LunaVLMLoader,
    "LunaTextConditioner": LunaTextConditioner,
    "LunaVLMChat": LunaVLMChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LunaVLMLoader": "LUNA VLM Loader (GGUF)",
    "LunaTextConditioner": "LUNA Text Conditioner",
    "LunaVLMChat": "LUNA VLM Chat",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
