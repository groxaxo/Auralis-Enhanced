from .XTTSv2 import XTTSv2Engine
from .components.vllm_mm_gpt import XttsGPT
from ..registry import register_model

from vllm import ModelRegistry

register_model("xtts", XTTSv2Engine)
ModelRegistry.register_model("XttsGPT", XttsGPT)
