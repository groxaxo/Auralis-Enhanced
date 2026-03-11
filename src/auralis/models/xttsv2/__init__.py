from .XTTSv2 import XTTSv2Engine
from ..registry import register_model

register_model("xtts", XTTSv2Engine)

try:
    from vllm import ModelRegistry
    from .components.vllm_mm_gpt import XttsGPT
except Exception:
    pass
else:
    ModelRegistry.register_model("XttsGPT", XttsGPT)
