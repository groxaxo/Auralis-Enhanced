from .components.vllm_mm_gpt import XttsGPT
from vllm import ModelRegistry

ModelRegistry.register_model("XttsGPT", XttsGPT)
