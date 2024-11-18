from .xttsv2.XTTSv2 import XTTSv2Engine
MODEL_REGISTRY = {
    "xtts": XTTSv2Engine,
}

def register_model(name, model):
    MODEL_REGISTRY[name] = model