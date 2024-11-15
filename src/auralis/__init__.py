from .core.tts import TTS
from .common.definitions.requests import TTSRequest
from .common.definitions.output import TTSOutput
from .common.logging.logger import setup_logger

setup_logger(__file__) # so it overwrite the vllm logger