import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar, AsyncGenerator, Callable, Any
from auralis.common.logging.logger import setup_logger


T = TypeVar('T')


@dataclass
class TTSMetricsTracker:
    logger = setup_logger(__file__)

    window_start: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)
    log_interval: float = 5.0  # sec between logs

    window_tokens: int = 0
    window_audio_seconds: float = 0
    window_requests: int = 0

    @property
    def requests_per_second(self) -> float:
        elapsed = time.time() - self.window_start
        return self.window_requests / elapsed if elapsed > 0 else 0

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.time() - self.window_start
        return self.window_tokens / elapsed if elapsed > 0 else 0

    @property
    def ms_per_second_of_audio(self) -> float:
        elapsed = (time.time() - self.window_start) * 1000  # in ms
        return elapsed / self.window_audio_seconds if self.window_audio_seconds > 0 else 0

    def reset_window(self) -> None:
        current_time = time.time()
        self.last_log_time = current_time
        # reset window
        self.window_start = current_time
        self.window_tokens = 0
        self.window_audio_seconds = 0
        self.window_requests = 0

    def update_metrics(self, tokens: int, audio_seconds: float) -> bool:
        self.window_tokens += tokens
        self.window_audio_seconds += audio_seconds
        self.window_requests += 1

        current_time = time.time()
        should_log = current_time - self.last_log_time >= self.log_interval

        return should_log


metrics = TTSMetricsTracker()


def track_generation(func: Callable[..., AsyncGenerator[T, None]]) -> Callable[..., AsyncGenerator[T, None]]:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator[T, None]:
        async for output in func(*args, **kwargs):
            if output.start_time:
                audio_seconds = output.array.shape[0] / output.sample_rate

                if metrics.update_metrics(output.token_length, audio_seconds):
                    metrics.logger.info(
                        f"Generation metrics | "
                        f"Throughput: {metrics.requests_per_second:.2f} req/s | "
                        f"{metrics.tokens_per_second:.1f} tokens/s | "
                        f"Latency: {metrics.ms_per_second_of_audio:.0f}ms per second of audio generated"
                    )
                    metrics.reset_window()
            yield output

    return wrapper