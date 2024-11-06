import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, AsyncGenerator, Callable, TypeVar, Awaitable
import asyncio
from enum import Enum
import time
from collections import OrderedDict, defaultdict

from fasterTTS.common.logger import setup_logger
from fasterTTS.common.output import TTSOutput
from fasterTTS.common.requests import TTSRequest
from fasterTTS.models.base_tts_engine import AudioTokenGenerator, AudioOutputGenerator

T = TypeVar('T')
R = TypeVar('R')

class TaskPhase(Enum):
    """Generic phases for a two-phase task execution."""
    FIRST = "first_phase"
    SECOND = "second_phase"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GeneratorPhasedTask:
    """Task for two-phase generator execution."""
    request_id: str  # ID della richiesta originale
    sub_task_id: int  # Indice all'interno della richiesta per gestire i generatori multipli
    global_order: int  # Ordine globale per preservare l'ordinamento originale
    first_phase_input: Any
    phase: TaskPhase
    context: Any = None
    error: Optional[Exception] = None
    start_time: float = field(default_factory=time.time)
    output_items: List[Any] = field(default_factory=list)
    output_ready_event: asyncio.Event = field(default_factory=asyncio.Event)

class GeneratorTwoPhaseScheduler:
    """
    Scheduler for managing two-phase generator processes with parallel processing
    while maintaining ordered output.
    """
    def __init__(
            self,
            second_phase_concurrency: int = 10,
            timeout: float = 5.0
    ):
        self.logger = setup_logger(__file__)
        self.second_phase_concurrency = second_phase_concurrency
        self.timeout = timeout

        # Task tracking per request_id
        self.tasks: Dict[str, Dict[int, GeneratorPhasedTask]] = defaultdict(dict)
        self.output_buffer: Dict[str, OrderedDict[int, List[Any]]] = defaultdict(OrderedDict)
        self.next_to_yield: Dict[str, int] = defaultdict(int)

        # Semaphore for second phase concurrency
        self.second_phase_sem = asyncio.Semaphore(second_phase_concurrency)
        self.buffer_lock = asyncio.Lock()

    async def execute_first_phase(
            self,
            task: GeneratorPhasedTask,
            first_phase_fn: Callable[[TTSRequest], Awaitable[Any]],
            second_phase_fn: Callable[[AudioTokenGenerator], AudioOutputGenerator]
    ) -> None:
        """Execute first phase to create context."""
        try:
            task.context = await first_phase_fn(task.first_phase_input)
            task.phase = TaskPhase.SECOND

            # Process parallel generators maintaining order
            for idx, gen_input in enumerate(task.context['parallel_inputs']):
                sub_task = GeneratorPhasedTask(
                    request_id=task.request_id,
                    sub_task_id=idx,
                    global_order=task.global_order * 1000 + idx,  # Preserve ordering
                    first_phase_input=gen_input,
                    phase=TaskPhase.SECOND
                )
                self.tasks[task.request_id][sub_task.global_order] = sub_task
                asyncio.create_task(self.execute_second_phase(sub_task, second_phase_fn))

        except Exception as e:
            task.error = e
            task.phase = TaskPhase.FAILED
            self.logger.error(f"First phase failed for task {task.id}: {e}")
            task.output_ready_event.set()

    async def execute_second_phase(
            self,
            task: GeneratorPhasedTask,
            second_phase_fn: Callable[[Any], AsyncGenerator]
    ) -> None:
        try:
            async with self.second_phase_sem:
                generator = second_phase_fn(task.first_phase_input)
                async for item in generator:
                    async with self.buffer_lock:
                        if task.global_order not in self.output_buffer[task.request_id]:
                            self.output_buffer[task.request_id][task.global_order] = []
                        self.output_buffer[task.request_id][task.global_order].append(item)

                task.phase = TaskPhase.COMPLETED

        except Exception as e:
            task.error = e
            task.phase = TaskPhase.FAILED
        finally:
            task.output_ready_event.set()

    async def yield_ordered_outputs(self, request_id: str) -> AsyncGenerator[Any, None]:
        current_order = 0
        while True:
            if request_id in self.output_buffer and current_order in self.output_buffer[request_id]:
                async with self.buffer_lock:
                    buffer = self.output_buffer[request_id][current_order]
                    while buffer:
                        yield buffer.pop(0)
                    del self.output_buffer[request_id][current_order]
                current_order += 1
                continue

            # Check if all tasks are completed
            if not any(task.phase != TaskPhase.COMPLETED
                      for tasks in self.tasks[request_id].values()
                      for task in [tasks] if isinstance(tasks, GeneratorPhasedTask)):
                break

            await asyncio.sleep(0.1)  # Prevent busy waiting

    async def run(
            self,
            request_id: str,
            inputs: TTSRequest,
            first_phase_fn: Callable[[Any], Awaitable[Any]],
            second_phase_fn: Callable[[Any], AsyncGenerator]
    ) -> AsyncGenerator[Any, None]:
        """
        Run the pipeline maintaining both request and generator ordering.

        Args:
            request_id: ID unico della richiesta
            inputs: TTSRequest to process
            first_phase_fn: Funzione per la prima fase
            second_phase_fn: Funzione per la seconda fase
        """
        try:
            if not isinstance(inputs.text, list):
                inputs.text = [inputs.text]
            # Initialize main tasks
            for i, input_data in enumerate(inputs.text):
                task = GeneratorPhasedTask(
                    request_id=request_id,
                    sub_task_id=0,  # Will be updated for parallel generators
                    global_order=i,
                    first_phase_input=input_data,
                    phase=TaskPhase.FIRST
                )
                self.tasks[request_id][i] = task
                asyncio.create_task(
                    self.execute_first_phase(task, first_phase_fn, second_phase_fn)
                )

            # Yield results in order
            async for item in self.yield_ordered_outputs(request_id):
                yield item

        finally:
            # Cleanup
            self.tasks.pop(request_id, None)
            self.output_buffer.pop(request_id, None)
            self.next_to_yield.pop(request_id, None)