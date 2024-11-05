import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, AsyncGenerator, Callable, TypeVar, Awaitable
import asyncio
import logging
from enum import Enum
import time
from collections import OrderedDict

from fasterTTS.common.logger import setup_logger

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
    id: int
    sequence_order: int  # New field to track original input order
    first_phase_input: Any
    metadata: Dict[str, Any]
    phase: TaskPhase
    context: Any = None
    error: Optional[Exception] = None
    start_time: float = field(default_factory=time.time)
    output_items: List[Any] = field(default_factory=list)
    output_ready_event: asyncio.Event = field(default_factory=asyncio.Event)

class GeneratorTwoPhaseScheduler:
    """
    Scheduler for managing two-phase generator processes with true parallel processing
    while maintaining ordered output.
    """
    def __init__(
            self,
            second_phase_concurrency: int = 10,
            timeout: float = 30.0
    ):
        self.logger = setup_logger(__file__)
        self.second_phase_concurrency = second_phase_concurrency
        self.timeout = timeout

        # Task tracking
        self.pending_tasks: Dict[int, GeneratorPhasedTask] = {}
        self.first_phase_tasks: Dict[int, GeneratorPhasedTask] = {}
        self.second_phase_tasks: Dict[int, GeneratorPhasedTask] = {}
        self.completed_tasks: Dict[int, GeneratorPhasedTask] = {}

        # Output buffer for maintaining order
        self.output_buffer: OrderedDict[int, List[Any]] = OrderedDict()
        self.next_sequence_to_yield = 0
        self.output_available_event = asyncio.Event()

        # Semaphore for second phase concurrency
        self.second_phase_sem = asyncio.Semaphore(self.second_phase_concurrency)

    async def execute_first_phase(
            self,
            task: GeneratorPhasedTask,
            first_phase_fn: Callable[[Any, Dict[str, Any]], Awaitable[Any]],
            second_phase_fn: Callable[[Any, Dict[str, Any]], AsyncGenerator]
    ) -> None:
        """Execute first phase to create context."""
        try:
            self.first_phase_tasks[task.id] = task
            task.context = await first_phase_fn(task.first_phase_input, task.metadata)
            task.phase = TaskPhase.SECOND

            # Initialize buffer for this task's sequence
            self.output_buffer[task.sequence_order] = []

            # Start the second phase immediately
            asyncio.create_task(self.execute_second_phase(task, second_phase_fn, task.context['parallel_inputs']))
        except Exception as e:
            task.error = e
            task.phase = TaskPhase.FAILED
            self.logger.error(f"First phase failed for task {task.id}: {e}")
            task.output_ready_event.set()
        finally:
            self.first_phase_tasks.pop(task.id, None)

    async def process_parallel_generator(
            self,
            generator: AsyncGenerator,
            context: Any,
            sequence_order: int,
            metadata: Dict[str, Any]
    ):
        """Process a single generator in the second phase."""
        try:
            async with self.second_phase_sem:
                async for item in generator:
                    self.output_buffer[sequence_order].append(item)
                    self.output_available_event.set()
        except Exception as e:
            raise e

    async def execute_second_phase(
            self,
            task: GeneratorPhasedTask,
            second_phase_fn: Callable[[Any, Dict[str, Any]], AsyncGenerator],
            parallel_inputs: List[Any]
    ) -> None:
        """Execute second phase with parallel processing of generators."""
        try:
            self.second_phase_tasks[task.id] = task
            self.logger.info(f"Starting second phase for task {task.id}")

            # Create tasks for parallel processing
            generator_tasks = []
            for idx, gen_input in enumerate(parallel_inputs):
                if task.context and 'request_ids' in task.context:
                    self.logger.info(f"Consuming generator for request id: {task.context['request_ids'][idx]}")
                generator = second_phase_fn(gen_input, task.metadata)
                generator_task = asyncio.create_task(
                    self.process_parallel_generator(
                        generator=generator,
                        context=task.context,
                        sequence_order=task.sequence_order,
                        metadata=task.metadata
                    )
                )
                generator_tasks.append(generator_task)

            # Wait for all generators to complete
            await asyncio.gather(*generator_tasks)
            task.phase = TaskPhase.COMPLETED
            execution_time = time.time() - task.start_time
            self.logger.info(
                f"Second phase completed successfully for task {task.id}. "
                f"Execution time: {execution_time:.2f}s"
            )

        except Exception as e:
            task.error = e
            task.phase = TaskPhase.FAILED
            execution_time = time.time() - task.start_time
            self.logger.error(
                f"Second phase failed for task {task.id} after {execution_time:.2f}s. "
                f"Error: {str(e)}"
            )
        finally:
            self.second_phase_tasks.pop(task.id, None)
            self.completed_tasks[task.id] = task
            task.output_ready_event.set()

            # Log final status
            status = "completed successfully" if task.phase == TaskPhase.COMPLETED else f"failed with error: {task.error}"
            total_time = time.time() - task.start_time
            self.logger.info(
                f"Task {task.id} {status}. "
                f"Total execution time: {total_time:.2f}s"
            )
    async def yield_ordered_outputs(self) -> AsyncGenerator[Any, None]:
        """Yield outputs in correct sequence order as they become available."""
        while self.output_buffer:
            if self.next_sequence_to_yield in self.output_buffer:
                buffer = self.output_buffer[self.next_sequence_to_yield]

                while buffer:  # Process all available items for current sequence
                    yield buffer.pop(0)

                if not buffer:  # If buffer is empty, remove it and advance sequence
                    self.output_buffer.pop(self.next_sequence_to_yield)
                    self.next_sequence_to_yield += 1
            else:
                # Wait for more output to become available
                self.output_available_event.clear()
                await self.output_available_event.wait()

    async def run(
            self,
            inputs: List[Any],
            metadata_list: List[Dict[str, Any]],
            first_phase_fn: Callable[[Any, Dict[str, Any]], Awaitable[Any]],
            second_phase_fn: Callable[[Any, Dict[str, Any]], AsyncGenerator],
    ) -> AsyncGenerator[Any, None]:
        """
        Run the pipeline with true parallel processing while maintaining output order.
        """
        # Initialize all tasks with sequence order
        for i, (input_data, metadata) in enumerate(zip(inputs, metadata_list)):
            task = GeneratorPhasedTask(
                id=i,
                sequence_order=i,
                first_phase_input=input_data,
                metadata=metadata,
                phase=TaskPhase.FIRST
            )
            self.pending_tasks[i] = task
            asyncio.create_task(
                self.execute_first_phase(task, first_phase_fn, second_phase_fn)
            )

        # Start output consumer
        async for item in self.yield_ordered_outputs():
            yield item

        # Wait for all tasks to complete and check for errors
        await asyncio.gather(*[t.output_ready_event.wait() for t in self.pending_tasks.values()])

        # Check for any errors after all tasks complete
        failed_tasks = [t for t in self.pending_tasks.values() if t.phase == TaskPhase.FAILED]
        if failed_tasks:
            raise failed_tasks[0].error  # Raise the first error encountered