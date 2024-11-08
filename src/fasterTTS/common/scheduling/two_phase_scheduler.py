import uuid
from typing import Any, Dict, AsyncGenerator, Callable, Awaitable
import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fasterTTS.common.definitions.scheduler import QueuedRequest, TaskState
from fasterTTS.common.logging.logger import setup_logger


class TwoPhaseScheduler:
    def __init__(
            self,
            second_phase_concurrency: int = 10,
            request_timeout: float = None,
            generator_timeout: float = None
    ):
        self.logger = setup_logger(__file__)
        self.second_phase_concurrency = second_phase_concurrency
        self.request_timeout = request_timeout
        self.generator_timeout = generator_timeout

        self.request_queue: asyncio.Queue[QueuedRequest] = None
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.second_phase_sem = None

        # Track active generator count
        self.active_generator_count = 0
        self.generator_count_lock = asyncio.Lock()

        self.is_running = False
        self.queue_processor_task = None
        self.cleanup_lock = asyncio.Lock()

    async def start(self):
        if self.is_running:
            return

        self.logger.debug("Starting the scheduler...")
        self.request_queue = asyncio.Queue()
        self.second_phase_sem = asyncio.Semaphore(self.second_phase_concurrency)
        self.is_running = True

        # Start multiple request processors to maintain concurrency
        self.queue_processor_tasks = [
            asyncio.create_task(self._process_queue())
            for _ in range(self.second_phase_concurrency)
        ]

    async def _process_queue(self):
        """Process requests from the queue, maintaining high concurrency."""
        while self.is_running:
            try:
                request = await self.request_queue.get()
                if request.state != TaskState.QUEUED:
                    continue

                async with self._request_lifecycle(request.id):
                    self.active_requests[request.id] = request
                    await self._process_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)

    @asynccontextmanager
    async def _request_lifecycle(self, request_id: str):
        """Context manager to handle request lifecycle and cleanup."""
        try:
            yield
        finally:
            async with self.cleanup_lock:
                self.active_requests.pop(request_id, None)
                self.logger.info(f"Cleaned up request {request_id}")


    async def _process_request(self, request: QueuedRequest):
        """Process a single request through both phases."""
        try:
            # First phase processing with timeout
            request.state = TaskState.PROCESSING_FIRST
            try:
                request.first_phase_result = await asyncio.wait_for(
                    request.first_fn(request.input),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"First phase timed out after {self.request_timeout}s")

            parallel_inputs = request.first_phase_result.get('parallel_inputs', [])
            request.generators_count = len(parallel_inputs)

            # Start second phase processing
            request.state = TaskState.PROCESSING_SECOND

            # Process generators concurrently but with controlled concurrency
            generator_tasks = []
            for sequence_idx, gen_input in enumerate(parallel_inputs):
                task = asyncio.create_task(
                    self._process_generator(request, gen_input, sequence_idx)
                )
                generator_tasks.append(task)

            # Wait for all generators with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*generator_tasks, return_exceptions=True),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                for task in generator_tasks:
                    if not task.done():
                        task.cancel()
                raise TimeoutError(f"Second phase timed out after {self.request_timeout}s")

            if request.error is None:
                request.state = TaskState.COMPLETED

        except Exception as e:
            request.error = e
            request.state = TaskState.FAILED
            self.logger.error(f"Request {request.id} failed: {e}")
        finally:
            request.completion_event.set()

    async def _process_generator(
            self,
            request: QueuedRequest,
            generator_input: Any,
            sequence_idx: int,
    ):
        """Process a single generator with semaphore control."""
        async with self.second_phase_sem:
            try:
                async with self.generator_count_lock:
                    self.active_generator_count += 1
                    current_count = self.active_generator_count

                self.logger.debug(f"Active generators: {current_count}/{self.second_phase_concurrency}")

                # Initialize generator events if not exists
                if not hasattr(request, 'generator_events'):
                    request.generator_events = {}
                request.generator_events[sequence_idx] = asyncio.Event()

                generator = request.second_fn(generator_input)
                buffer = request.sequence_buffers[sequence_idx]

                while True:
                    try:
                        item = await asyncio.wait_for(
                            generator.__anext__(),
                            timeout=self.generator_timeout
                        )
                        event = asyncio.Event()
                        event.set()
                        buffer.append((item, event))
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Generator {sequence_idx} timed out")

            except asyncio.CancelledError:
                self.logger.warning(f"Generator {sequence_idx} for request {request.id} was cancelled")
                raise
            except Exception as e:
                self.logger.error(f"Generator {sequence_idx} for request {request.id} failed: {e}")
                if request.error is None:
                    request.error = e
            finally:
                async with self.generator_count_lock:
                    self.active_generator_count -= 1
                    current_count = self.active_generator_count

                self.logger.debug(
                    f"Active generators after completion: {current_count}/{self.second_phase_concurrency}")
                request.completed_generators += 1
                if sequence_idx in request.generator_events:
                    request.generator_events[sequence_idx].set()

    async def _yield_ordered_outputs(self, request: QueuedRequest) -> AsyncGenerator[Any, None]:
        """Yield outputs in correct sequence order with timeout protection."""
        current_index = 0
        last_progress_time = time.time()

        while True:
            # Check for completion or timeout
            if (request.state in (TaskState.COMPLETED, TaskState.FAILED) and
                    request.completed_generators >= request.generators_count and
                    all(len(buffer) == 0 for buffer in request.sequence_buffers.values())):
                break

            # Check for deadlock - no progress for too long
            if  self.request_timeout and time.time() - last_progress_time > self.request_timeout:
                raise TimeoutError("No progress in output generation")

            if request.error:
                raise request.error

            if current_index in request.sequence_buffers:
                buffer = request.sequence_buffers[current_index]

                while buffer:
                    item, event = buffer[0]
                    # Wait for item with timeout
                    try:
                        await asyncio.wait_for(event.wait(), timeout=self.generator_timeout)
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Timeout waiting for item in sequence {current_index}")

                    yield item
                    buffer.pop(0)
                    last_progress_time = time.time()

                # Move to next sequence if current is complete
                if (hasattr(request, 'generator_events') and
                        current_index in request.generator_events and
                        request.generator_events[current_index].is_set()):
                    current_index += 1

            await asyncio.sleep(0.01)  # Prevent tight loop

    async def run(
            self,
            inputs: Any,
            first_phase_fn: Callable[[Any], Awaitable[Any]],
            second_phase_fn: Callable[[Dict], AsyncGenerator]
    ) -> AsyncGenerator[Any, None]:
        if not self.is_running:
            await self.start()

        request = QueuedRequest(
            id=uuid.uuid4().hex,
            input=inputs,
            first_fn=first_phase_fn,
            second_fn=second_phase_fn
        )

        self.logger.info(f"Starting request {request.id}")
        await self.request_queue.put(request)

        try:
            async for item in self._yield_ordered_outputs(request):
                yield item

            await asyncio.wait_for(
                request.completion_event.wait(),
                timeout=self.request_timeout
            )
            if request.error:
                raise request.error

        finally:
            async with self.cleanup_lock:
                self.active_requests.pop(request.id, None)

    async def shutdown(self):
        self.is_running = False

        # Cancel all processor tasks
        for task in self.queue_processor_tasks:
            if task and not task.done():
                task.cancel()

        await asyncio.gather(*self.queue_processor_tasks, return_exceptions=True)

        if self.active_requests:
            await asyncio.gather(
                *(request.completion_event.wait() for request in self.active_requests.values()),
                return_exceptions=True
            )