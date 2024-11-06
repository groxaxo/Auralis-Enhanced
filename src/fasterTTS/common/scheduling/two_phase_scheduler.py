import uuid

from typing import Any, Dict, AsyncGenerator, Callable, Awaitable
import asyncio
import logging

import time

from FasterTTS import setup_logger
from src.fasterTTS.common.definitions.requests import TTSRequest
from src.fasterTTS.common.definitions.scheduler import QueuedRequest, TaskState
from src.fasterTTS.models.base import AudioOutputGenerator



class TwoPhaseScheduler:
    def __init__(
            self,
            second_phase_concurrency: int = 10,
            request_timeout: float = 300.0,
            generator_timeout: float = 30.0
    ):
        self.logger = setup_logger(__file__, logging.DEBUG)
        self.second_phase_concurrency = second_phase_concurrency
        self.request_timeout = request_timeout
        self.generator_timeout = generator_timeout

        # Queues and state tracking
        self.request_queue: asyncio.Queue[QueuedRequest] = None
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.second_phase_sem = None

        self.is_running = False
        self.queue_processor_task = None

    async def start(self):
        """Initialize async components and start the queue processor."""
        if self.is_running:
            return

        self.logger.debug("Starting the scheduler...")

        self.request_queue = asyncio.Queue()
        self.second_phase_sem = asyncio.Semaphore(self.second_phase_concurrency)
        self.is_running = True
        self.queue_processor_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Main queue processor that handles incoming requests."""
        while self.is_running:
            try:
                request = await self.request_queue.get()
                if request.state != TaskState.QUEUED:
                    continue

                # Start processing the request
                self.active_requests[request.id] = request
                asyncio.create_task(self._process_request(request))

            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on persistent errors

    async def _process_request(
            self,
            request: QueuedRequest,
    ):
        """Process a single request through both phases."""
        try:
            # First phase processing
            request.state = TaskState.PROCESSING_FIRST
            try:
                request.first_phase_result = await asyncio.wait_for(
                    request.first_fn(request.input),
                    timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"First phase timed out after {self.request_timeout}s")

            # Extract parallel inputs
            parallel_inputs = request.first_phase_result.get('parallel_inputs', [])
            request.generators_count = len(parallel_inputs)

            # Start second phase processing
            request.state = TaskState.PROCESSING_SECOND
            generator_tasks = []
            for sequence_idx, gen_input in enumerate(parallel_inputs):
                task = asyncio.create_task(
                    self._process_generator(
                        request=request,
                        generator_input=gen_input,
                        sequence_idx=sequence_idx,
                    )
                )
                generator_tasks.append(task)

            # Wait for all generators to complete
            await asyncio.gather(*generator_tasks)

            if request.error is None:
                request.state = TaskState.COMPLETED

        except TimeoutError as e:
            request.error = e
            request.state = TaskState.FAILED
            self.logger.error(f"Request {request.id} timed out after {time.time() - request.start_time:.2f}s")
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
        """Process a single generator in the second phase."""
        try:
            async with self.second_phase_sem:
                self.logger.debug(f"Starting to process sub-request {request.id}_{sequence_idx}")

                generator = request.second_fn(generator_input)

                # Creiamo un Event per segnalare quando questo generatore ha finito
                if 'generator_events' not in request.__dict__:
                    request.generator_events = {}
                request.generator_events[sequence_idx] = asyncio.Event()

                while True:
                    try:
                        item = await asyncio.wait_for(
                            generator.__anext__(),
                            timeout=self.generator_timeout
                        )
                        # Aggiungiamo l'item al buffer con un event per segnalare la sua disponibilità
                        request.sequence_buffers[sequence_idx].append((item, asyncio.Event()))
                        request.sequence_buffers[sequence_idx][-1][1].set()  # Segnaliamo che l'item è pronto

                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Generator {sequence_idx} timed out after {self.generator_timeout}s")

        except Exception as e:
            self.logger.error(f"Generator {sequence_idx} for request {request.id} failed: {e}")
            if request.error is None:  # Store first error encountered
                request.error = e
        finally:
            self.logger.debug(f"Finished sub-request {request.id}_{sequence_idx}")
            request.completed_generators += 1
            if hasattr(request, 'generator_events') and sequence_idx in request.generator_events:
                request.generator_events[sequence_idx].set()

    async def _yield_ordered_outputs(self, request: QueuedRequest) -> AsyncGenerator[Any, None]:
        """Yield outputs in correct sequence order as soon as they're available."""
        current_index = 0

        while True:
            # Verifichiamo se abbiamo finito SOLO se la richiesta è completata o fallita
            if (request.state in (TaskState.COMPLETED, TaskState.FAILED) and
                    request.completed_generators >= request.generators_count and
                    all(len(buffer) == 0 for buffer in request.sequence_buffers.values())):
                break

            # Se c'è un errore, lo propaghiamo
            if request.error:
                raise request.error

            # Controlliamo se ci sono items disponibili nel buffer corrente
            if current_index in request.sequence_buffers:
                while request.sequence_buffers[current_index]:
                    item, event = request.sequence_buffers[current_index][0]
                    # Aspettiamo che l'item sia effettivamente pronto
                    await event.wait()
                    yield item
                    request.sequence_buffers[current_index].pop(0)

                # Se abbiamo finito con questo buffer e il generatore ha finito,
                # passiamo al prossimo indice
                if (hasattr(request, 'generator_events') and
                        current_index in request.generator_events and
                        request.generator_events[current_index].is_set()):
                    current_index += 1
            else:
                # Se non abbiamo ancora il buffer per questo indice, aspettiamo un po'
                await asyncio.sleep(0.01)

    async def run(
            self,
            inputs: TTSRequest,
            first_phase_fn: Callable[[Any], Awaitable[Any]],
            second_phase_fn: Callable[[Dict], AudioOutputGenerator]
    ) -> AsyncGenerator[Any, None]:
        """Add a request to the queue and yield results in order."""
        if not self.is_running:
            await self.start()

        # Split input if it's a batch
        assert not isinstance(inputs.text, list), "Batch processing not supported in async mode"

        request = QueuedRequest(
                id=uuid.uuid4().hex,
                input=inputs,
                first_fn=first_phase_fn,
                second_fn=second_phase_fn
        )
        self.logger.info(f"Starting request {request.id}")
        # Add to queue immediately
        await self.request_queue.put(request)

        try:
            async for item in self._yield_ordered_outputs(request):
                yield item

            # Wait for completion with timeout
            try:
                await asyncio.wait_for(
                    request.completion_event.wait(),
                    timeout=self.request_timeout
                )
                if request.error:
                    raise request.error
            except asyncio.TimeoutError:
                raise TimeoutError(f"Request timed out waiting for completion after {self.request_timeout}s")

        finally:
            self.active_requests.pop(request.id, None)
            self.logger.info(f"Finished request {request.id}")

    async def shutdown(self):
        """Gracefully shutdown the scheduler."""
        self.is_running = False
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass

        # Wait for active requests to complete
        if self.active_requests:
            self.logger.info(f"Waiting for {len(self.active_requests)} active requests to complete...")
            await asyncio.gather(
                *(request.completion_event.wait() for request in self.active_requests.values()),
                return_exceptions=True
            )