import asyncio
import importlib.util
import logging
import sys
import types


def _load_scheduler_modules():
    root = "/home/runner/work/Auralis-Enhanced/Auralis-Enhanced/src"
    sys.modules.setdefault("auralis", types.ModuleType("auralis"))
    sys.modules.setdefault("auralis.common", types.ModuleType("auralis.common"))
    sys.modules.setdefault("auralis.common.logging", types.ModuleType("auralis.common.logging"))
    sys.modules.setdefault("auralis.common.definitions", types.ModuleType("auralis.common.definitions"))
    sys.modules.setdefault("auralis.common.scheduling", types.ModuleType("auralis.common.scheduling"))

    logger_module = types.ModuleType("auralis.common.logging.logger")
    logger_module.setup_logger = logging.getLogger
    sys.modules["auralis.common.logging.logger"] = logger_module

    definitions_spec = importlib.util.spec_from_file_location(
        "auralis.common.definitions.scheduler",
        f"{root}/auralis/common/definitions/scheduler.py",
    )
    definitions_module = importlib.util.module_from_spec(definitions_spec)
    sys.modules["auralis.common.definitions.scheduler"] = definitions_module
    definitions_spec.loader.exec_module(definitions_module)

    scheduler_spec = importlib.util.spec_from_file_location(
        "auralis.common.scheduling.two_phase_scheduler",
        f"{root}/auralis/common/scheduling/two_phase_scheduler.py",
    )
    scheduler_module = importlib.util.module_from_spec(scheduler_spec)
    sys.modules["auralis.common.scheduling.two_phase_scheduler"] = scheduler_module
    scheduler_spec.loader.exec_module(scheduler_module)
    return definitions_module, scheduler_module


def test_run_generator_stores_raw_items_without_events():
    async def _run():
        definitions_module, scheduler_module = _load_scheduler_modules()
        scheduler = scheduler_module.TwoPhaseScheduler()
        request = definitions_module.QueuedRequest(id="req-1", input=None, second_fn=None)
        request.sequence_buffers = {0: []}

        async def _generator():
            yield "chunk-1"
            yield "chunk-2"

        request.second_fn = lambda _: _generator()
        await scheduler._run_generator(request, None, 0)
        assert request.sequence_buffers[0] == ["chunk-1", "chunk-2"]

    asyncio.run(_run())


def test_yield_ordered_outputs_reads_raw_items():
    async def _run():
        definitions_module, scheduler_module = _load_scheduler_modules()
        scheduler = scheduler_module.TwoPhaseScheduler()
        request = definitions_module.QueuedRequest(id="req-2", input=None, second_fn=None)
        request.sequence_buffers = {0: ["a"], 1: ["b"]}
        request.generators_count = 2
        request.completed_generators = 2
        request.state = definitions_module.TaskState.COMPLETED

        yielded = []
        async for item in scheduler._yield_ordered_outputs(request):
            yielded.append(item)

        assert yielded == ["a", "b"]

    asyncio.run(_run())


def test_process_request_tracks_phase_durations():
    async def _run():
        definitions_module, scheduler_module = _load_scheduler_modules()
        scheduler = scheduler_module.TwoPhaseScheduler()

        async def _first_fn(_):
            return {"parallel_inputs": [None]}

        async def _second_gen(_):
            yield "chunk"

        request = definitions_module.QueuedRequest(
            id="req-3",
            input=None,
            first_fn=_first_fn,
            second_fn=_second_gen,
        )

        await scheduler._process_request(request)

        assert request.first_phase_duration >= 0
        assert request.second_phase_duration >= 0
        assert request.total_duration >= 0
        assert request.total_duration >= request.first_phase_duration
        assert request.total_duration >= request.second_phase_duration

    asyncio.run(_run())
