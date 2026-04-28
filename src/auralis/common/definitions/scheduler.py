import time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from typing import Any, Dict, Optional, List, Callable, TypeVar, Deque

import asyncio

T = TypeVar('T')
R = TypeVar('R')


class TaskState(Enum):
    QUEUED = "queued"
    PROCESSING_FIRST = "processing_first"
    PROCESSING_SECOND = "processing_second"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedRequest:
    id: str
    input: Any
    state: TaskState = TaskState.QUEUED
    error: Optional[Exception] = None
    first_phase_result: Any = None
    generators_count: int = 0
    completed_generators: int = 0
    first_fn: Callable = None
    second_fn: Callable = None
    sequence_buffers: Dict[int, Deque[Any]] = field(default_factory=lambda: defaultdict(deque))
    # Per-sequence events that are set whenever a new item is written to the
    # corresponding buffer.  This allows _yield_ordered_outputs to await
    # instead of busy-polling with asyncio.sleep.
    buffer_ready_events: Dict[int, asyncio.Event] = field(default_factory=dict)
    next_sequence_to_yield: int = 0
    start_time: float = field(default_factory=time.time)
    first_phase_duration: float = 0.0
    second_phase_duration: float = 0.0
    total_duration: float = 0.0
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
