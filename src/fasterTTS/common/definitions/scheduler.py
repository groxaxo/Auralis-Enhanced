import time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from typing import Any, Dict, Optional, List, Callable, TypeVar

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
    sequence_buffers: Dict[int, List[Any]] = field(default_factory=lambda: defaultdict(list))
    next_sequence_to_yield: int = 0
    start_time: float = field(default_factory=time.time)
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
