import sys
import torch
import traceback
import functools
import threading
from typing import Optional
from contextlib import contextmanager


class CUDAOperationTracer:
    def __init__(self, max_trace_history: int = 100):
        self.operation_history = []
        self.max_history = max_trace_history
        self.lock = threading.Lock()
        self.enabled = True

    def trace_operation(self, op_name: str, shapes: tuple):
        if not self.enabled:
            return

        with self.lock:
            stack = traceback.extract_stack()
            # Rimuovi le funzioni interne del tracer
            relevant_stack = [frame for frame in stack if 'cuda_trace_debug' not in frame.filename]

            operation_info = {
                'op': op_name,
                'shapes': shapes,
                'stack': relevant_stack,
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }

            self.operation_history.append(operation_info)
            if len(self.operation_history) > self.max_history:
                self.operation_history.pop(0)

    def print_last_operations(self, n: int = 10):
        print("\n=== Last CUDA Operations ===")
        for op in self.operation_history[-n:]:
            print(f"\nOperation: {op['op']}")
            print(f"Shapes: {op['shapes']}")
            print(f"Memory Allocated: {op['memory_allocated'] / 1e9:.2f} GB")
            print("Stack:")
            for frame in op['stack'][-3:]:  # Mostra solo gli ultimi 3 frame dello stack
                print(f"  File {frame.filename}, line {frame.lineno}, in {frame.name}")
                if frame.line:
                    print(f"    {frame.line}")


def trace_cuda_ops(tensor_op):
    """Decoratore per tracciare le operazioni tensor."""

    @functools.wraps(tensor_op)
    def wrapper(*args, **kwargs):
        if not hasattr(torch, '_cuda_tracer'):
            torch._cuda_tracer = CUDAOperationTracer()

        shapes = tuple(arg.shape if isinstance(arg, torch.Tensor) else None
                       for arg in args)

        try:
            result = tensor_op(*args, **kwargs)
            torch._cuda_tracer.trace_operation(
                tensor_op.__name__,
                shapes
            )
            return result
        except Exception as e:
            print("\n!!! CUDA Operation Failed !!!")
            torch._cuda_tracer.print_last_operations()
            raise

    return wrapper


# Patch delle operazioni tensor comuni
def patch_torch_ops():
    ops_to_trace = [
        'matmul', 'mm', 'bmm', 'addmm', 'multiply', 'mul', 'add',
        'conv1d', 'conv2d', 'conv3d', 'linear'
    ]

    for op_name in ops_to_trace:
        if hasattr(torch.Tensor, op_name):
            original_op = getattr(torch.Tensor, op_name)
            setattr(torch.Tensor, op_name, trace_cuda_ops(original_op))
        if hasattr(torch, op_name):
            original_op = getattr(torch, op_name)
            setattr(torch, op_name, trace_cuda_ops(original_op))


@contextmanager
def cuda_trace():
    """Context manager per attivare il tracciamento CUDA."""
    if not hasattr(torch, '_cuda_tracer'):
        torch._cuda_tracer = CUDAOperationTracer()

    patch_torch_ops()
    original_excepthook = sys.excepthook

    def cuda_exception_hook(exctype, value, tb):
        if 'CUDA' in str(value):
            print("\n=== CUDA Error Detected ===")
            print(f"Error: {str(value)}")
            torch._cuda_tracer.print_last_operations()
        original_excepthook(exctype, value, tb)

    sys.excepthook = cuda_exception_hook

    try:
        yield torch._cuda_tracer
    finally:
        sys.excepthook = original_excepthook