import ast
from pathlib import Path


def _get_process_tokens_to_speech_method():
    xtts_file = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "auralis"
        / "models"
        / "xttsv2"
        / "XTTSv2.py"
    )
    source = xtts_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "XTTSv2Engine":
            for method in node.body:
                if (
                    isinstance(method, ast.AsyncFunctionDef)
                    and method.name == "process_tokens_to_speech"
                ):
                    return method
    raise AssertionError("XTTSv2Engine.process_tokens_to_speech not found")


def _is_cuda_memory_manager_with(node: ast.AsyncWith) -> bool:
    for item in node.items:
        ctx = item.context_expr
        if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
            if ctx.func.attr == "cuda_memory_manager":
                return True
    return False


def test_process_tokens_to_speech_yield_is_outside_cuda_cleanup_context():
    method = _get_process_tokens_to_speech_method()

    cuda_with_ranges = [
        (node.lineno, node.end_lineno)
        for node in ast.walk(method)
        if isinstance(node, ast.AsyncWith) and _is_cuda_memory_manager_with(node)
    ]
    assert cuda_with_ranges, "Expected a cuda_memory_manager context in method"

    yield_lines = [
        node.lineno for node in ast.walk(method) if isinstance(node, ast.Yield)
    ]
    assert yield_lines, "Expected at least one yield in method"

    for line in yield_lines:
        assert not any(start <= line <= end for start, end in cuda_with_ranges), (
            "yield must be outside cuda_memory_manager context so CUDA cleanup can run"
        )


def test_process_tokens_to_speech_releases_large_tensors_before_yield():
    method = _get_process_tokens_to_speech_method()
    deleted_names = set()

    for node in ast.walk(method):
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    deleted_names.add(target.id)

    assert "wav_tensor" in deleted_names
    assert "hidden_states" in deleted_names
