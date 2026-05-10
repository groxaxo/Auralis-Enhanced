import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _parse_xtts_module() -> ast.Module:
    return ast.parse(
        (ROOT / "src/auralis/models/xttsv2/XTTSv2.py").read_text(encoding="utf-8")
    )


def _get_class_method(module: ast.Module, class_name: str, method_name: str):
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name == method_name
                ):
                    return method
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_speaker_cache_uses_OrderedDict():
    module = _parse_xtts_module()
    init_method = _get_class_method(module, "XTTSv2Engine", "__init__")

    ordered_dict_assignments = [
        ast.unparse(node.value)
        for node in ast.walk(init_method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if ast.unparse(target) == "self._speaker_embedding_cache"
    ]

    assert ordered_dict_assignments == ["OrderedDict()"]


def test_speaker_cache_hits_refresh_lru_position():
    module = _parse_xtts_module()
    method = _get_class_method(module, "XTTSv2Engine", "get_conditioning_latents")

    move_to_end_calls = [
        ast.unparse(node)
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "self._speaker_embedding_cache.move_to_end"
    ]

    assert "self._speaker_embedding_cache.move_to_end(cache_key)" in move_to_end_calls


def test_speaker_cache_evicts_least_recently_used_entry():
    module = _parse_xtts_module()
    method = _get_class_method(module, "XTTSv2Engine", "get_conditioning_latents")

    popitem_calls = [
        ast.unparse(node)
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "self._speaker_embedding_cache.popitem"
    ]

    assert "self._speaker_embedding_cache.popitem(last=False)" in popitem_calls


def test_speaker_cache_stores_audio_on_cpu():
    module = _parse_xtts_module()
    method = _get_class_method(module, "XTTSv2Engine", "get_conditioning_latents")

    cpu_audio_values = [
        ast.unparse(value)
        for node in ast.walk(method)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "audio"
    ]

    assert "audio.cpu()" in cpu_audio_values


def test_conditioning_audio_is_moved_to_device_after_concat():
    module = _parse_xtts_module()
    method = _get_class_method(module, "XTTSv2Engine", "get_conditioning_latents")

    full_audio_assignments = [
        ast.unparse(node.value)
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if ast.unparse(target) == "full_audio"
    ]

    assert "torch.cat(audios, dim=-1).to(self.device)" in full_audio_assignments
