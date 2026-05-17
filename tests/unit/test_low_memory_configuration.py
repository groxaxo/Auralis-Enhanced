import ast
import hashlib
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple, Union


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GPU_MEMORY_UTILIZATION = 0.35
DEFAULT_CPU_OFFLOAD_GB = 8.0
DEFAULT_SWAP_SPACE = 2.0


def _parse_module(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def _get_class_method(module: ast.Module, class_name: str, method_name: str):
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == method_name:
                    return method
    raise AssertionError(f"{class_name}.{method_name} not found")


def _get_function(module: ast.Module, function_name: str):
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found")


def _load_xtts_cache_key_harness():
    source = (
        ROOT / "src" / "auralis" / "models" / "xttsv2" / "XTTSv2.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)

    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "XTTSv2Engine"
    )
    helper_sources = []
    for method in class_node.body:
        if (
            isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name
            in {"_normalize_audio_reference_for_cache", "_get_conditioning_cache_key"}
        ):
            helper_sources.append(textwrap.indent(ast.unparse(method), "    "))

    namespace = {
        "hashlib": hashlib,
        "Path": Path,
        "Union": Union,
        "List": List,
        "Tuple": Tuple,
        "Optional": Optional,
    }
    exec("class CacheKeyHarness:\n" + "\n\n".join(helper_sources), namespace)
    return namespace["CacheKeyHarness"]()


def _literal_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"Expected literal constant, got {ast.dump(node)}")


def test_tts_defaults_to_single_scheduler_concurrency():
    module = _parse_module("src/auralis/core/tts.py")
    init_method = _get_class_method(module, "TTS", "__init__")

    scheduler_arg = next(
        arg for arg in init_method.args.args if arg.arg == "scheduler_max_concurrency"
    )
    default_index = init_method.args.args.index(scheduler_arg) - (
        len(init_method.args.args) - len(init_method.args.defaults)
    )
    assert _literal_value(init_method.args.defaults[default_index]) == 1


def test_streaming_path_does_not_accumulate_chunks():
    module = _parse_module("src/auralis/core/tts.py")
    method = _get_class_method(module, "TTS", "generate_speech_async")
    process_chunks = next(
        node for node in method.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "process_chunks"
    )

    append_calls = [
        node for node in ast.walk(process_chunks)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "chunks"
        and node.func.attr == "append"
    ]
    assert len(append_calls) == 1

    append_call = append_calls[0]
    stream_if = next(
        node for node in ast.walk(process_chunks)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "request"
        and node.test.attr == "stream"
    )

    assert any(
        append_call in ast.walk(else_node) for else_node in stream_if.orelse
    ), "chunks.append should only run in the non-streaming branch"
    assert not any(
        append_call in ast.walk(body_node) for body_node in stream_if.body
    ), "chunks.append must not run while streaming"


def test_server_exposes_low_memory_cli_defaults():
    module = _parse_module("src/auralis/entrypoints/oai_server.py")
    main_fn = _get_function(module, "main")

    add_argument_calls = [
        node for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]

    defaults = {}
    for call in add_argument_calls:
        option = _literal_value(call.args[0])
        for keyword in call.keywords:
            if keyword.arg == "default":
                defaults[option] = _literal_value(keyword.value)

    assert defaults["--max_concurrency"] == 1
    assert defaults["--device"] == "auto"
    assert defaults["--gpu_memory_utilization"] == DEFAULT_GPU_MEMORY_UTILIZATION
    assert defaults["--cpu_offload_gb"] == DEFAULT_CPU_OFFLOAD_GB
    assert defaults["--swap_space"] == DEFAULT_SWAP_SPACE


def test_server_passes_device_and_memory_knobs_to_model_loading():
    module = _parse_module("src/auralis/entrypoints/oai_server.py")
    ensure_tts_engine = _get_function(module, "ensure_tts_engine")

    create_tts_engine = next(
        node for node in ast.walk(ensure_tts_engine)
        if isinstance(node, ast.FunctionDef) and node.name == "create_tts_engine"
    )
    from_pretrained_call = next(
        node for node in ast.walk(create_tts_engine)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    )

    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in from_pretrained_call.keywords}
    assert keywords["device_map"] == "args.device"
    assert keywords["max_concurrency"] == "scheduler_concurrency"
    assert keywords["gpu_memory_utilization"] == "args.gpu_memory_utilization"
    assert keywords["cpu_offload_gb"] == "args.cpu_offload_gb"
    assert keywords["swap_space"] == "args.swap_space"


def test_xtts_from_pretrained_respects_requested_device():
    module = _parse_module("src/auralis/models/xttsv2/XTTSv2.py")
    method = _get_class_method(module, "XTTSv2Engine", "from_pretrained")

    to_calls = [
        ast.unparse(node.args[0])
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
        and node.args
    ]

    assert '"cuda"' not in to_calls
    assert "target_device" in to_calls

    assignments = [
        node for node in method.body
        if isinstance(node, ast.Assign)
    ]
    assert any(
        any(isinstance(target, ast.Subscript) and ast.unparse(target) == "kwargs['device_map']" for target in assign.targets)
        and ast.unparse(assign.value) == "device_map"
        for assign in assignments
    )


def test_xtts_vllm_engine_configuration_has_cpu_and_gpu_paths():
    module = _parse_module("src/auralis/models/xttsv2/XTTSv2.py")
    method = _get_class_method(module, "XTTSv2Engine", "init_vllm_engine")

    assert any(
        isinstance(node, ast.If) and ast.unparse(node.test) == "self.is_cpu"
        for node in ast.walk(method)
    ), "Expected explicit CPU handling in init_vllm_engine"

    subscript_assignments = {
        ast.unparse(target): ast.unparse(node.value)
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
    }

    assert subscript_assignments["engine_kwargs['swap_space']"] == "self.swap_space"
    assert subscript_assignments["engine_kwargs['cpu_offload_gb']"] == "self.cpu_offload_gb"
    assert subscript_assignments["engine_kwargs['gpu_memory_utilization']"] == "mem_utilization"


def test_xtts_conditioning_perceiver_keeps_sdp_attention_enabled_on_cuda():
    module = _parse_module("src/auralis/models/xttsv2/XTTSv2.py")
    init_method = _get_class_method(module, "XTTSv2Engine", "__init__")

    assignments = {
        ast.unparse(target): ast.unparse(node.value)
        for node in ast.walk(init_method)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["use_flash_attn"] == "not self.is_cpu"
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_device_properties"
        for node in ast.walk(init_method)
    )


def test_xtts_conditioning_cache_uses_lru_and_compact_reference_keys():
    module = _parse_module("src/auralis/models/xttsv2/XTTSv2.py")
    init_method = _get_class_method(module, "XTTSv2Engine", "__init__")
    normalize_method = _get_class_method(
        module, "XTTSv2Engine", "_normalize_audio_reference_for_cache"
    )
    conditioning_method = _get_class_method(
        module, "XTTSv2Engine", "get_audio_conditioning"
    )

    assert any(
        any(
            isinstance(target, ast.Attribute)
            and ast.unparse(target) == "self._conditioning_cache"
            for target in node.targets
        )
        and ast.unparse(node.value) == "OrderedDict()"
        for node in ast.walk(init_method)
        if isinstance(node, ast.Assign)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "hashlib.blake2b"
        for node in ast.walk(normalize_method)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sorted"
        for node in ast.walk(conditioning_method)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "move_to_end"
        for node in ast.walk(conditioning_method)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "popitem"
        and any(
            keyword.arg == "last" and ast.unparse(keyword.value) == "False"
            for keyword in node.keywords
        )
        for node in ast.walk(conditioning_method)
    )


def test_xtts_conditioning_cache_key_preserves_order_at_runtime():
    harness = _load_xtts_cache_key_harness()

    first_order_key = harness._get_conditioning_cache_key(
        ["speaker_b.wav", "speaker_a.wav"], 30, 6, 6, None, False, 22050
    )
    second_order_key = harness._get_conditioning_cache_key(
        ["speaker_a.wav", "speaker_b.wav"], 30, 6, 6, None, False, 22050
    )
    bytes_key = harness._get_conditioning_cache_key(
        [b"raw-audio"], 30, 6, 6, None, False, 22050
    )

    assert first_order_key[0] == (
        ("ref", "speaker_b.wav"),
        ("ref", "speaker_a.wav"),
    )
    assert second_order_key[0] == (
        ("ref", "speaker_a.wav"),
        ("ref", "speaker_b.wav"),
    )
    assert first_order_key != second_order_key
    assert bytes_key[0][0][0] == "bytes"
    assert bytes_key[0][0][1] != "raw-audio"
    assert len(bytes_key[0][0][1]) == 32
