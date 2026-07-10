from auralis.backends import selection


def test_explicit_backend_aliases():
    assert selection.resolve_backend("mlx") == "mlx"
    assert selection.resolve_backend("metal") == "mlx"
    assert selection.resolve_backend("cuda") == "vllm"


def test_auto_selects_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.delenv("AURALIS_BACKEND", raising=False)
    monkeypatch.setattr(selection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(selection.platform, "machine", lambda: "arm64")
    assert selection.resolve_backend("auto") == "mlx"


def test_environment_override(monkeypatch):
    monkeypatch.setenv("AURALIS_BACKEND", "vllm")
    assert selection.resolve_backend("auto") == "vllm"
