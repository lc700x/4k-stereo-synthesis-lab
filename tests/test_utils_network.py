from utils import network


def test_configure_huggingface_endpoint_disables_symlink_usage(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(network, "is_cn_ip", lambda: False)

    endpoint = network.configure_huggingface_endpoint(async_probe=False)

    assert endpoint == "https://huggingface.co"
    assert network.os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert network.os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert network.os.environ["HF_ENDPOINT"] == "https://huggingface.co"


def test_configure_huggingface_endpoint_uses_cn_mirror(monkeypatch):
    monkeypatch.setattr(network, "is_cn_ip", lambda: True)

    endpoint = network.configure_huggingface_endpoint(async_probe=False)

    assert endpoint == "https://hf-mirror.com"
    assert network.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_configure_huggingface_endpoint_default_does_not_block_on_probe(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(network, "_HF_ENDPOINT_PROBE_STARTED", False)
    calls = []

    def fail_if_called():
        calls.append("called")
        raise AssertionError("network probe should run in background")

    monkeypatch.setattr(network, "is_cn_ip", fail_if_called)
    monkeypatch.setattr(network, "_start_huggingface_endpoint_probe_once", lambda: None)

    endpoint = network.configure_huggingface_endpoint()

    assert endpoint == "https://hf-mirror.com"
    assert network.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert calls == []
