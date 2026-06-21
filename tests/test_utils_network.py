from utils import network


def test_configure_huggingface_endpoint_disables_symlink_usage(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(network, "is_cn_ip", lambda: False)

    endpoint = network.configure_huggingface_endpoint()

    assert endpoint == "https://huggingface.co"
    assert network.os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert network.os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert network.os.environ["HF_ENDPOINT"] == "https://huggingface.co"


def test_configure_huggingface_endpoint_uses_cn_mirror(monkeypatch):
    monkeypatch.setattr(network, "is_cn_ip", lambda: True)

    endpoint = network.configure_huggingface_endpoint()

    assert endpoint == "https://hf-mirror.com"
    assert network.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
