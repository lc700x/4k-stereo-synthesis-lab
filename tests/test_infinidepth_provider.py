import pytest

from stereo_runtime.depth_provider import DepthProviderConfig, InfiniDepthProvider, create_depth_provider
from stereo_runtime.model_registry import ModelRegistry


@pytest.mark.parametrize(
    ("model_id", "encoder"),
    [
        ("lc700x/InfiniDepth-Small", "vits16"),
        ("lc700x/InfiniDepth-SmallPlus", "vits16plus"),
        ("lc700x/InfiniDepth-Base", "vitb16"),
        ("lc700x/InfiniDepth-Large", "vitl16"),
    ],
)
def test_infinidepth_models_use_specialized_provider(model_id, encoder):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_cuda",
            model_id=model_id,
            model_name=model_id.rsplit("/", 1)[-1],
            device="cpu",
            local_files_only=True,
        )
    )

    assert isinstance(provider, InfiniDepthProvider)
    assert provider.encoder == encoder
    assert provider.info.provider == "models.InfiniDepth.api.InfiniDepthModel"
    assert provider.info.runtime == "infinidepth"


def test_model_registry_contains_infinidepth_family():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("InfiniDepth-Small") == "lc700x/InfiniDepth-Small"
    assert registry.resolve_model_id("InfiniDepth-SmallPlus") == "lc700x/InfiniDepth-SmallPlus"
    assert registry.resolve_model_id("InfiniDepth-Base") == "lc700x/InfiniDepth-Base"
    assert registry.resolve_model_id("InfiniDepth-Large") == "lc700x/InfiniDepth-Large"
