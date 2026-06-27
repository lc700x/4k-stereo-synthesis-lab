from __future__ import annotations

import pytest
import torch

from stereo_runtime import StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult
from stereo_runtime.settings_snapshot import (
    RuntimeSettingsRestartRequired,
    RuntimeSettingsSnapshot,
    SnapshotChangeClass,
)


class FakeDepthProvider:
    def __init__(self):
        self.closed = False
        self.loaded = False

    def load(self):
        self.loaded = True

    def predict_profile(self, rgb_frame):
        height = int(rgb_frame.shape[-2])
        width = int(rgb_frame.shape[-1])
        depth = torch.zeros((1, 1, height, width), dtype=torch.float32, device=rgb_frame.device)
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=0.0, postprocess_ms=0.0)

    def close(self):
        self.closed = True


def test_settings_snapshot_classifies_change_levels():
    assert RuntimeSettingsSnapshot(version=1, timestamp=1.0).classify() is SnapshotChangeClass.NO_CHANGE
    assert RuntimeSettingsSnapshot(version=2, timestamp=1.0, temporal_strength=0.5).classify() is SnapshotChangeClass.HOT_RELOAD
    assert RuntimeSettingsSnapshot(version=3, timestamp=1.0, depth_backend="pytorch_cuda").classify() is SnapshotChangeClass.PIPELINE_REBUILD
    assert RuntimeSettingsSnapshot(version=4, timestamp=1.0, device="cuda:1").classify() is SnapshotChangeClass.SESSION_RESTART


def test_settings_snapshot_classifies_spec_layer_fields():
    assert (
        RuntimeSettingsSnapshot(version=5, timestamp=1.0, runtime_quality_mode="movie").classify()
        is SnapshotChangeClass.HOT_RELOAD
    )
    assert (
        RuntimeSettingsSnapshot(version=6, timestamp=1.0, render_size_policy="scaled").classify()
        is SnapshotChangeClass.PIPELINE_REBUILD
    )
    assert (
        RuntimeSettingsSnapshot(version=7, timestamp=1.0, stereo_render_scale=0.5).classify()
        is SnapshotChangeClass.PIPELINE_REBUILD
    )
    assert (
        RuntimeSettingsSnapshot(version=8, timestamp=1.0, stereo_synthesis_mode="full_synthesis_eyes").classify()
        is SnapshotChangeClass.PIPELINE_REBUILD
    )
    assert (
        RuntimeSettingsSnapshot(version=9, timestamp=1.0, application_runtime_target="openxr").classify()
        is SnapshotChangeClass.SESSION_RESTART
    )


def test_settings_snapshot_excludes_non_config_spec_layer_fields_from_runtime_updates():
    snapshot = RuntimeSettingsSnapshot(
        version=10,
        timestamp=1.0,
        runtime_quality_mode="movie",
        render_size_policy="scaled",
        stereo_render_scale=0.5,
        stereo_synthesis_mode="full_synthesis_eyes",
        output_transport="openxr_swapchain",
        presentation_flags={"cross_eyed": True},
        debug_flags={"depth": True},
        depth_strength=1.25,
    )

    updates = snapshot.to_config_updates()

    assert updates == {"depth_strength": 1.25}


def test_settings_snapshot_maps_ipd_mm_to_runtime_ipd():
    snapshot = RuntimeSettingsSnapshot(version=5, timestamp=1.0, ipd_mm=64.0, depth_strength=1.25)

    updates = snapshot.to_config_updates()

    assert updates["ipd_mm"] == 64.0
    assert updates["ipd"] == 0.064
    assert updates["depth_strength"] == 1.25


def test_settings_snapshot_maps_parallax_budget_fields():
    snapshot = RuntimeSettingsSnapshot(
        version=11,
        timestamp=1.0,
        max_disparity_px=96.0,
        parallax_preset="standard",
    )

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD
    assert updates["max_disparity_px"] == 96.0
    assert updates["parallax_preset"] == "standard"


def test_settings_snapshot_maps_temporal_reset_fields():
    snapshot = RuntimeSettingsSnapshot(
        version=12,
        timestamp=1.0,
        auto_reset_temporal=True,
        scene_reset_threshold=0.33,
        reset_cooldown_frames=9,
    )

    updates = snapshot.to_config_updates()

    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD
    assert updates["auto_reset_temporal"] is True
    assert updates["scene_reset_threshold"] == 0.33
    assert updates["reset_cooldown_frames"] == 9


def test_runtime_applies_hot_settings_snapshot_without_rebuild():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models", stereo_preset="cinema"),
        depth_provider=FakeDepthProvider(),
    )
    snapshot = RuntimeSettingsSnapshot(
        version=12,
        timestamp=1.0,
        depth_strength=1.5,
        temporal_strength=0.4,
        ipd_mm=60.0,
    )

    change_class = runtime.apply_settings_snapshot(snapshot, active_preset="cinema")

    assert change_class is SnapshotChangeClass.HOT_RELOAD
    assert runtime.config.depth_strength == 1.5
    assert runtime.config.temporal_strength == 0.4
    assert runtime.config.ipd == pytest.approx(0.06)
    assert runtime.active_settings_version == 12


def test_runtime_result_debug_info_tracks_active_settings_snapshot_fields():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    provider = runtime.depth_provider
    runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(
            version=12,
            timestamp=1.0,
            runtime_quality_mode="movie",
            stereo_synthesis_mode="full_synthesis_eyes",
            render_size_policy="scaled",
            stereo_render_scale=0.5,
            output_transport="openxr_swapchain",
        )
    )
    assert runtime.depth_provider is provider
    runtime.apply_settings_snapshot(RuntimeSettingsSnapshot(version=13, timestamp=2.0, depth_strength=1.2))

    result = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    assert result.debug_info["active_settings_version"] == 13
    assert result.debug_info["hot_reload_class"] == SnapshotChangeClass.HOT_RELOAD.value
    assert result.debug_info["hot_reload_changed_fields"] == ["depth_strength"]
    assert result.debug_info["runtime_quality_mode"] == "movie"
    assert result.debug_info["stereo_synthesis_mode"] == "full_synthesis_eyes"
    assert result.debug_info["render_size_policy"] == "scaled"
    assert result.debug_info["stereo_render_scale"] == 0.5
    assert result.debug_info["output_transport"] == "openxr_swapchain"


def test_runtime_result_debug_info_tracks_active_settings_version():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
        collect_memory_stats=False,
    )
    runtime.apply_settings_snapshot(RuntimeSettingsSnapshot(version=13, timestamp=1.0, depth_strength=1.2))

    result = runtime.process_openxr_frame(torch.zeros((1, 3, 2, 2), dtype=torch.float32))

    assert result.debug_info["active_settings_version"] == 13
    assert result.debug_info["hot_reload_class"] == SnapshotChangeClass.HOT_RELOAD.value
    assert result.debug_info["hot_reload_changed_fields"] == ["depth_strength"]


def test_runtime_rejects_session_restart_snapshot():
    runtime = StereoRuntime(
        StereoRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="models"),
        depth_provider=FakeDepthProvider(),
    )
    snapshot = RuntimeSettingsSnapshot(version=14, timestamp=1.0, device="cuda:1")

    with pytest.raises(RuntimeSettingsRestartRequired):
        runtime.apply_settings_snapshot(snapshot)
