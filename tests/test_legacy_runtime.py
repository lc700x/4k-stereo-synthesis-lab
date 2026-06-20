import queue
from types import SimpleNamespace

from streaming.legacy_runtime import LegacyStreamCallbacks, LegacyStreamConfig, run_legacy_stream_mode


class OneShotShutdown:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 1


class FakeStats:
    current_fps = 60.0
    avg_fps = 60.0
    low_fps_avg = 55.0

    def __init__(self):
        self.frames = 0

    def record_frame(self, now):
        self.frames += 1
        return True


def test_run_legacy_stream_mode_sets_one_frame(monkeypatch):
    frames = []

    class FakeStreamer:
        def start(self):
            pass

        def set_frame(self, frame):
            frames.append(frame)

    monkeypatch.setattr(
        "streaming.mjpeg_streamer.MJPEGStreamer",
        lambda **kwargs: FakeStreamer(),
    )
    monkeypatch.setattr(
        "streaming.legacy_runtime.runtime_output_to_numpy",
        lambda frame: f"numpy-{frame}",
    )
    runtime_q = queue.Queue()
    runtime_q.put((SimpleNamespace(sbs="sbs"), 0.0))
    stats = FakeStats()

    streamer = run_legacy_stream_mode(
        runtime_q,
        LegacyStreamConfig(stream_port=8000, fps=60, stream_quality=80, time_sleep=0.01),
        LegacyStreamCallbacks(shutdown_is_set=OneShotShutdown().is_set, now=lambda: 1.0),
        stats,
    )

    assert streamer is not None
    assert frames == ["numpy-sbs"]
    assert stats.frames == 1
