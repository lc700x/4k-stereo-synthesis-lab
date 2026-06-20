from __future__ import annotations

import threading

from capture.types import CaptureConfig
from capture.session import CaptureSessionCallbacks, CaptureSessionLoop


class FakeRunner:
    def __init__(self, config):
        self.config = config

    def run(
        self,
        *,
        shutdown_event,
        on_frame,
        on_error=None,
        on_closed=None,
        is_paused=None,
        is_hard_idle=None,
        on_paused=None,
        on_session_update=None,
        on_tick=None,
    ):
        on_session_update("session", "control")
        on_tick()
        on_paused("paused")
        on_frame("frame", (16, 9), 1.25)
        on_error(RuntimeError("boom"))
        on_closed()


def test_capture_session_loop_wires_runner_callbacks(monkeypatch):
    events = []
    stats = {}
    breakdown = {}

    def fake_create_capture_runner(config):
        events.append(("runner_config", config.window_title))
        return FakeRunner(config)

    def inc_source_stat(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def inc_breakdown(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    def put_raw_latest(item):
        events.append(("put", item))
        return True

    monkeypatch.setattr("capture.session.create_capture_runner", fake_create_capture_runner)

    callbacks = CaptureSessionCallbacks(
        clear_raw_queue=lambda: events.append(("clear_raw",)),
        inc_source_stat=inc_source_stat,
        inc_breakdown=inc_breakdown,
        put_raw_latest=put_raw_latest,
        is_shutdown=lambda: False,
        is_paused=lambda: False,
        is_hard_idle=lambda: False,
        on_session_update=lambda session, control: events.append(("session", session, control)),
        on_tick=lambda: events.append(("tick",)),
    )
    config = CaptureConfig(
        output_resolution=(16, 9),
        fps=60,
        window_title="Example",
        capture_mode="Window",
        monitor_index=0,
        capture_tool="Fake",
        os_name="Windows",
    )

    CaptureSessionLoop(config, callbacks).run(threading.Event())

    assert ("runner_config", "Example") in events
    assert ("session", "session", "control") in events
    assert ("tick",) in events
    assert ("clear_raw",) in events
    assert ("put", ("frame", (16, 9), 1.25)) in events
    assert stats["capture_dropped_paused"] == 1
    assert stats["capture_frames"] == 1
    assert stats["last_capture_ts"] == 1.25
    assert stats["raw_overwritten"] == 1
    assert stats["raw_put"] == 1
    assert stats["capture_errors"] == 1
    assert "RuntimeError: boom" in stats["last_error"]
    assert breakdown["capture"] == 1
    assert breakdown["raw_overwritten"] == 1
