from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable
from stereo_runtime.output_convert import runtime_output_to_numpy


@dataclass
class LegacyStreamConfig:
    stream_port: int
    fps: int
    stream_quality: int
    time_sleep: float


@dataclass
class LegacyStreamCallbacks:
    shutdown_is_set: Callable
    now: Callable = time.perf_counter


def create_legacy_streamer(config: LegacyStreamConfig):
    from streaming.mjpeg_streamer import MJPEGStreamer

    streamer = MJPEGStreamer(
        port=config.stream_port,
        fps=config.fps,
        quality=config.stream_quality,
    )
    streamer.start()
    print("[Main] Legacy Streamer Started")
    return streamer


def run_legacy_stream_mode(runtime_q, config: LegacyStreamConfig, callbacks: LegacyStreamCallbacks, stats):
    streamer = create_legacy_streamer(config)
    while not callbacks.shutdown_is_set():
        try:
            runtime_result, _ = runtime_q.get(timeout=config.time_sleep)
            streamer.set_frame(runtime_output_to_numpy(runtime_result.sbs))

            current_time = callbacks.now()
            if stats.record_frame(current_time):
                print(
                    f"{stats.current_fps:.1f} FPS | "
                    f"Avg: {stats.avg_fps:.1f} | "
                    f"1% Low Avg: {stats.low_fps_avg:.1f}"
                )

        except queue.Empty:
            continue
        except Exception as exc:
            if not callbacks.shutdown_is_set():
                print(f"Streamer error: {exc}")
            break
    return streamer
