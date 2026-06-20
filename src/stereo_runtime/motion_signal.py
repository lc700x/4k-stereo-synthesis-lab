from __future__ import annotations


def clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


class RuntimeMotionSampler:
    def __init__(self):
        self.last_motion_frame = None
        self.pending_motion = None
        self.pending_motion_event = None
        self.last_motion_score = 0.0

    def sample(self, rgb_frame) -> float:
        try:
            import torch

            if self.pending_motion is not None:
                if self.pending_motion_event is None or self.pending_motion_event.query():
                    self.last_motion_score = clamp01(float(self.pending_motion.item()) * 4.0)
                    self.pending_motion = None
                    self.pending_motion_event = None

            frame = rgb_frame.detach()
            if frame.ndim == 4:
                frame = frame[0]
            if frame.ndim != 3:
                return self.last_motion_score
            if frame.shape[0] in (3, 4):
                frame = frame[:3]
            else:
                frame = frame[..., :3].permute(2, 0, 1)
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0).float(),
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            )[0]
            if self.last_motion_frame is None:
                self.last_motion_frame = frame.detach()
                return self.last_motion_score
            motion_tensor = (frame - self.last_motion_frame).abs().mean()
            self.last_motion_frame = frame.detach()
            if motion_tensor.is_cuda:
                if self.pending_motion is None:
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(motion_tensor.device))
                    self.pending_motion = motion_tensor.detach()
                    self.pending_motion_event = event
            else:
                self.last_motion_score = clamp01(float(motion_tensor.item()) * 4.0)
                self.pending_motion = None
                self.pending_motion_event = None
            return self.last_motion_score
        except Exception:
            return self.last_motion_score
