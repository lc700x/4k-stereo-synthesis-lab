class EffectWorker:
    def __init__(self, viewer):
        self.viewer = viewer

    def prewarm_after_submit(self):
        prewarm = getattr(self.viewer, "_prewarm_runtime_effect_downsample", None)
        if callable(prewarm):
            prewarm()
