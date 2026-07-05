class EffectSubmitter:
    def __init__(self, viewer):
        self.viewer = viewer

    def flush_after_submit(self, *, should_render, screen_frame_uploaded):
        if not should_render or screen_frame_uploaded:
            return False
        viewer = self.viewer
        should_submit = getattr(viewer, '_should_submit_runtime_effect_source', None)
        if callable(should_submit) and not should_submit():
            return False
        viewer._flush_runtime_effect_submit()
        return True
