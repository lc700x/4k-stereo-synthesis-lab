import moderngl


class AsyncEffectResultPool:
    def __init__(self):
        self.staging_tex = None
        self.staging_size = None
        self.ready_tex = None
        self.ready_size = None
        self.ready_frame_id = 0
        self.safe_tex = None
        self.safe_size = None
        self.spare_tex = None
        self.spare_size = None
        self.safe_frame_id = 0
        self.state = 'idle'

    def release(self):
        released = set()
        for tex in (self.staging_tex, self.ready_tex, self.safe_tex, self.spare_tex):
            if tex is not None and id(tex) not in released:
                released.add(id(tex))
                try:
                    tex.release()
                except Exception:
                    pass
        self.staging_tex = None
        self.staging_size = None
        self.ready_tex = None
        self.ready_size = None
        self.ready_frame_id = 0
        self.safe_tex = None
        self.safe_size = None
        self.spare_tex = None
        self.spare_size = None
        self.safe_frame_id = 0
        self.state = 'idle'

    def ensure_staging(self, ctx, w, h):
        if self.staging_size == (w, h) and self.staging_tex is not None:
            self.state = 'writing'
            return self.staging_tex
        if self.staging_tex is not None:
            try:
                self.staging_tex.release()
            except Exception:
                pass
        tex = ctx.texture((w, h), 4, dtype='f1')
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.staging_tex = tex
        self.staging_size = (w, h)
        self.state = 'writing'
        return tex

    def mark_ready(self, w, h, frame_id):
        old_ready_tex = self.ready_tex
        old_ready_size = self.ready_size
        if old_ready_tex is not None and old_ready_tex is not self.staging_tex:
            if self.spare_tex is None:
                self.spare_tex = old_ready_tex
                self.spare_size = old_ready_size
            else:
                try:
                    old_ready_tex.release()
                except Exception:
                    pass
        self.ready_tex = self.staging_tex
        self.ready_size = (w, h)
        self.ready_frame_id = int(frame_id or 0)
        self.staging_tex = None
        self.staging_size = None
        self.state = 'ready'

    def promote_ready(self):
        if self.ready_tex is None:
            self.state = 'safe' if self.safe_tex is not None else 'idle'
            return False
        old_safe_tex = self.safe_tex
        old_safe_size = self.safe_size
        self.safe_tex = self.ready_tex
        self.safe_size = self.ready_size
        self.safe_frame_id = self.ready_frame_id
        self.ready_tex = None
        self.ready_size = None
        self.ready_frame_id = 0
        self.staging_tex = self.spare_tex
        self.staging_size = self.spare_size
        self.spare_tex = old_safe_tex
        self.spare_size = old_safe_size
        self.state = 'safe'
        return True

    def publish(self, w, h, frame_id):
        self.mark_ready(w, h, frame_id)
        return True


class EffectScheduler:
    def __init__(self, pool=None):
        self.pool = pool or AsyncEffectResultPool()
        self.pending_source = None

    def queue_source(self, source):
        overwritten = self.pending_source is not None
        self.pending_source = source
        return overwritten

    def clear_pending_source(self):
        self.pending_source = None

    def flush_pending_source(self, submit_source, promote_ready=None):
        source = self.pending_source
        if source is None:
            return 'empty'
        submitted = submit_source(source)
        if submitted is False:
            return 'skipped'
        if callable(promote_ready):
            promote_ready()
        self.pending_source = None
        return 'submitted'

    def ensure_staging(self, ctx, w, h):
        return self.pool.ensure_staging(ctx, w, h)

    def submit_screen_frame(self, ctx, w, h):
        return self.ensure_staging(ctx, w, h)

    def publish_completed(self, w, h, frame_id):
        return self.pool.publish(w, h, frame_id)

    def poll_completed(self):
        return self.pool.promote_ready()

    def latest_safe(self):
        return self.pool.safe_tex, self.pool.safe_size, self.pool.safe_frame_id

    def latest_safe_glow(self):
        return self.latest_safe()

    def latest_safe_light_probe(self):
        return self.latest_safe()

    def release(self):
        self.pool.release()
