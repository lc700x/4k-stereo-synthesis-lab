import moderngl
from dataclasses import dataclass


@dataclass
class EffectResultSlot:
    tex: object = None
    size: tuple[int, int] | None = None
    frame_id: int = 0
    state: str = 'idle'


class AsyncEffectResultPool:
    def __init__(self):
        self.slots = [EffectResultSlot() for _ in range(3)]
        self.writing_slot = None
        self.ready_slot = None
        self.safe_slot = None
        self._safe_frame_id = 0
        self.state = 'idle'

    def release(self):
        released = set()
        for slot in self.slots:
            tex = slot.tex
            if tex is not None and id(tex) not in released:
                released.add(id(tex))
                try:
                    tex.release()
                except Exception:
                    pass
            slot.tex = None
            slot.size = None
            slot.frame_id = 0
            slot.state = 'idle'
        self.writing_slot = None
        self.ready_slot = None
        self.safe_slot = None
        self._safe_frame_id = 0
        self.state = 'idle'

    def ensure_staging(self, ctx, w, h):
        size = (int(w), int(h))
        if self.writing_slot is not None and self.writing_slot.size == size and self.writing_slot.tex is not None:
            self.state = 'writing'
            return self.writing_slot.tex
        slot = self._idle_slot()
        if slot.tex is not None and slot.size != size:
            try:
                slot.tex.release()
            except Exception:
                pass
            slot.tex = None
        if slot.tex is None:
            slot.tex = ctx.texture(size, 4, dtype='f1')
            slot.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        slot.size = size
        slot.frame_id = 0
        slot.state = 'writing'
        self.writing_slot = slot
        self.state = 'writing'
        return slot.tex

    def mark_ready(self, w, h, frame_id):
        if self.writing_slot is None:
            return False
        if self.ready_slot is not None and self.ready_slot is not self.writing_slot:
            self.ready_slot.state = 'idle'
        slot = self.writing_slot
        slot.size = (int(w), int(h))
        slot.frame_id = int(frame_id or 0)
        slot.state = 'ready'
        self.ready_slot = slot
        self.writing_slot = None
        self.state = 'ready'
        return True

    def promote_ready(self):
        if self.ready_slot is None:
            self.state = 'safe' if self.safe_slot is not None else 'idle'
            return False
        if self.safe_slot is not None and self.safe_slot is not self.ready_slot:
            self.safe_slot.state = 'idle'
        self.safe_slot = self.ready_slot
        self.safe_slot.state = 'safe'
        self._safe_frame_id = self.safe_slot.frame_id
        self.ready_slot = None
        self.state = 'safe'
        return True

    def publish(self, w, h, frame_id):
        return self.mark_ready(w, h, frame_id)

    def _idle_slot(self):
        for slot in self.slots:
            if slot.state == 'idle':
                return slot
        if self.ready_slot is not None:
            slot = self.ready_slot
            self.ready_slot = None
            slot.state = 'idle'
            return slot
        for slot in self.slots:
            if slot is not self.safe_slot and slot is not self.writing_slot:
                slot.state = 'idle'
                return slot
        raise RuntimeError("no writable effect result slot")

    @property
    def staging_tex(self):
        return self.writing_slot.tex if self.writing_slot is not None else None

    @property
    def staging_size(self):
        return self.writing_slot.size if self.writing_slot is not None else None

    @property
    def ready_tex(self):
        return self.ready_slot.tex if self.ready_slot is not None else None

    @ready_tex.setter
    def ready_tex(self, tex):
        slot = self.ready_slot or self._idle_slot()
        slot.tex = tex
        slot.state = 'ready' if tex is not None else 'idle'
        self.ready_slot = slot if tex is not None else None

    @property
    def ready_size(self):
        return self.ready_slot.size if self.ready_slot is not None else None

    @property
    def ready_frame_id(self):
        return self.ready_slot.frame_id if self.ready_slot is not None else 0

    @property
    def safe_tex(self):
        return self.safe_slot.tex if self.safe_slot is not None else None

    @safe_tex.setter
    def safe_tex(self, tex):
        slot = self.safe_slot or self._idle_slot()
        slot.tex = tex
        slot.state = 'safe' if tex is not None else 'idle'
        self.safe_slot = slot if tex is not None else None

    @property
    def safe_size(self):
        return self.safe_slot.size if self.safe_slot is not None else None

    @safe_size.setter
    def safe_size(self, size):
        if self.safe_slot is None:
            self.safe_slot = self._idle_slot()
            self.safe_slot.state = 'safe'
        self.safe_slot.size = size

    @property
    def safe_frame_id(self):
        return self.safe_slot.frame_id if self.safe_slot is not None else self._safe_frame_id

    @safe_frame_id.setter
    def safe_frame_id(self, frame_id):
        self._safe_frame_id = int(frame_id or 0)
        if self.safe_slot is None:
            self.safe_slot = self._idle_slot()
            self.safe_slot.state = 'safe'
        self.safe_slot.frame_id = self._safe_frame_id


class EffectScheduler:
    def __init__(self, pool=None):
        self.pool = pool or AsyncEffectResultPool()
        self.pending_source = None
        self.promote_frame_id = -1

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

    def promote_ready_once(self, frame_id):
        frame_id = int(frame_id or 0)
        if self.promote_frame_id == frame_id:
            return 'reused'
        self.promote_frame_id = frame_id
        return 'promoted' if self.poll_completed() else 'unchanged'

    def latest_safe(self):
        return self.pool.safe_tex, self.pool.safe_size, self.pool.safe_frame_id

    def latest_safe_downsample(self, cached_downsample=None):
        source_tex, source_size, source_frame_id = self.latest_safe()
        if source_tex is None or source_size is None or not callable(cached_downsample):
            return None, None, source_frame_id
        tex = cached_downsample(source_tex, source_size)
        if tex is not None:
            return tex, None, source_frame_id
        return None, None, source_frame_id

    def latest_safe_glow(self):
        return self.latest_safe()

    def latest_safe_light_probe(self):
        return self.latest_safe()

    def release(self):
        self.pool.release()
