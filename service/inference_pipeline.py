"""N independent Python interpreters, at most K main BPU chains in flight.

Same scheduling idea as the original (latest frame only, bounded in-flight work,
ordered output), generalised:
  RAC_WORKERS  number of worker processes (default 2 = original)
  RAC_INFLIGHT max frames admitted before the previous one reached its CPU tail (default 1 = original)
  RAC_PRESCALE resize to <=1280 in the parent before pickling (identical math, 2.25x less IPC for 1080p)
  RAC_THREADS  cv2 threads per worker (default 4 = original)
"""
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path

WORKERS = int(os.environ.get('RAC_WORKERS', '2'))
INFLIGHT = int(os.environ.get('RAC_INFLIGHT', '1'))
PRESCALE = os.environ.get('RAC_PRESCALE', '1') == '1'


def prescale(frame):
    import cv2
    scale = min(1, 1280 / max(frame.shape[:2]))
    return cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame


def render_worker(slot, tasks, events):
    import sys
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'vendor'))
    import cv2
    from board_engine import BoardCamera
    camera = BoardCamera()
    cv2.setNumThreads(int(os.environ.get('RAC_THREADS', '4')))
    camera.scene_size = 960
    events.put(('init', slot))
    while True:
        task = tasks.get()
        if task is None:
            return
        seq, frame, submitted = task
        signaled = False

        def ready():
            nonlocal signaled
            if not signaled:
                events.put(('stage', slot, seq))
                signaled = True

        camera.on_bpu_chain_done = ready
        try:
            small = prescale(frame)  # no-op when the parent already prescaled
            begin = time.perf_counter()
            rgb = camera.render(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            processing_ms = (time.perf_counter() - begin) * 1000
            rgb = cv2.flip(rgb, 1)
            metrics = {
                'processing_ms': processing_ms,
                'latency_ms': (time.perf_counter() - submitted) * 1000,
                'faces': int(camera.last_face_stats.get('refined', 0)),
                'detected_faces': int(camera.last_face_stats.get('detected', 0)),
                'capture_sequence': seq,
            }
            ready()
            events.put(('done', slot, seq, rgb, metrics))
        except Exception as exc:
            ready()
            events.put(('error', slot, seq, repr(exc)))


def run_pipeline(latest, running):
    """Yield ordered completed frames; latest() returns (camera_sequence, BGR[, capture_time])."""
    ctx = mp.get_context('spawn')
    events = ctx.Queue(maxsize=8 + 2 * WORKERS)
    tasks = [ctx.Queue(maxsize=1) for _ in range(WORKERS)]
    workers = [ctx.Process(target=render_worker, args=(i, tasks[i], events), daemon=True) for i in range(WORKERS)]
    try:
        for worker in workers:
            worker.start()
        initialized = set()
        deadline = time.monotonic() + 120
        while running() and len(initialized) < WORKERS:
            if time.monotonic() > deadline:
                raise RuntimeError('BPU workers did not initialize')
            if any(p.exitcode is not None for p in workers):
                raise RuntimeError('BPU worker exited during initialization')
            try:
                event = events.get(timeout=.1)
            except queue.Empty:
                continue
            if event[0] == 'init':
                initialized.add(event[1])
        busy = set()
        gated = set()          # slots admitted whose BPU chain has not finished yet
        previous = displayed = -1
        progress = time.monotonic()
        while running():
            if any(not p.is_alive() for p in workers):
                raise RuntimeError('BPU worker exited')
            if busy and time.monotonic() - progress > 30:
                raise RuntimeError('BPU worker stalled')
            free = [s for s in range(WORKERS) if s not in busy]
            if free and len(gated) < INFLIGHT:
                got = latest()
                seq, frame = got[0], got[1]
                stamp = got[2] if len(got) > 2 else time.perf_counter()
                if frame is not None and seq != previous:
                    previous = seq
                    slot = free[0]
                    tasks[slot].put((seq, prescale(frame) if PRESCALE else frame, stamp))
                    busy.add(slot)
                    gated.add(slot)
                    progress = time.monotonic()
            try:
                event = events.get(timeout=.005)
            except queue.Empty:
                continue
            progress = time.monotonic()
            kind, slot, seq = event[:3]
            if kind == 'stage':
                gated.discard(slot)
            elif kind in ('done', 'error'):
                busy.discard(slot)
                gated.discard(slot)
                if kind == 'error':
                    print('INFERENCE_WORKER', event[3], flush=True)
                elif seq > displayed:
                    displayed = seq
                    yield event[3], event[4]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=3)
        events.close()
        for task in tasks:
            task.close()
