"""USB MJPEG capture with the board hardware JPEG decoder (hobot_vio Decoder, type 3 = MJPEG).

The camera's MJPEG bitstream is read without CPU decode (CAP_PROP_CONVERT_RGB=0), decoded to NV12
by the codec block, scaled to <=1280 on the NV12 planes and converted to BGR once at the small size.
Falls back to OpenCV's CPU decode when hobot_vio is unavailable or RAC_HWDEC=0.
"""
import os
import cv2
import numpy as np


def nv12_to_bgr(buf, w, h, stride_h, max_side=1280):
    """JPEG decoder NV12 is full-range BT.601 (JFIF), so convert with YCrCb (full range), not YUV2BGR_NV12."""
    a = np.frombuffer(buf, np.uint8)
    y = a[:w * stride_h].reshape(stride_h, w)[:h]
    uv = a[w * stride_h:w * stride_h + w * stride_h // 2].reshape(stride_h // 2, w // 2, 2)[:h // 2]
    s = min(1, max_side / max(w, h))
    nw, nh = (round(w * s), round(h * s)) if s < 1 else (w, h)
    if (nw, nh) != (w, h):
        y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_LINEAR)
    uv = cv2.resize(uv, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.merge([y, uv[:, :, 1], uv[:, :, 0]])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


class HwMjpegCapture:
    def __init__(self, device, width, height, fps=30, chn=0):
        self.cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height); self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.w, self.h = int(self.cap.get(3)), int(self.cap.get(4))
        self.dec = None
        if os.environ.get('RAC_HWDEC', '1') == '1':
            try:
                from hobot_vio import libsrcampy
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                self.dec = libsrcampy.Decoder()
                if self.dec.decode('', chn, 3, self.w, self.h)[0] != 0:
                    raise RuntimeError('decoder init failed')
            except Exception as exc:  # CPU fallback
                print('HWDEC_FALLBACK', repr(exc), flush=True)
                self.dec = None
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.stride_h = None

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def read(self):
        ok, f = self.cap.read()
        if not ok or self.dec is None:
            return ok, f
        self.dec.set_img(f.tobytes(), 0, 0)
        img = self.dec.get_img()
        if img is None:
            return False, None
        if self.stride_h is None:
            self.stride_h = len(img) * 2 // 3 // self.w
        return True, nv12_to_bgr(img, self.w, self.h, self.stride_h)

    def release(self):
        if self.dec is not None:
            self.dec.close()
        self.cap.release()
