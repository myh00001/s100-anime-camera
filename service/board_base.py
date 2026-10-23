from pathlib import Path
import cv2
import numpy as np
from face_tracking import FaceContinuity
ROOT=Path(__file__).resolve().parent
class AnimeCamera:

    def __init__(self):
        self.scene_size = 640
        self.previous_source = self.previous_output = None
        self.use_tracking = self.use_temporal = False
        self.face_continuity = FaceContinuity()
        self.last_face_stats = {}
        cv2.setNumThreads(4)
        import os as _os
        self.detector = None
        if _os.environ.get('RAC_FACE', '0') == '1':
            self.detector = cv2.FaceDetectorYN.create(str(ROOT / 'models/face_detection_yunet_2023mar.onnx'), '', (480, 480), 0.65, 0.3, 2000)

    def face_mask(self, crop, refined):
        yy, xx = np.mgrid[0:512, 0:512]
        radius = ((xx - 256) / 195) ** 2 + ((yy - 260) / 235) ** 2
        return np.clip((1 - radius) * 6, 0, 1).astype(np.float32)

    def render(self, rgb):
        h, w = rgb.shape[:2]
        self._scene_request = True
        try:
            out = self.generate(rgb, self.scene_size)
        finally:
            self._scene_request = False
        oh, ow = out.shape[:2]
        source = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        import os as _os
        if _os.environ.get('RAC_FACE', '0') != '1':
            gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            self.last_face_stats = {'detected': 0, 'tracked': 0, 'refined': 0, 'input_sharpness': round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 2), 'face_mode': 'off'}
            face_coverage = np.zeros((oh, ow), np.float32)
            self._remaining_faces = 0
            if getattr(self, 'on_bpu_chain_done', None):
                self.on_bpu_chain_done()
            self.current_face_coverage = face_coverage
            detail = out.astype(np.float32) - cv2.GaussianBlur(out.astype(np.float32), (0, 0), 0.75)
            out = np.clip(out.astype(np.float32) * 0.98 + 4 + detail * 0.22, 0, 255).astype(np.uint8)
            if self.use_temporal and self.previous_source is not None and (self.previous_source.shape == source.shape):
                change = np.max(np.abs(source.astype(np.float32) - self.previous_source.astype(np.float32)), axis=2)
                change = cv2.dilate(change, np.ones((7, 7), np.uint8))
                disagreement = np.max(np.abs(out.astype(np.float32) - self.previous_output.astype(np.float32)), axis=2)
                weight = np.clip(1 - change / 9, 0, 1) * np.clip(1 - disagreement / 22, 0, 1) * 0.25
                out = np.uint8(out.astype(np.float32) * (1 - weight[..., None]) + self.previous_output * weight[..., None])
            self.previous_source = source.copy()
            self.previous_output = out.copy()
            return out
        detect_scale = 480 / max(ow, oh)
        dw, dh = (round(ow * detect_scale), round(oh * detect_scale))
        px, py = ((480 - dw) // 2, (480 - dh) // 2)
        detect = np.zeros((480, 480, 3), dtype=np.uint8)
        detect[py:py + dh, px:px + dw] = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        _, faces = self.detector.detect(cv2.cvtColor(detect, cv2.COLOR_RGB2BGR))
        landmarks = []
        head_boxes = []
        if faces is not None:
            ordered = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:3]
            landmarks = [(face[4:14].reshape(5, 2) - np.array([px, py])) / detect_scale for face in ordered]
            head_boxes = [np.r_[(face[:2] - np.array([px, py])) / detect_scale, face[2:4] / detect_scale] for face in ordered]
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        if self.use_tracking:
            points_list, tracked = self.face_continuity.update(gray, landmarks)
        else:
            points_list, tracked = (landmarks, 0)
        self.last_face_stats = {'detected': len(landmarks), 'tracked': tracked, 'refined': 0, 'input_sharpness': round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 2)}
        face_coverage = np.zeros((oh, ow), np.float32)
        self._remaining_faces = len(points_list)
        if not points_list and getattr(self, 'on_bpu_chain_done', None):
            self.on_bpu_chain_done()
        if points_list:
            for face_index, points in enumerate(points_list):
                self._remaining_faces -= 1
                eyes = points[:2][np.argsort(points[:2, 0])]
                mouth = points[3:5].mean(axis=0)
                if np.linalg.norm(eyes[1] - eyes[0]) < 8 * max(ow, oh) / 960:
                    continue
                original = np.float32([eyes[0], eyes[1], mouth])
                canonical = np.float32([[176, 205], [336, 205], [256, 328]])
                if hasattr(self, 'alignment_template'):
                    transform, _ = cv2.estimateAffinePartial2D(np.float32(points), self.alignment_template, method=cv2.LMEDS)
                    if transform is None:
                        continue
                else:
                    transform = cv2.getAffineTransform(original, canonical)
                if abs(np.linalg.det(transform[:, :2])) < 0.01:
                    continue
                if face_index < len(head_boxes):
                    bx, by, bw, bh = head_boxes[face_index]
                    left, top = (max(0, bx - bw * 0.22), max(0, by - bh * 0.45))
                    right, bottom = (min(ow, bx + bw * 1.22), min(oh, by + bh * 1.14))
                    corners = np.float32([[left, top], [right, top], [left, bottom], [right, bottom]])
                    projected = cv2.transform(corners[None], transform)[0]
                    extent = projected.max(axis=0) - projected.min(axis=0)
                    scale = min(1.0, 440 / max(float(extent.max()), 1))
                    center = (projected.max(axis=0) + projected.min(axis=0)) / 2
                    transform[:, :2] *= scale
                    transform[:, 2] = transform[:, 2] * scale + np.array([256, 256]) - center * scale
                original_to_scene = np.float32([[ow / w, 0, 0], [0, oh / h, 0], [0, 0, 1]])
                original_to_face = transform @ original_to_scene
                crop = cv2.warpAffine(rgb, original_to_face, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
                refined = self.generate(crop, 512)
                inverse = cv2.invertAffineTransform(transform)
                bounds = cv2.transform(np.float32([[[0,0],[512,0],[0,512],[512,512]]]), inverse)[0]
                x0,y0 = np.maximum(0,np.floor(bounds.min(axis=0)-2)).astype(int)
                x1,y1 = np.minimum([ow,oh],np.ceil(bounds.max(axis=0)+2)).astype(int)
                if x1<=x0 or y1<=y0:continue
                local_inverse=inverse.copy();local_inverse[:,2]-=[x0,y0]
                restored=cv2.warpAffine(refined,local_inverse,(x1-x0,y1-y0),flags=cv2.INTER_CUBIC)
                mask=self.face_mask(crop,refined)
                mask=cv2.warpAffine(mask,local_inverse,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR)[...,None]
                face_coverage[y0:y1,x0:x1]=np.maximum(face_coverage[y0:y1,x0:x1],mask[:,:,0])
                out[y0:y1,x0:x1]=np.uint8(out[y0:y1,x0:x1].astype(np.float32)*(1-mask)+restored*mask)
                self.last_face_stats['refined'] += 1
        detail = out.astype(np.float32) - cv2.GaussianBlur(out.astype(np.float32), (0, 0), 0.75)
        out = np.clip(out.astype(np.float32) * 0.98 + 4 + detail * 0.22, 0, 255).astype(np.uint8)
        if self.use_temporal and self.previous_source is not None and (self.previous_source.shape == source.shape):
            change = np.max(np.abs(source.astype(np.float32) - self.previous_source.astype(np.float32)), axis=2)
            change = cv2.dilate(change, np.ones((7, 7), np.uint8))
            disagreement = np.max(np.abs(out.astype(np.float32) - self.previous_output.astype(np.float32)), axis=2)
            weight = np.clip(1 - change / 9, 0, 1) * np.clip(1 - disagreement / 22, 0, 1) * 0.25
            weight *= 1 - face_coverage
            out = np.uint8(out.astype(np.float32) * (1 - weight[..., None]) + self.previous_output * weight[..., None])
        self.current_face_coverage = face_coverage
        self.previous_source = source.copy()
        self.previous_output = out.copy()
        return out
