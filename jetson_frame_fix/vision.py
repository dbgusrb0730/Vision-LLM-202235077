"""YOLO person detection and MediaPipe pose, on the same unmirrored frame."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from framing import Observation, Landmark


class Vision:
    def __init__(self, config):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(config.get('cpu_threads', 2))
        self.cv2, self.mp, self.config = cv2, mp, config
        for key in ('yolo_model', 'pose_model'):
            if not Path(config[key]).is_file():
                raise FileNotFoundError(f'{key}: {config[key]} — python prepare_models.py --vision')
        self.yolo = YOLO(config['yolo_model'], task='detect')
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=config['pose_model'],
                                           delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
            min_tracking_confidence=.5))
        self.timestamp = 0
        self.timings = (0.0, 0.0)

    def observe(self, frame, now):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        width = min(w, self.config['analysis_width'])
        small = cv2.resize(frame, (width, round(h * width / w)))
        started = time.monotonic()
        results = self.yolo.predict(small, classes=[0], conf=.5,
                                    imgsz=self.config.get('yolo_imgsz', 320),
                                    device='cpu', verbose=False)
        boxes = results[0].boxes
        yolo_done = time.monotonic()
        self.timings = (yolo_done - started, 0.0)
        if len(boxes) != 1:
            return Observation(len(boxes))
        sh, sw = small.shape[:2]
        raw = boxes.xyxy[0].cpu().tolist()
        box = tuple(v / (sw if i % 2 == 0 else sh) for i, v in enumerate(raw))
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self.pose.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        points = tuple(Landmark(p.x, p.y, p.visibility, p.presence)
                       for p in result.pose_landmarks[0]) if result.pose_landmarks else ()
        self.timings = (yolo_done - started, time.monotonic() - yolo_done)
        return Observation(1, box, points)

    def close(self):
        self.pose.close()


class LiveVision:
    """One analysis at a time; busy frames are skipped, never queued."""
    def __init__(self, config):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.config, self.model, self.pending = config, None, None

    def analyze(self, frame, captured_at):
        if self.model is None:
            self.model = Vision(self.config)
        observation = self.model.observe(frame, captured_at)
        return frame, captured_at, observation, self.model.timings

    def step(self, frame, captured_at):
        result = None
        if self.pending is not None and self.pending.done():
            result = self.pending.result()
            self.pending = None
        if self.pending is None:
            self.pending = self.pool.submit(self.analyze, frame.copy(), captured_at)
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.model is not None:
            self.model.close()
