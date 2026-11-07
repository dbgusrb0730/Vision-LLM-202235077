"""YOLO person detection and MediaPipe pose, on the same unmirrored frame."""
from pathlib import Path
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

    def observe(self, frame, now):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        width = min(w, self.config['analysis_width'])
        small = cv2.resize(frame, (width, round(h * width / w)))
        results = self.yolo.predict(small, classes=[0], conf=.5,
                                    imgsz=self.config.get('yolo_imgsz', 320),
                                    device='cpu', verbose=False)
        boxes = results[0].boxes
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
        return Observation(1, box, points)

    def close(self):
        self.pose.close()
