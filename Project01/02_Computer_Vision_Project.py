"""Section Project: 초록색 공 추적 및 손과의 관계 시각화.

준비: python -m pip install opencv-python numpy mediapipe
노트북에서 안내한 hand_landmarker.task를 다음 경로에 저장:
    hand_landmarker.task (이 Python 파일과 같은 폴더)
실행: python 02_Computer_Vision_Project.py
영상: python 02_Computer_Vision_Project.py --source video.mp4
조작: q/ESC 종료, r 추적 초기화. --mirror로 좌우 반전.

초록색 공을 준비하고 손과 함께 카메라에 비춘다. 파란 원은 검출,
노란 원은 칼만 추정 위치, 자홍색 선은 손바닥까지의 화면상 거리다.
NEAR는 영상 평면에서 가깝다는 뜻이며 실제 접촉/3D 거리를 뜻하지 않는다.
"""

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "hand_landmarker.task"


def detect_ball(frame, clahe, min_area=300):
    """필터링 → LAB 명도 개선 → HSV 분리 → 형태학 → 윤곽선."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR
    )
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 70, 40]), np.array([85, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < min_area or perimeter == 0:
            continue
        circularity = 4 * np.pi * area / perimeter**2
        if circularity >= 0.55:
            candidates.append(contour)
    if not candidates:
        return None, mask
    contour = max(candidates, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    moments = cv2.moments(contour)
    center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
    return (center, radius, contour), mask


class BallTracker:
    """[x, y, vx, vy] 상태의 등속도 칼만 필터. 장기 미검출 시 초기화."""

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 10
        self.active = False
        self.missing_time = 0.0
        self.radius = 0.0
        self.trail = deque(maxlen=60)

    def update(self, detection, dt):
        if detection is None and not self.active:
            return None
        if detection is not None and not self.active:
            center, self.radius, _ = detection
            state = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
            self.kalman.statePost = state.copy()
            self.kalman.statePre = state.copy()
            self.active = True
        self.kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]], np.float32)
        self.kalman.processNoiseCov = g @ g.T * 100
        state = self.kalman.predict()
        if detection is not None:
            center, self.radius, _ = detection
            state = self.kalman.correct(center.astype(np.float32).reshape(2, 1))
            self.missing_time = 0.0
        else:
            self.missing_time += dt
            if self.missing_time > 1.0:
                self.active = False
                self.trail.clear()
                return None
        center = tuple(int(round(float(v))) for v in state[:2, 0])
        self.trail.append(center)
        return center


def draw_hands(frame, result, connections):
    """손 관절과 연결선을 표시하고 손바닥 중심/크기를 반환."""
    height, width = frame.shape[:2]
    palms = []
    for hand in result.hand_landmarks:
        points = np.array([(p.x * width, p.y * height) for p in hand])
        pixels = np.rint(points).astype(int)
        for connection in connections:
            cv2.line(frame, tuple(pixels[connection.start]),
                     tuple(pixels[connection.end]), (0, 200, 0), 2)
        for i, point in enumerate(pixels):
            color = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (255, 100, 0)
            cv2.circle(frame, tuple(point), 4, color, -1)
        palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
        palm_size = max(float(np.linalg.norm(points[0] - points[9])), 1.0)
        palms.append((palm, palm_size))
    return palms


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="카메라 번호 또는 영상 경로")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--jetson", action="store_true", help="CSI 카메라 사용")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--min-area", type=float, default=300)
    args = parser.parse_args()
    if args.min_area <= 0:
        parser.error("--min-area는 양수여야 합니다.")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        parser.error("MediaPipe가 필요합니다: python -m pip install mediapipe")
    if not args.model.is_file():
        parser.error(f"손 인식 모델이 없습니다: {args.model}\n"
                     "노트북 M절의 hand_landmarker.task 다운로드 안내를 확인하세요.")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    source = int(args.source) if args.source.isdecimal() else args.source
    if args.jetson:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! queue leaky=downstream max-size-buffers=1 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError("카메라/영상을 열 수 없습니다. --source 또는 --jetson을 확인하세요.")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tracker = BallTracker()
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_dt = 1 / fps if np.isfinite(fps) and fps > 0 else 1 / 30
        is_file = isinstance(source, str) and not args.jetson
        previous = time.monotonic()
        timestamp = 0
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("영상이 종료되었거나 프레임을 읽을 수 없습니다.")
                    break
                now = time.monotonic()
                dt = video_dt if is_file else max(now - previous, 0.001)
                previous = now
                timestamp += max(1, round(dt * 1000))
                scale = min(960 / frame.shape[1], 1.0)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                detection, mask = detect_ball(frame, clahe, args.min_area)
                center = tracker.update(detection, dt)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                )
                palms = draw_hands(frame, result, vision.HandLandmarksConnections.HAND_CONNECTIONS)
                status = "NO BALL"
                if detection is not None:
                    measured, radius, contour = detection
                    cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                    cv2.circle(frame, tuple(np.rint(measured).astype(int)), round(radius), (255, 0, 0), 2)
                if center is not None:
                    cv2.circle(frame, center, round(tracker.radius), (0, 255, 255), 2)
                    if len(tracker.trail) > 1:
                        cv2.polylines(frame, [np.array(tracker.trail, np.int32)], False, (0, 255, 255), 2)
                    status = "TRACKING" if detection is not None else "PREDICTED (occluded)"
                    if palms and detection is not None:
                        palm, size = min(palms, key=lambda p: np.linalg.norm(p[0] - detection[0]))
                        distance = float(np.linalg.norm(palm - detection[0]))
                        relation = "NEAR" if distance <= tracker.radius + size * 0.7 else "FAR"
                        status += f" | Hand: {relation} ({distance:.0f}px)"
                        cv2.line(frame, tuple(np.rint(detection[0]).astype(int)),
                                 tuple(np.rint(palm).astype(int)), (255, 0, 255), 2)
                    elif not palms:
                        status += " | NO HAND"
                cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Blue: detected  Yellow: tracked | q: quit  r: reset",
                            (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Section Project - Ball and Hand", frame)
                cv2.imshow("Color segmentation mask", mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker = BallTracker()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
