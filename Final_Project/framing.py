"""Pure framing rules and capture state machine. Coordinates are normalized, unmirrored.

Thresholds are demo heuristics, not an aesthetic score or a depth measurement.
Keep this module independent of camera, GPU and language-model dependencies.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class Observation:
    people: int
    box: tuple | None = None
    landmarks: tuple = ()
    detection_fresh: bool = True


@dataclass(frozen=True)
class Advice:
    code: str
    text: str
    ready: bool = False


def required_landmarks(mode):
    # Full: face, shoulders and hips. Close-up: face and shoulders only.
    return [0, 11, 12, 23, 24] if mode == 'full' else [0, 11, 12]


def assess(obs, mode):
    if mode not in ('full', 'upper'):
        raise ValueError('mode must be full or upper')
    if obs.people == 0:
        return Advice('no_person', '사람이 보이지 않아요. 카메라 앞에 서 주세요.')
    if obs.people != 1:
        return Advice('many_people', '한 사람씩 촬영해요. 다른 분은 화면 밖으로 이동해 주세요.')
    if obs.box is None or len(obs.landmarks) != 33:
        return Advice('unknown_pose', '자세를 확인하기 어려워요. 카메라를 바라보고 몸이 보이게 서 주세요.')
    x1, y1, x2, y2 = obs.box
    if not all(math.isfinite(v) for v in obs.box) or x2 <= x1 or y2 <= y1:
        return Advice('unknown_pose', '인물 위치를 확인하지 못했어요.')
    for i in required_landmarks(mode):
        p = obs.landmarks[i]
        if (not all(math.isfinite(v) for v in (p.x, p.y, p.visibility, p.presence))
                or p.visibility < .55 or p.presence < .55
                or not .01 < p.x < .99 or not .01 < p.y < .99):
            message = ('얼굴과 어깨, 허리가 보이게 서 주세요. 발은 화면 밖이어도 괜찮아요.'
                       if mode == 'full' else '얼굴과 양쪽 어깨가 보이게 카메라를 바라봐 주세요.')
            return Advice('uncertain_body', message)
    if y1 < (.035 if mode == 'full' else .015):
        return Advice('head_margin', '머리 위 여유가 부족해요. 조금 뒤로 이동해 주세요.')
    if mode == 'full' and (x1 < .025 or x2 > .975):
        return Advice('side_margin', '몸이 화면 가장자리에 가까워요. 조금 뒤로 이동해 주세요.')
    center = (obs.landmarks[11].x + obs.landmarks[12].x) / 2
    # Camera-facing subject: raw image left corresponds to subject's right.
    tolerance = .07 if mode == 'full' else .15
    if center < .5 - tolerance:
        return Advice('move_subject_left', '카메라를 바라본 상태에서 본인의 왼쪽으로 조금 이동해 주세요.')
    if center > .5 + tolerance:
        return Advice('move_subject_right', '카메라를 바라본 상태에서 본인의 오른쪽으로 조금 이동해 주세요.')
    if mode == 'full':
        if y2 - y1 < .65:
            return Advice('too_small', '전신이 조금 작게 보여요. 카메라 쪽으로 조금 이동해 주세요.')
    else:
        shoulder_width = abs(obs.landmarks[11].x - obs.landmarks[12].x)
        if shoulder_width < .22:
            return Advice('too_small', '얼굴과 어깨가 더 크게 나오도록 카메라 쪽으로 조금 와 주세요.')
    if y1 > (.22 if mode == 'full' else .40):
        return Advice('camera_tilt', '머리 위 빈 공간이 많아요. 삼각대의 높이나 기울기를 조정해 주세요.')
    return Advice('ready', '구도가 맞아요. 촬영을 위해 잠시 멈춰 주세요.', True)


def signature(obs, mode):
    if obs.box is None or len(obs.landmarks) != 33:
        return None
    # Cropped legs/hips and the lower box edge must not reset close-up stability.
    box = tuple(obs.box) if mode == 'full' else (obs.box[1],)
    return box + tuple(v for i in required_landmarks(mode)
                       for v in (obs.landmarks[i].x, obs.landmarks[i].y))


class CaptureGate:
    """An explicit request arms one shot. Motion cancels countdown, not the request.

    Compare against a fixed anchor (not only the last frame) to catch slow drift.
    A long observation gap also invalidates stability. All times are monotonic.
    """
    def __init__(self, stable_seconds=.5, countdown_seconds=4):
        self.stable_seconds = stable_seconds
        self.countdown_seconds = countdown_seconds
        self.cancel()

    def cancel(self):
        self.armed_until = None
        self.stable_since = None
        self.countdown_at = None
        self.anchor = None
        self.last_time = None
        self.ready_after = None

    def request(self, now, delay=0):
        self.cancel()
        self.armed_until = now + delay + 45
        self.ready_after = now + delay

    def command(self, action, now):
        """Selecting either mode starts one automatic shot, not a repeating loop."""
        if action in ('full', 'upper', 'shoot'):
            self.request(now)
        elif action == 'prepare':
            self.request(now, delay=8)
        elif action == 'cancel':
            self.cancel()

    def update(self, now, advice, pose_signature, allow_capture=True):
        if self.armed_until is not None and now >= self.armed_until:
            self.cancel()
            return 'expired', None
        if self.ready_after is not None and now < self.ready_after:
            return 'preparing', math.ceil(self.ready_after - now)
        gap = self.last_time is not None and now - self.last_time > 1.0
        self.last_time = now
        moved = (self.anchor is not None and pose_signature is not None
                 and max(abs(a - b) for a, b in zip(self.anchor, pose_signature)) > .025)
        if not advice.ready or pose_signature is None or gap or moved:
            was_counting = self.countdown_at is not None
            self.stable_since = None
            self.countdown_at = None
            self.anchor = None
            return ('interrupted' if was_counting else 'waiting'), None
        if self.stable_since is None:
            self.stable_since, self.anchor = now, pose_signature
        if self.armed_until is None:
            return 'idle', None
        if now - self.stable_since < self.stable_seconds:
            return 'stabilizing', None
        if self.countdown_at is None:
            self.countdown_at = now
        remaining = self.countdown_seconds - (now - self.countdown_at)
        if remaining <= 0:
            if not allow_capture:
                return 'countdown', 1  # Wait for a frame with a fresh person count/box.
            self.cancel()
            return 'capture', 0
        return 'countdown', math.ceil(remaining)
