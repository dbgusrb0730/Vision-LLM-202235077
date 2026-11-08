"""Smart Photo Director: YOLO + pose + local Gemma + offline speech."""
import argparse
from datetime import datetime
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from queue import Empty
import sys
import shutil
import subprocess
from threading import Thread
import time

from framing import Advice, Observation, assess, signature, CaptureGate
from settings import load_config


def check(config, vision_only=False, text_only=False):
    packages = ['cv2', 'numpy', 'mediapipe', 'ultralytics', 'PIL']
    paths = ['yolo_model', 'pose_model', 'font_path']
    if not vision_only:
        packages.append('llama_cpp')
        paths.append('gemma_model')
        if not text_only:
            if config.get('audio_backend') != 'alsa':
                packages.append('sounddevice')
            paths += ['whisper_executable', 'whisper_model', 'piper_python', 'piper_model']
    ok = True
    for name in packages:
        present = importlib.util.find_spec(name) is not None
        print(f'[{"OK" if present else "MISSING"}] package {name}')
        ok &= present
    for name in paths:
        present = Path(config[name]).is_file()
        print(f'[{"OK" if present else "MISSING"}] {name}: {config[name]}')
        ok &= present
    if not (vision_only or text_only):
        present = Path(config['piper_model'] + '.json').is_file()
        print(f'[{"OK" if present else "MISSING"}] Piper model JSON')
        ok &= present
        if config.get('audio_backend') == 'alsa':
            for command in ['arecord', 'aplay'] + (['pasuspender'] if config.get('use_pasuspender') else []):
                present = shutil.which(command) is not None
                print(f'[{"OK" if present else "MISSING"}] {command}')
                ok &= present
    print('추론 장치: CPU (YOLO / MediaPipe / Gemma / Whisper / Piper)')
    return ok


def terminal_input(worker):
    while not worker.stop_event.is_set():
        try:
            text = input()
        except (EOFError, OSError):
            return
        if text.strip() and not worker.submit('command', text.strip()):
            print('이전 명령을 처리 중입니다. 잠시 후 다시 입력하세요.')


class Overlay:
    def __init__(self, config):
        from PIL import ImageFont
        self.font = ImageFont.truetype(config['font_path'], 21)

    @lru_cache(maxsize=32)
    def panel(self, width, lines):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw
        image = Image.new('RGB', (width, 900), (15, 20, 30))
        draw = ImageDraw.Draw(image)
        y = 12
        for text in lines:
            # Wrap by rendered width, including Korean. Screen overlay is not saved.
            line = ''
            for char in text:
                if draw.textlength(line + char, font=self.font) > image.width - 30:
                    draw.rectangle((8, y - 2, image.width - 8, y + 27), fill=(15, 20, 30))
                    draw.text((14, y), line, font=self.font, fill='white')
                    line, y = '', y + 30
                line += char
            draw.rectangle((8, y - 2, image.width - 8, y + 27), fill=(15, 20, 30))
            draw.text((14, y), line, font=self.font, fill='white')
            y += 30
        return cv2.cvtColor(np.asarray(image.crop((0, 0, width, y))), cv2.COLOR_RGB2BGR)

    def draw(self, frame, lines):
        panel = self.panel(frame.shape[1], tuple(lines))
        height = min(frame.shape[0], panel.shape[0])
        frame[:height] = panel[:height]
        return frame


def save_photo(frame, config, mode, obs):
    import cv2
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = output / f'{stem}_{mode}.jpg'
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError('JPEG 인코딩에 실패했습니다.')
    encoded.tofile(str(path))  # Supports Windows non-ASCII paths.
    path.with_suffix('.json').write_text(json.dumps({
        'timestamp': datetime.now().astimezone().isoformat(), 'mode': mode,
        'people': obs.people, 'box_normalized': obs.box,
        'image_mirrored': False, 'note': 'Original camera frame, no overlay or automatic crop.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def run(config, args):
    import cv2
    from vision import LiveVision
    vision = worker = cap = None
    try:
        vision = LiveVision(config)
        backend = {'auto': cv2.CAP_ANY, 'gstreamer': cv2.CAP_GSTREAMER,
                   'dshow': cv2.CAP_DSHOW, 'v4l2': cv2.CAP_V4L2}[config['camera_backend']]
        cap = cv2.VideoCapture(config['camera'], backend)
        if not cap.isOpened():
            raise RuntimeError('카메라를 열 수 없습니다. camera 번호와 backend를 확인하세요.')
        if config['camera_backend'] != 'gstreamer':
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera_height'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        overlay = Overlay(config)
        if not args.vision_only:
            from assistant_worker import AssistantWorker
            worker = AssistantWorker(config, args.text_only)
            worker.start()
            Thread(target=terminal_input, args=(worker,), daemon=True).start()
        gate = CaptureGate(config['stable_seconds'], config['countdown_seconds'])
        mode, revision = 'full', 0
        caption = '1 전신 | 2 상반신 | Space 촬영 | C 취소 | Q 종료'
        caption_until, running, loaded = 0.0, True, args.vision_only
        last_code, status_since, previous_time = None, time.monotonic(), time.monotonic()
        fps, actions = 0.0, []
        obs, observation_at = Observation(0), 0.0
        observation_after = 0.0
        state, remaining = 'waiting', None
        analysis_fps, fps_label, label_at = 0.0, '', 0.0
        timing_log_at = 0.0
        print('카메라 창: 1 전신 / 2 상반신 / Space 촬영 / C 취소 / Q 종료')
        print('Gemma 로딩 후 터미널에 한국어 명령을 입력하고 Enter를 눌러도 됩니다.')
        while running:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('카메라 프레임을 읽지 못했습니다. 촬영을 중지합니다.')
            start = time.monotonic()
            result = vision.step(frame, start)
            now = time.monotonic()
            if result is not None:
                analyzed_frame, observation_at, obs, timings = result
                analysis_fps = 1 / max(now - observation_at, .001)
                if now - timing_log_at >= 5:
                    yolo_label = f'{timings[0]*1000:.0f} ms' if obs.detection_fresh else '이전 탐지 재사용'
                    print(f'영상 분석: YOLO {yolo_label} / '
                          f'Pose {timings[1]*1000:.0f} ms', flush=True)
                    timing_log_at = now
            if worker:
                while True:
                    try:
                        kind, value = worker.events.get_nowait()
                    except Empty:
                        break
                    if kind == 'fatal':
                        raise RuntimeError(value)
                    elif kind == 'loaded':
                        loaded = True
                    if kind == 'intent':
                        actions.append(value)
                    elif kind in ('caption', 'heard', 'error', 'loaded'):
                        caption = ('들은 말: ' if kind == 'heard' else '') + value
                        caption_until = now + 8
                        print(caption, flush=True)
            for action in actions:
                if loaded or action == 'cancel':
                    gate.command(action, now)
                    observation_after = now
                    state, remaining = 'waiting', None
                if action in ('full', 'upper'):
                    mode = action
                    revision += 1
                    if worker:
                        worker.mode_selected = True
                elif action == 'cancel':
                    revision += 1
                elif action == 'quit':
                    running = False
            actions.clear()
            if not running:
                break
            advice = assess(obs, mode)
            if advice.code != last_code:
                last_code, status_since = advice.code, now
            # Do not capture a frame that took too long to analyze.
            if now - observation_at > 1:
                advice = Advice('slow_frame', '영상 처리 지연이 커요. 입력 크기와 CPU 부하를 확인해 주세요.')
            if observation_at < observation_after:
                advice = Advice('waiting_frame', '새 구도를 확인하고 있어요. 잠시 멈춰 주세요.')
            gate_advice = advice
            if worker and worker.busy.is_set():
                gate_advice = Advice('assistant_busy', '음성 안내가 끝나면 촬영 준비를 시작해요.')
            # Never advance a countdown using the same observation repeatedly.
            # Save the exact analyzed frame, not an unverified newer preview.
            if result is not None or not gate_advice.ready:
                state, remaining = gate.update(now, gate_advice, signature(obs, mode),
                                               allow_capture=obs.detection_fresh)
            elif state in ('capture', 'interrupted', 'expired'):
                state, remaining = 'waiting', None
            if state == 'capture':
                path = save_photo(analyzed_frame, config, mode, obs)
                caption, caption_until = f'저장 완료: {path.name}', now + 8
                print('사진 저장:', path)
                if worker:
                    worker.submit('say', '사진을 저장했어요. 다시 찍으려면 사진이라고 말해 주세요.')
            elif state == 'interrupted':
                caption, caption_until = '움직임 또는 구도 변화로 카운트다운을 중단했어요. 다시 안정되면 시작합니다.', now + 5
            elif state == 'expired':
                caption, caption_until = '촬영 대기 시간이 지났어요. 하나, 둘 또는 사진으로 다시 시작하세요.', now + 5
            if worker:
                # Require a persistent observation before automatic spoken feedback.
                worker.publish(mode, advice, remaining if state == 'countdown' else None,
                               revision, now - status_since >= .5)
            preview = cv2.flip(frame, 1) if config['mirror_preview'] else frame.copy()
            h, w = preview.shape[:2]
            for fraction in (1 / 3, 2 / 3):
                cv2.line(preview, (int(w * fraction), 0), (int(w * fraction), h), (80, 80, 80), 1)
                cv2.line(preview, (0, int(h * fraction)), (w, int(h * fraction)), (80, 80, 80), 1)
            color = (0, 210, 70) if advice.ready else (0, 170, 255)
            if obs.box:
                x1, y1, x2, y2 = obs.box
                if config['mirror_preview']:
                    x1, x2 = 1 - x2, 1 - x1
                cv2.rectangle(preview, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
            for point in obs.landmarks:
                if point.visibility > .55 and 0 <= point.x <= 1 and 0 <= point.y <= 1:
                    x = 1 - point.x if config['mirror_preview'] else point.x
                    cv2.circle(preview, (int(x*w), int(point.y*h)), 3, color, -1)
            fps = .9 * fps + .1 / max(now - previous_time, .001)
            previous_time = now
            if now - label_at >= 1:
                fps_label = f'화면 {fps:.1f} / 분석 {analysis_fps:.1f} FPS'
                label_at = now
            lines = [f'포토 디렉터 | {"전신" if mode == "full" else "상반신"} | {fps_label}',
                     advice.text,
                     (f'이동 준비 시간 {remaining}초' if state == 'preparing' else
                      f'촬영까지 {remaining}' if remaining else
                      '자동 촬영 대기 · 구도를 맞춰 주세요' if gate.armed_until else '하나: 전신 / 둘: 상반신 / 사진: 다시 촬영'),
                     worker.state if worker else 'VISION ONLY · Gemma/음성 미사용 검사 모드',
                     caption if now < caption_until else '1 전신 | 2 상반신 | Space 촬영 | C 취소 | Q 종료']
            cv2.imshow('Smart Photo Director', overlay.draw(preview, lines))
            key = cv2.waitKey(1) & 0xff
            action = {ord('1'): 'full', ord('2'): 'upper', 32: 'shoot',
                      ord('c'): 'cancel', ord('q'): 'quit', 27: 'quit'}.get(key)
            if action:
                actions.append(action)
            # Jetson GTK may not report this property reliably; Q/Esc exits there.
            if sys.platform == 'win32' and cv2.getWindowProperty('Smart Photo Director', cv2.WND_PROP_VISIBLE) < 1:
                running = False
    finally:
        if worker:
            worker.close()
            worker.join(timeout=2)
        if cap:
            cap.release()
        if vision:
            vision.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.laptop.json')
    parser.add_argument('--vision-only', action='store_true', help='Explicit diagnostic mode: no Gemma or audio')
    parser.add_argument('--text-only', action='store_true', help='Local Gemma + terminal commands, no microphone/TTS')
    parser.add_argument('--check', action='store_true', help='Check local dependencies/files without opening camera')
    parser.add_argument('--audio-devices', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.audio_devices:
        if config.get('audio_backend') == 'alsa':
            subprocess.run(['arecord', '-l'], check=True)
            subprocess.run(['aplay', '-l'], check=True)
        else:
            import sounddevice
            print(sounddevice.query_devices())
        return 0
    ready = check(config, args.vision_only, args.text_only)
    if args.check:
        return 0 if ready else 1
    if not ready:
        print('누락된 항목을 준비하세요. README.md에 단계별 실행 방법이 있습니다.')
        return 1
    try:
        run(config, args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f'실행 중지: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
