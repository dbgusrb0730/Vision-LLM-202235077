"""Single background owner of Gemma and audio; the camera loop never awaits them."""
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time
from commands import direct_intent


class AssistantWorker(Thread):
    def __init__(self, config, text_only=False):
        super().__init__(daemon=True, name='local-gemma-audio')
        self.config, self.text_only = config, text_only
        self.stop_event, self.lock = Event(), Lock()
        self.busy = Event()
        self.busy.set()
        self.requests, self.events = Queue(maxsize=8), Queue()
        self.snapshot = None
        self.state = 'Gemma 로딩 중'
        self.mode_selected = False

    def publish(self, mode, advice, countdown, revision, persistent=True):
        with self.lock:
            self.snapshot = (mode, advice, countdown, revision, time.monotonic(), persistent)

    def current(self):
        with self.lock:
            return self.snapshot

    def submit(self, kind, text):
        try:
            self.requests.put_nowait((kind, text))
            return True
        except Full:
            return False

    def speak(self, text, audio):
        already_busy = self.busy.is_set()
        self.busy.set()
        try:
            self.state = '안내 중'
            self.events.put(('caption', text))
            if audio:
                audio.say(text)
        finally:
            if not already_busy:
                self.busy.clear()

    def process(self, utterance, director, audio, source='text'):
        self.state = '명령 해석 중'
        self.events.put(('heard', utterance))
        intent = direct_intent(utterance)
        if intent is None:
            if source == 'voice':
                # Misheard audio must not be converted into a guessed control action.
                intent = 'unknown'
            else:
                self.state = 'Gemma가 명령 해석 중'
                intent = director.intent(utterance)
        if intent in ('full', 'upper'):
            self.mode_selected = True
        if intent == 'prepare':
            self.speak('8초 동안 촬영 위치로 이동하세요. 그 뒤 구도가 안정되면 자동으로 한 장 찍어요.', audio)
            self.events.put(('intent', 'prepare'))
            return
        # UI applies state changes immediately, before lengthy speech synthesis.
        self.events.put(('intent', intent))
        if intent == 'status':
            snapshot = self.current()
            if snapshot is None:
                return
            mode, advice, _, revision, _, _ = snapshot
            answer = director.explain(mode, advice)
            current = self.current()
            # Do not speak an old left/right instruction after the person moves.
            if (current and current[3] == revision and current[1].code == advice.code
                    and time.monotonic() - current[4] < 1):
                self.speak(answer, audio)
        else:
            reply = {
                'full': '전신 모드입니다. 구도가 맞으면 자동으로 찍어요.',
                'upper': '상반신 모드입니다. 구도가 맞으면 자동으로 찍어요.',
                'shoot': f'구도가 맞으면 {self.config.get("countdown_seconds", 4)}초 카운트다운 후 찍어요.',
                'cancel': '촬영 예약을 취소했어요.',
                'unknown': '잘 못 들었어요. 하나, 둘, 사진, 준비 중 하나로 다시 말해 주세요.',
            }.get(intent)
            if reply:
                self.speak(reply, audio)

    def run(self):
        try:
            from gemma import GemmaDirector
            director = GemmaDirector(self.config)
            audio = None
            if not self.text_only:
                from audio_io import AudioIO
                self.state = '음성 모델과 카운트다운 준비 중'
                audio = AudioIO(self.config)
                self.speak('포토 디렉터입니다. 전신은 하나, 상반신은 둘이라고 말해 주세요. 구도가 맞으면 자동으로 찍어요.', audio)
            self.events.put(('loaded', '로컬 Gemma 준비 완료'))
            self.busy.clear()
        except Exception as exc:
            self.state = '시작 실패'
            self.events.put(('fatal', f'{type(exc).__name__}: {exc}'))
            return
        last_guidance, last_code, last_count = 0.0, None, None
        while not self.stop_event.is_set():
            try:
                snapshot = self.current()
                count = snapshot[2] if snapshot and time.monotonic() - snapshot[4] < 1 else None
                if count:
                    self.state = '촬영 카운트다운 · 음성 대기 일시 중지'
                    if count != last_count and audio:
                        audio.count(count)
                    last_count = count
                    self.stop_event.wait(.03)
                    continue
                last_count = None
                try:
                    kind, text = self.requests.get_nowait()
                except Empty:
                    kind = text = None
                if kind == 'command':
                    self.busy.set()
                    try:
                        self.process(text, director, audio)
                    finally:
                        self.busy.clear()
                    continue
                if kind == 'say':
                    self.speak(text, audio)
                    continue
                snapshot = self.current()
                now = time.monotonic()
                if snapshot and now - snapshot[4] < 1:
                    _, advice, _, _, _, persistent = snapshot
                    if (self.mode_selected and persistent and not advice.ready
                            and now - last_guidance >= self.config['guidance_interval']
                            and advice.code != last_code):
                        # Automatic corrections are authoritative rule text. Gemma is
                        # used for free-form intent and explicit composition questions.
                        self.speak(advice.text, audio)
                        last_guidance, last_code = time.monotonic(), advice.code
                        continue
                self.state = '입력 대기' if audio is None else '듣는 중 · 지금 말씀하세요'
                if audio:
                    try:
                        utterance = audio.listen(self.stop_event, lambda: not self.requests.empty()
                                                 or bool(self.current() and self.current()[2]),
                                                 self.busy.set)
                    finally:
                        self.busy.clear()
                    if utterance:
                        self.busy.set()
                        try:
                            self.process(utterance, director, audio, source='voice')
                        finally:
                            self.busy.clear()
                else:
                    self.stop_event.wait(.1)
            except Exception as exc:
                # Fail visibly instead of silently switching to an external service.
                self.state = '음성/모델 오류'
                self.events.put(('error', f'{type(exc).__name__}: {exc}'))
                self.stop_event.wait(1)

    def close(self):
        self.stop_event.set()
