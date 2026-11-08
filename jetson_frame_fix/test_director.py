"""Hardware-independent tests for framing/capture safety and command parsing."""
import unittest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
from framing import Landmark, Observation, assess, signature, CaptureGate
from gemma import parse_intent
from commands import direct_intent
from assistant_worker import AssistantWorker
from audio_io import AudioIO
from settings import load_config
from threading import Event
from vision import LiveVision


class LiveVisionTests(unittest.TestCase):
    def test_busy_frames_are_skipped_and_result_keeps_its_original_frame(self):
        import numpy as np
        started, release = Event(), Event()
        model = Mock()
        model.timings = (.1, .2)
        def observe(frame, timestamp):
            started.set()
            if not release.wait(2):
                raise TimeoutError('test worker did not unblock')
            return Observation(1)
        model.observe.side_effect = observe
        with patch('vision.Vision', return_value=model):
            worker = LiveVision({})
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            try:
                self.assertIsNone(worker.step(frame, 10))
                self.assertTrue(started.wait(2))
                frame[:] = 1
                self.assertIsNone(worker.step(frame, 11))
                self.assertEqual(model.observe.call_count, 1)
                release.set()
                worker.pending.result(timeout=2)
                result = worker.step(frame, 12)
                self.assertEqual(result[1], 10)
                self.assertEqual(int(result[0].sum()), 0)
                worker.pending.result(timeout=2)
                self.assertEqual([call.args[1] for call in model.observe.call_args_list], [10, 12])
            finally:
                release.set()
                worker.close()
            model.close.assert_called_once()

    def test_inference_failure_reaches_main_loop(self):
        import numpy as np
        with patch('vision.Vision', side_effect=RuntimeError('model failed')):
            worker = LiveVision({})
            try:
                frame = np.zeros((2, 2, 3), dtype=np.uint8)
                worker.step(frame, 1)
                with self.assertRaisesRegex(RuntimeError, 'model failed'):
                    worker.pending.result(timeout=2)
                with self.assertRaisesRegex(RuntimeError, 'model failed'):
                    worker.step(frame, 2)
            finally:
                worker.close()


def person(center=.5, upper=False):
    points = [Landmark(center, .4) for _ in range(33)]
    for i in (0, 1, 2, 3, 4, 5, 6):
        points[i] = Landmark(center, .45 if upper else .15)
    for i in (11, 12):
        offset = .16 if upper else .08
        points[i] = Landmark(center + (-offset if i == 11 else offset), .75 if upper else .3)
    for i in (23, 24):
        points[i] = Landmark(center, 1.3 if upper else .55, .1 if upper else 1.0)
    for i in (27, 28, 29, 30, 31, 32):
        points[i] = Landmark(center, 1.4 if upper else .90)
    return Observation(1, (center - .15, .08, center + .15, .99 if upper else .92), tuple(points))


class FramingTests(unittest.TestCase):
    def test_full_and_upper_have_different_visibility_requirements(self):
        self.assertTrue(assess(person(), 'full').ready)
        self.assertTrue(assess(person(upper=True), 'upper').ready)
        self.assertFalse(assess(person(upper=True), 'full').ready)

    def test_missing_or_multiple_people_never_ready(self):
        for obs in (Observation(0), Observation(2), Observation(1, (.1,.1,.8,.9))):
            self.assertFalse(assess(obs, 'full').ready)

    def test_low_visibility_is_unknown_even_with_good_box(self):
        obs = person()
        points = list(obs.landmarks)
        points[11] = Landmark(.5, .3, .1)
        self.assertEqual(assess(Observation(1, obs.box, tuple(points)), 'full').code, 'uncertain_body')

    def test_full_allows_missing_feet_and_box_touching_bottom(self):
        obs = person()
        points = list(obs.landmarks)
        for i in range(25, 33):
            points[i] = Landmark(.5, 1.5, .01, .01)
        cropped = Observation(1, (.3, .08, .7, 1.0), tuple(points))
        self.assertTrue(assess(cropped, 'full').ready)

    def test_closeup_like_reference_allows_hidden_hips_and_cropped_body(self):
        obs = person(upper=True)
        closeup = Observation(1, (0, .24, 1, 1), obs.landmarks)
        self.assertTrue(assess(closeup, 'upper').ready)
        self.assertFalse(assess(closeup, 'full').ready)

    def test_closeup_still_requires_visible_shoulders_and_sufficient_size(self):
        obs = person(upper=True)
        points = list(obs.landmarks)
        points[11] = Landmark(.49, .75)
        points[12] = Landmark(.51, .75)
        self.assertEqual(assess(Observation(1, obs.box, tuple(points)), 'upper').code, 'too_small')
        points[11] = Landmark(.3, .75, .1)
        self.assertFalse(assess(Observation(1, obs.box, tuple(points)), 'upper').ready)

    def test_hidden_joint_jitter_does_not_block_photo(self):
        for mode in ['full', 'upper']:
            obs = person(upper=mode == 'upper')
            gate = CaptureGate()
            gate.command(mode, 0)
            states = []
            for frame in range(10):
                points = list(obs.landmarks)
                for i in range(23 if mode == 'upper' else 25, 33):
                    points[i] = Landmark(frame * .3, 1.5 + frame, .01, .01)
                box = obs.box if mode == 'full' else (.1, .08, .9, .8 + frame * .02)
                current = Observation(1, box, tuple(points))
                states.append(gate.update(frame * .5, assess(current, mode), signature(current, mode))[0])
            self.assertEqual(states[-1], 'capture')

    def test_nan_landmark_never_ready(self):
        obs = person()
        points = list(obs.landmarks)
        points[0] = Landmark(float('nan'), .1)
        self.assertFalse(assess(Observation(1, obs.box, tuple(points)), 'full').ready)

    def test_direction_is_subject_relative_before_preview_mirroring(self):
        self.assertEqual(assess(person(.35), 'full').code, 'move_subject_left')
        self.assertEqual(assess(person(.65), 'full').code, 'move_subject_right')

    def test_head_margin(self):
        p = person()
        obs = Observation(1, (.3, .005, .7, .92), p.landmarks)
        self.assertEqual(assess(obs, 'full').code, 'head_margin')


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.gate = CaptureGate(1, 3)
        self.advice = assess(person(), 'full')
        self.sig = signature(person(), 'full')

    def update(self, t, advice=None, sig=None):
        return self.gate.update(t, advice or self.advice, sig or self.sig)

    def test_never_shoot_without_request(self):
        for i in range(20):
            self.assertNotEqual(self.update(i * .5)[0], 'capture')

    def test_one_request_one_shot_after_stability_and_countdown(self):
        self.gate.request(0)
        states = [self.update(i * .5)[0] for i in range(20)]
        self.assertEqual(states.count('capture'), 1)
        self.assertEqual(states[8], 'capture')

    def test_lost_person_at_last_moment_prevents_capture(self):
        self.gate.request(0)
        for i in range(8):
            self.update(i * .5)
        self.assertEqual(self.update(4, assess(Observation(0), 'full'))[0], 'interrupted')
        self.assertNotEqual(self.update(4.1)[0], 'capture')

    def test_slow_drift_resets_stability(self):
        self.gate.request(0)
        for i in range(4):
            sig = tuple(v + i * .01 for v in self.sig)
            state, _ = self.update(i * .5, sig=sig)
        self.assertEqual(state, 'interrupted')
        self.assertIsNone(self.gate.countdown_at)

    def test_observation_gap_cancels_countdown(self):
        self.gate.request(0)
        for t in (0, .5, 1):
            self.update(t)
        self.assertEqual(self.update(4.1)[0], 'interrupted')

    def test_cancel_and_expiry(self):
        self.gate.request(0)
        self.gate.cancel()
        self.assertEqual(self.update(1)[0], 'idle')
        self.gate.request(1)
        self.assertEqual(self.update(47)[0], 'expired')

    def test_walk_delay_requires_fresh_stability_then_single_capture(self):
        self.gate.request(0, delay=8)
        for i in range(16):
            self.assertEqual(self.update(i * .5)[0], 'preparing')
        states = [self.update(8 + i * .5)[0] for i in range(15)]
        self.assertEqual(states[0], 'stabilizing')
        self.assertEqual(states[8], 'capture')
        self.assertEqual(states.count('capture'), 1)


class IntentTests(unittest.TestCase):
    def test_valid_json_only_triggers_known_actions(self):
        self.assertEqual(parse_intent('```json\n{"intent":"upper"}\n```'), 'upper')
        self.assertEqual(parse_intent('{"intent":"delete"}'), 'unknown')
        self.assertEqual(parse_intent('shoot'), 'unknown')
        self.assertEqual(parse_intent('{"intent": "cancel"}'), 'cancel')

    def test_malformed_intent_is_unknown(self):
        self.assertEqual(parse_intent('{"intent": ["shoot"]}'), 'unknown')


class VoiceControlTests(unittest.TestCase):
    def test_number_words_choose_modes(self):
        for phrase in ['하나', '하나.', '1', '1번', '일 번']:
            self.assertEqual(direct_intent(phrase), 'full')
        for phrase in ['둘', '둘.', '2', '2번', '이 번']:
            self.assertEqual(direct_intent(phrase), 'upper')
        for phrase in ['둘 다 싫어', '하나는 아니야', '두 사람']:
            self.assertIsNone(direct_intent(phrase))

    def test_short_commands(self):
        for phrase, intent in [('사진', 'shoot'), ('-사진.', 'shoot'), ('촬영', 'shoot'),
                               ('준비', 'prepare'), ('전신', 'full'), ('상반신', 'upper'), ('구도', 'status')]:
            self.assertEqual(direct_intent(phrase), intent)

    def test_reported_misrecognitions_never_trigger_gemma_action_guess(self):
        for phrase in ['만지신 사진', '삭발진 모드.', '신 모드.', '지구', '-좋아.']:
            worker, director = AssistantWorker({}), Mock()
            director.intent.return_value = 'full'
            worker.process(phrase, director, None, source='voice')
            director.intent.assert_not_called()
            intents = []
            while not worker.events.empty():
                kind, value = worker.events.get_nowait()
                if kind == 'intent':
                    intents.append(value)
            self.assertEqual(intents, ['unknown'])
            self.assertFalse(worker.mode_selected)

    def test_prepare_acknowledges_before_starting_walk_timer(self):
        worker, director, audio = AssistantWorker({}), Mock(), Mock()
        def check_before_timer(text):
            self.assertNotIn(('intent', 'prepare'), list(worker.events.queue))
        audio.say.side_effect = check_before_timer
        worker.process('준비', director, audio, source='voice')
        self.assertIn(('intent', 'prepare'), list(worker.events.queue))
        director.intent.assert_not_called()

    def test_spoken_full_body_phrases(self):
        for phrase in ['나 전신 사진 찍을 거야', '전신 모드', '전신 사진 찍고 싶어.', '전신으로 바꿔 줘']:
            self.assertEqual(direct_intent(phrase), 'full', phrase)

    def test_spoken_upper_body_phrases(self):
        for phrase in ['상반신 사진 찍을 거야', '상반신으로 바꿔 줘', '상반신 모드로 바꿔 줘']:
            self.assertEqual(direct_intent(phrase), 'upper', phrase)

    def test_negative_or_ambiguous_phrases_need_gemma(self):
        for phrase in ['전신 말고 상반신', '상반신 모드로 바꾸지 마', '전신은 싫어', '상반신이 뭐야?']:
            self.assertIsNone(direct_intent(phrase))
        self.assertEqual(direct_intent('사진 찍지 마'), 'cancel')

    def test_stt_text_emits_real_mode_change_and_confirmation_without_llm(self):
        worker, director, speaker = AssistantWorker({}), Mock(), Mock()
        # The microphone and terminal both call this same process method.
        worker.process('상반신으로 바꿔 줘', director, speaker)
        director.intent.assert_not_called()
        events = []
        while not worker.events.empty():
            events.append(worker.events.get_nowait())
        self.assertIn(('intent', 'upper'), events)
        self.assertIn('상반신', speaker.say.call_args.args[0])
        self.assertNotIn(('intent', 'shoot'), events)

    def test_complex_expression_uses_gemma(self):
        worker, director = AssistantWorker({}), Mock()
        director.intent.return_value = 'full'
        worker.process('다리까지 나오는 사진으로 바꿔 줄래?', director, None)
        director.intent.assert_called_once()


class AutomaticPhotoTests(unittest.TestCase):
    def setUp(self):
        self.gate = CaptureGate()
        self.advice = assess(person(), 'full')
        self.sig = signature(person(), 'full')

    def test_mode_command_alone_takes_one_photo_after_four_second_countdown(self):
        for word in ['하나', '둘']:
            self.gate.command(direct_intent(word), 0)
            states = [self.gate.update(i * .5, self.advice, self.sig) for i in range(21)]
            self.assertEqual(states[1], ('countdown', 4))
            self.assertEqual(states[9], ('capture', 0))
            self.assertEqual(sum(state == 'capture' for state, _ in states), 1)

    def test_loss_of_composition_restarts_full_timer(self):
        self.gate.command('upper', 0)
        for i in range(8):
            self.gate.update(i * .5, self.advice, self.sig)
        self.assertEqual(self.gate.update(4, assess(Observation(0), 'full'), None)[0], 'interrupted')
        self.assertEqual(self.gate.update(4.5, self.advice, self.sig)[0], 'stabilizing')
        self.assertEqual(self.gate.update(5, self.advice, self.sig), ('countdown', 4))

    def test_cancel_stops_automatic_shot_until_new_command(self):
        self.gate.command('full', 0)
        self.gate.command('cancel', .1)
        for i in range(20):
            self.assertNotEqual(self.gate.update(i * .5, self.advice, self.sig)[0], 'capture')
        self.gate.command('shoot', 10)
        states = [self.gate.update(10 + i * .5, self.advice, self.sig)[0] for i in range(10)]
        self.assertEqual(states[-1], 'capture')

    def test_changing_modes_restarts_countdown(self):
        self.gate.command('full', 0)
        for i in range(8):
            self.gate.update(i * .5, self.advice, self.sig)
        self.gate.command('upper', 4)
        self.assertEqual(self.gate.update(4, self.advice, self.sig)[0], 'stabilizing')


class CPUConfigTests(unittest.TestCase):
    def test_classroom_paths_reuse_course_environment(self):
        config = load_config('config.jetson.json')
        self.assertTrue(config['piper_python'].endswith('.piper_venv/bin/python') or
                        config['piper_python'].endswith('.piper_venv\\bin\\python'))
        self.assertIn('build-cpu', config['whisper_executable'])
        self.assertEqual(config['audio_backend'], 'alsa')
        self.assertTrue(config['yolo_model'].endswith('.pt'))

    def test_gemma_disables_layer_and_kv_offload(self):
        from gemma import GemmaDirector
        module = Mock()
        with patch.dict('sys.modules', {'llama_cpp': module}):
            GemmaDirector(load_config('config.jetson.json'))
        options = module.Llama.call_args.kwargs
        self.assertEqual(options['n_gpu_layers'], 0)
        self.assertIs(options['offload_kqv'], False)

    def test_whisper_is_forced_to_cpu(self):
        audio = AudioIO.__new__(AudioIO)
        audio.config = {'whisper_executable': 'whisper-cli', 'whisper_model': 'base.bin', 'cpu_threads': 2}
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / 'command.wav'
            def result(command, **kwargs):
                prefix = command[command.index('-of') + 1]
                Path(prefix + '.txt').write_text('전신 모드', encoding='utf-8')
            with patch('audio_io.subprocess.run', side_effect=result) as run:
                self.assertEqual(audio.transcribe(wav), '전신 모드')
                self.assertIn('--no-gpu', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
