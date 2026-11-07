"""Offline speech: sounddevice microphone -> whisper.cpp -> Piper -> speaker.

Only the assistant worker uses audio, so synthesized speech cannot be recorded
by this application at the same time. No remote transcription service is used.
"""
from pathlib import Path
import subprocess
import tempfile
import time
import wave
import os
import signal
import shutil


class AudioIO:
    def __init__(self, config):
        import numpy as np
        self.np, self.config = np, config
        self.backend = config.get('audio_backend', 'sounddevice')
        self.sd = None
        for key in ('whisper_executable', 'whisper_model', 'piper_python', 'piper_model'):
            if not Path(config[key]).is_file():
                raise FileNotFoundError(f'{key}: {config[key]} — README의 음성 설정을 확인하세요.')
        if not Path(config['piper_model'] + '.json').is_file():
            raise FileNotFoundError('Piper .onnx.json 설정 파일이 필요합니다.')
        if self.backend == 'alsa':
            commands = ['arecord', 'aplay'] + (['pasuspender'] if config.get('use_pasuspender') else [])
            for command in commands:
                if not shutil.which(command):
                    raise FileNotFoundError(f'강의 음성 실행 도구가 없습니다: {command}')
        else:
            import sounddevice as sd
            self.sd = sd
            sd.check_input_settings(device=config['input_device'], channels=1,
                                    samplerate=config['sample_rate'], dtype='int16')
        # Synthesize once before the camera can be armed, avoiding Piper process
        # startup delays during the countdown.
        self.countdown_audio = {i: self.synthesize(str(i))
                                for i in range(1, config['countdown_seconds'] + 1)}

    def listen(self, stop, interrupt, on_speech_started=lambda: None):
        """Energy-gated speech segments, with pre-roll and silence endpointing.

        Keeps a longer listening window to avoid frequent stream reopen gaps.
        Interrupt recording for urgent UI actions; discard partial utterances.
        """
        if self.backend == 'alsa':
            return self.listen_alsa(stop, interrupt, on_speech_started)
        np, c = self.np, self.config
        sr, block_seconds = c['sample_rate'], .1
        block = int(sr * block_seconds)
        chunks, preroll = [], []
        active, silence, elapsed = False, 0.0, 0.0
        with self.sd.InputStream(samplerate=sr, channels=1, dtype='int16',
                                 device=c['input_device'], blocksize=block) as stream:
            while not stop.is_set() and not interrupt():
                data, overflow = stream.read(block)
                if overflow:
                    return ''
                data = data.copy()
                rms = float(np.sqrt(np.mean((data.astype(np.float32) / 32768) ** 2)))
                elapsed += block_seconds
                if not active:
                    preroll.append(data)
                    preroll = preroll[-5:]
                    if rms >= c['speech_rms']:
                        active, chunks = True, list(preroll)
                        on_speech_started()
                    elif elapsed >= 8:
                        return ''
                else:
                    chunks.append(data)
                    silence = silence + block_seconds if rms < c['speech_rms'] else 0
                    if silence >= c['speech_silence_seconds'] or len(chunks) * block_seconds >= c['speech_max_seconds']:
                        break
        if stop.is_set() or interrupt() or not active or len(chunks) < 4:
            return ''
        with tempfile.TemporaryDirectory(prefix='photo-stt-') as tmp:
            wav = Path(tmp) / 'command.wav'
            with wave.open(str(wav), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr)
                f.writeframes(np.concatenate(chunks).tobytes())
            return self.transcribe(wav)

    def transcribe(self, wav):
        c = self.config
        prefix = str(Path(wav).with_suffix('')) + '-transcript'
        subprocess.run([c['whisper_executable'], '-m', c['whisper_model'],
                        '-f', str(wav), '-l', 'ko', '-otxt', '-of', prefix, '-nt',
                        '--no-gpu', '-t', str(c.get('cpu_threads', 2)),
                        '--prompt', '사진 촬영 명령. 하나. 둘. 사진. 준비. 구도. 취소. 종료.',
                        '--no-fallback'],
                       capture_output=True, check=True, timeout=120)
        return Path(prefix + '.txt').read_text(encoding='utf-8').strip()

    def listen_alsa(self, stop, interrupt, on_speech_started):
        """Reuse the course's pasuspender/arecord path, no PortAudio dependency.

        A short poll allows keyboard cancellation/countdown to stop recording.
        Stop the whole process group so pasuspender cannot leave arecord behind.
        """
        c = self.config
        with tempfile.TemporaryDirectory(prefix='photo-alsa-') as tmp:
            wav = Path(tmp) / 'command.wav'
            command = ['arecord', '-q', '-D', c['input_device'] or 'default',
                       '-f', 'S16_LE', '-r', str(c['sample_rate']), '-c', '1',
                       '-d', str(c.get('record_seconds', 5)), str(wav)]
            if c.get('use_pasuspender'):
                command = ['pasuspender', '--'] + command
            with tempfile.TemporaryFile() as error:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=error,
                                           start_new_session=True)
                deadline = time.monotonic() + c.get('record_seconds', 5) + 5
                try:
                    while process.poll() is None:
                        if stop.is_set() or interrupt():
                            return ''
                        if time.monotonic() > deadline:
                            raise TimeoutError('arecord 녹음 시간이 초과되었습니다.')
                        stop.wait(.05)
                    if process.returncode:
                        error.seek(0)
                        raise RuntimeError('arecord 오류: ' + error.read().decode('utf-8', errors='replace'))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            if stop.is_set() or interrupt():
                return ''
            with wave.open(str(wav), 'rb') as f:
                samples = self.np.frombuffer(f.readframes(f.getnframes()), dtype=self.np.int16)
            # Gate silence before Whisper to reduce hallucinated commands.
            blocks = int(c['sample_rate'] * .1)
            usable = samples[:len(samples) // blocks * blocks]
            if not usable.size:
                return ''
            energy = self.np.sqrt(self.np.mean((usable.reshape(-1, blocks).astype(self.np.float32) / 32768) ** 2, axis=1))
            if self.np.count_nonzero(energy >= c['speech_rms']) < 2:
                return ''
            on_speech_started()
            return self.transcribe(wav)

    def synthesize(self, text):
        c = self.config
        with tempfile.TemporaryDirectory(prefix='photo-tts-') as tmp:
            output = Path(tmp) / 'response.wav'
            subprocess.run([c['piper_python'], '-m', 'piper', '-m', c['piper_model'],
                            '-f', str(output), '--', text],
                           capture_output=True, check=True, timeout=45)
            with wave.open(str(output), 'rb') as f:
                if f.getsampwidth() != 2:
                    raise RuntimeError('Piper 출력은 16-bit PCM WAV여야 합니다.')
                rate, channels = f.getframerate(), f.getnchannels()
                samples = self.np.frombuffer(f.readframes(f.getnframes()), dtype=self.np.int16)
            return samples.reshape(-1, channels).copy(), rate

    def play(self, audio):
        samples, rate = audio
        if self.backend == 'alsa':
            with tempfile.TemporaryDirectory(prefix='photo-play-') as tmp:
                wav = Path(tmp) / 'play.wav'
                with wave.open(str(wav), 'wb') as f:
                    f.setnchannels(samples.shape[1])
                    f.setsampwidth(2)
                    f.setframerate(rate)
                    f.writeframes(samples.tobytes())
                command = ['aplay', '-q']
                if self.config['output_device']:
                    command += ['-D', str(self.config['output_device'])]
                subprocess.run(command + [str(wav)], check=True, capture_output=True, timeout=45)
        else:
            self.sd.play(samples, samplerate=rate,
                         device=self.config['output_device'], blocking=True)

    def count(self, number):
        self.play(self.countdown_audio[number])

    def say(self, text):
        self.play(self.synthesize(text))
        time.sleep(.2)  # Let speaker reverberation decay before reopening the microphone.
