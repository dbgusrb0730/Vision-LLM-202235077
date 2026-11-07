import json
import os
from pathlib import Path

# Applies only to this program and its children; keep the classroom installation.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = Path(__file__).resolve().parent
PATH_KEYS = ('yolo_model', 'pose_model', 'gemma_model', 'whisper_executable',
             'whisper_model', 'piper_python', 'piper_model', 'font_path', 'output_dir')


def load_config(name):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding='utf-8'))
    for key in PATH_KEYS:
        p = Path(config[key]).expanduser()
        config[key] = str(p if p.is_absolute() else ROOT / p)
    for key in ('sample_rate', 'speech_rms', 'speech_silence_seconds', 'speech_max_seconds',
                'stable_seconds', 'countdown_seconds', 'analysis_width', 'guidance_interval'):
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not isinstance(config['countdown_seconds'], int) or config['countdown_seconds'] > 10:
        raise ValueError('countdown_seconds must be an integer from 1 to 10')
    if Path(config['yolo_model']).suffix.lower() != '.pt':
        raise ValueError('CPU 실행에는 YOLO .pt 모델을 지정하세요.')
    return config
