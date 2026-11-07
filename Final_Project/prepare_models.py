"""Explicit model preparation. Inference itself never downloads models."""
import argparse
import shutil
import urllib.request
from settings import ROOT

REPO_ID = 'bartowski/google_gemma-4-E2B-it-GGUF'
MODEL_FILE = 'google_gemma-4-E2B-it-Q4_K_M.gguf'


def fetch(url, relative, existing=None):
    target = ROOT / 'models' / relative
    if target.is_file():
        print('이미 있음:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    if existing and existing.is_file():
        shutil.copyfile(existing, temporary)
        print('기존 강의 모델 복사:', existing)
    else:
        print('다운로드:', relative, flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as f:
            shutil.copyfileobj(response, f)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--gemma', action='store_true')
    parser.add_argument('--speech', action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('Choose --vision, --gemma, --speech (or combine them).')
    if args.vision:
        fetch('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
              'YOLO/yolo11n.pt', ROOT.parent / 'src/models/YOLO/yolo11n.pt')
        fetch('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
              'MediaPipe/pose_landmarker_full.task', ROOT.parent / 'src/models/MediaPipe/pose_landmarker_full.task')
    if args.gemma:
        import os
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download
        load_dotenv(ROOT / '.env')
        load_dotenv(ROOT.parent / '.env')
        hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, token=os.getenv('HF_TOKEN'),
                         local_dir=ROOT / 'models/Gemma4')
    if args.speech:
        fetch('https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
              'Whisper/ggml-base.bin')
        for filename in ('ko_KR-kss-medium.onnx', 'ko_KR-kss-medium.onnx.json', 'MODEL_CARD'):
            fetch('https://huggingface.co/rhasspy/piper-voices/resolve/main/ko/ko_KR/kss/medium/' + filename,
                  'Piper/' + filename)
    print('모델 준비 완료. whisper.cpp 실행 파일과 Piper 환경은 README에 따라 별도 준비하세요.')


if __name__ == '__main__':
    main()
