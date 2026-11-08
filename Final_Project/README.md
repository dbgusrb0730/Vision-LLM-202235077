# Final_Project — Smart Photo Director

**202235077 유현규 · Vision–LLM 융합 시청각 멀티모달 시스템 최종 프로젝트**

혼자 삼각대 앞에서 사진을 찍을 때, 카메라 화면을 보러 왕복하지 않고 음성 안내를 받아 구도를 맞추는 촬영 도우미입니다. 모드를 선택하면 구도를 확인하고 자동으로 한 장을 촬영합니다.

YOLO, MediaPipe Pose, 로컬 Gemma, Whisper.cpp, Piper를 사용합니다. **노트북과 Jetson Orin Nano 모두 CPU로 실행**

## 주요 기능과 사용법

| 음성 명령 | 동작 | 키보드 |
|---|---|---|
| 하나 | 전신 모드 선택과 자동 촬영 대기 | `1` |
| 둘 | 가까운 상반신 모드 선택과 자동 촬영 대기 | `2` |
| 사진 | 현재 모드에서 한 장 재촬영 예약 | `Space` |
| 준비 | 8초 이동 시간 후 자동 촬영 대기 | — |
| 구도 | Gemma가 현재 구도 판정 설명 | — |
| 취소 | 촬영 예약 취소 | `C` |
| 종료 | 프로그램 종료 | `Q` / `Esc` |

1. 시작 안내가 끝나고 **듣는 중**이 표시되면 하나 또는 둘이라고 말합니다.
2. 안내에 따라 촬영 위치와 구도를 맞춥니다.
3. 구도가 **0.5초 안정적으로 유지되면 4초 카운트다운** 후 한 장을 저장합니다.
4. 촬영 후에는 멈춥니다. 사진 또는 모드 명령으로 다음 촬영을 시작합니다.

움직임, 인물 미검출, 다인 감지, 처리 지연이 발생하면 카운트다운을 초기화합니다. 촬영 예약은 45초 후 만료되며 준비 명령에는 이동 시간 8초가 추가됩니다.

- **전신:** 얼굴·어깨·허리를 확인하며, 발목·발끝이 잘려도 허용합니다.
- **상반신:** 얼굴과 양쪽 어깨 중심의 클로즈업입니다. 허리·다리는 보이지 않아도 됩니다.
- 사진은 `Final_Project/photos/`에 JPEG와 촬영 정보 JSON으로 저장합니다. 저장 사진에는 안내 글씨·박스·미리보기 좌우 반전을 적용하지 않습니다.
- 안내 재생, Gemma 처리, 카운트다운 중에는 음성을 듣지 않습니다. 즉시 취소는 키보드 `C`를 사용합니다.
- 음성 명령은 정해진 표현을 사용합니다. 불명확한 인식 결과로 모드나 촬영을 추측하지 않습니다. 터미널의 자유로운 문장은 Gemma가 해석합니다.

## 시스템 구성

```text
카메라 → YOLO 사람 탐지 → MediaPipe 관절 인식 → 구도·움직임 판정 → 자동 촬영
                                                   ↓
                                              한국어 구도 안내
마이크 → Whisper 한국어 인식 → 명령 처리 / Gemma 구도 설명 → Piper → 스피커
```

카메라 처리는 메인 스레드, 음성과 Gemma 처리는 별도 스레드에서 실행합니다. 짧은 명령과 자동 이동 안내는 코드가 처리하고, Gemma는 관측 정보를 바탕으로 구도 설명과 터미널 문장 해석을 담당합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `main.py` | 실행, 카메라 반복 처리, 화면 표시, 사진 저장 |
| `vision.py` | YOLO와 MediaPipe 추론 |
| `framing.py` | 구도 판정과 촬영 상태 관리 |
| `commands.py` | 음성·문자 명령 변환 |
| `assistant_worker.py` | 백그라운드 음성·Gemma 처리 |
| `audio_io.py` | 녹음, Whisper 실행, Piper 합성, 음성 재생 |
| `gemma.py` | 로컬 Gemma 로드와 프롬프트 처리 |
| `settings.py` | 설정 읽기와 경로 처리 |
| `config.jetson.json` / `config.laptop.json` | 장치별 모델·카메라·음성 설정 |
| `prepare_models.py` | 모델 준비 |
| `check_gemma.py` / `test_director.py` | Gemma 연결 검사 / 자동 검사 |
| `requirements*.txt` | 환경별 의존성 안내 |

설정 안의 상대 경로는 실행한 터미널 위치가 아닌 **Final_Project 폴더 기준**입니다.

## Jetson Orin Nano 실행

**Python 가상환경을 활성화한 터미널**에서 실행합니다.

```bash
cd ~/vision-llm/vision-llm-202235077
python Final_Project/main.py --config config.jetson.json --check
python Final_Project/main.py --config config.jetson.json
```

기존 강의 환경에는 새 가상환경 생성이나 모델 재설치가 필요하지 않습니다. `--check`에서 누락된 항목이 나올 때만 실제 경로와 패키지를 확인합니다. Pillow가 없으면 활성화된 환경에 `python -m pip install Pillow`로 추가합니다. 한글 폰트가 없으면 `font_path`를 설치된 한글 폰트로 지정하거나 `sudo apt install fonts-nanum`으로 준비합니다.

Jetson에는 Windows용 `requirements.txt`를 통째로 설치하지 않습니다.

### 카메라와 마이크

설정 파일에는 `camera_backend`를 `gstreamer`로 지정합니다. 해당 카메라 형식과 OpenCV의 GStreamer 지원이 필요합니다.

Jetson 음성 입력은 `pasuspender -- arecord`로 5초씩 녹음하고 Whisper CPU 실행 파일로 인식합니다. Piper는 별도 `.piper_venv`에서 실행하고 `aplay`로 재생합니다.

```bash
python Final_Project/main.py --config config.jetson.json --audio-devices
```

마이크 기본 설정은 `plughw:3,0`입니다. 예를 들어 card 2, device 0이면 `input_device`를 `plughw:2,0`으로 바꿉니다. `output_device: null`은 기본 스피커입니다.

## Windows 노트북 실행

### 현재 준비된 환경

PowerShell에서 Final_Project으로 이동한 뒤 실행합니다.

```powershell
cd C:\Users\dbgus\Documents\GitHub\Vision-LLM-202235077\Final_Project
.\.venv311\Scripts\python.exe main.py --check
.\.venv311\Scripts\python.exe main.py
```

음성 없이 카메라·구도·촬영을 확인하려면:

```powershell
.\.venv311\Scripts\python.exe main.py --vision-only
```

노트북 음성 입력은 sounddevice를 사용합니다. 기본 마이크가 잘못 선택되면 `main.py --audio-devices`로 장치를 확인하고 `input_device`를 변경합니다. 발화 후 약 1초의 무음 뒤 인식을 시작합니다.

### 새 노트북에 처음 설치하는 경우

아래는 Python 3.11이 기본 사용자 경로에 설치된 경우입니다. 실제 Python 위치가 다르면 `$python311`만 변경합니다. 기존 환경에 이미 설치했다면 반복할 필요 없습니다.

```powershell
$python311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $python311 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m pip install --upgrade --only-binary=:all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.\.venv311\Scripts\python.exe prepare_models.py --vision --gemma --speech

& $python311 -m venv .venv-piper
.\.venv-piper\Scripts\python.exe -m pip install --upgrade pip
.\.venv-piper\Scripts\python.exe -m pip install piper-tts
```

Whisper 실행 파일은 별도로 준비합니다. 사용한 공식 Windows CPU 배포 파일:

```powershell
New-Item -ItemType Directory -Force -Path vendor | Out-Null
Invoke-WebRequest -Uri "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-x64.zip" -OutFile "vendor/whisper-bin-x64.zip"
Expand-Archive -LiteralPath "vendor/whisper-bin-x64.zip" -DestinationPath "vendor/whisper-windows" -Force
```

`config.laptop.json`의 `whisper_executable`은 `vendor/whisper-windows/Release/whisper-cli.exe`입니다. 배포 파일 구조가 다르면 실행 파일과 필요한 DLL을 함께 유지하고 실제 경로를 지정합니다.

Gemma 연결 검사:

```powershell
.\.venv311\Scripts\python.exe check_gemma.py
```

## 구도 판정과 제한 사항

- 1인, 고정 카메라, 카메라 정면을 바라보는 자세를 대상으로 합니다.
- 전신은 코·어깨·골반, 상반신은 코·양쪽 어깨의 가시성을 확인합니다.
- 상반신은 어깨 너비가 화면 너비의 22% 이상인지 확인하고, 몸통·팔이 화면 가장자리에서 잘려도 허용합니다.
- 움직임은 시작 위치와 현재 관절 좌표를 비교합니다. 상반신에서 화면 밖 허리·다리의 추정값은 안정 판정에 사용하지 않습니다.
- 구도 기준은 초기 시연용 경험적 값입니다. 실제 거리·보폭이나 미적 완성도를 측정하지 않습니다.
- 관측 간격 또는 분석 시간이 1초를 넘으면 오래된 영상으로 촬영하지 않도록 보류합니다.
- 실제 음성 인식은 마이크·거리·소음에 영향을 받습니다. 노트북에서는 오인식이 관찰됐으며, Jetson의 전체 음성 시연은 실기기 확인이 필요합니다.



## 참고 자료

- 강의 `00_Initial-Setup.ipynb`, `05-LLM_and_Gemma.ipynb`, `06-Vision_LLM_Multimodal_Systems.ipynb`
- [Gemma GGUF 모델](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Piper](https://github.com/OHF-Voice/piper1-gpl)
- [한국어 Piper 모델](https://huggingface.co/rhasspy/piper-voices/tree/main/ko/ko_KR/kss/medium)
