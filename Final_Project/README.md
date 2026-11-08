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

카메라 읽기·화면 표시·사진 저장은 메인 스레드, YOLO·MediaPipe 분석은 별도 작업 스레드, 음성·Gemma 처리는 별도 음성 스레드에서 실행합니다. 터미널 입력도 별도 스레드에서 받습니다. 분석은 한 번에 하나만 실행하고 작업 중 들어온 프레임을 대기열에 쌓지 않습니다.

**Gemma는 사진을 직접 분석하지 않습니다.** 구도 판정과 기본 이동 방향은 `framing.py`가 결정합니다. 사용자가 “구도”라고 질문하면 현재 모드와 판정 문장을 Gemma에 전달해 설명을 생성하고 Piper로 읽습니다. 정해진 음성 명령은 규칙 기반으로 처리하며, 규칙 밖의 자유 문장은 **터미널에 입력했을 때만** Gemma가 해석합니다.

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
| `check_gemma.py` | 실제 Gemma 모델 로드와 의도 분류 검사 |
| `presentation/` | 발표 PPT·PDF 및 시연 영상 |
| `requirements*.txt` | 환경별 의존성 안내 |

설정 안의 상대 경로는 실행한 터미널 위치가 아닌 **Final_Project 폴더 기준**입니다.

## Jetson Orin Nano 실행

강의에서 사용한 Python 가상환경을 활성화합니다. 아래 명령은 저장소 폴더명이 `vision-llm-202235077`인 경우입니다. Linux에서는 대소문자를 구분하므로 실제 폴더명이 `Vision-LLM-202235077`이면 그 이름을 사용하세요.

```bash
cd ~/vision-llm/vision-llm-202235077
python Final_Project/main.py --config config.jetson.json --check
python Final_Project/main.py --config config.jetson.json
```

00~06 실습에서 준비한 환경과 모델을 재사용합니다. `--check`에서 누락된 항목이 나올 때만 실제 경로와 패키지를 확인합니다. Pillow가 없으면 활성화된 환경에 `python -m pip install Pillow`로 추가합니다. 기본 한글 폰트 경로는 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`입니다. 경로가 없으면 `fc-list :lang=ko file family`로 설치된 한글 글꼴을 찾아 `font_path`를 변경합니다.

Jetson에는 Windows용 `requirements.txt`를 통째로 설치하지 않습니다.

### 카메라와 마이크

설정 파일에는 `camera_backend`를 `gstreamer`로 지정합니다. 해당 카메라 형식과 OpenCV의 GStreamer 지원이 필요합니다.

Jetson 음성 입력은 `pasuspender -- arecord`로 5초씩 녹음하고 Whisper CPU 실행 파일로 인식합니다. Piper는 별도 `.piper_venv`에서 실행하고 `aplay`로 재생합니다.

```bash
python Final_Project/main.py --config config.jetson.json --audio-devices
```

카메라 기본 경로는 `/dev/video0`, 마이크 기본값은 `plughw:3,0`, `output_device: null`은 기본 스피커입니다. 이 값들은 모든 Jetson에서 동일하지 않습니다.

```bash
v4l2-ctl --list-devices
arecord -l
aplay -l
```

카메라는 `v4l2-ctl -d /dev/video번호 --list-formats-ext`에서 YUYV 640×480, 30 FPS 캡처 지원을 확인하고 `camera` 파이프라인의 장치 경로만 수정합니다. 마이크·스피커는 각각 실제 card/device 번호에 맞게 `input_device`, `output_device`를 지정합니다. 예를 들어 실제 마이크가 card 3, device 0이면 `plughw:3,0`입니다. 장치 번호는 재연결 후 바뀔 수 있습니다.

상대 경로는 다음 강의 폴더 구조를 기준으로 합니다.

```text
~/vision-llm/
├─ .venv/                       # 강의용 Python 환경
├─ .piper_venv/                 # Piper 전용 환경
├─ whisper.cpp/
│  ├─ build-cpu/bin/whisper-cli
│  └─ models/ggml-base.bin
├─ src/models/Piper/
└─ vision-llm-202235077/
   ├─ src/models/               # YOLO, MediaPipe, Gemma4
   └─ Final_Project/
```

## Windows 노트북 실행

### 현재 준비된 환경

PowerShell에서 `Final_Project` 폴더로 이동한 뒤 실행합니다. 아래 경로는 개발 노트북 예시이므로 본인의 저장소 위치에 맞춥니다.

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
- 실제 음성 인식은 마이크·거리·소음에 영향을 받습니다. 노트북에서는 오인식이 관찰됐으며, Jetson에서는 음성 녹음·재생 시간 초과도 보고되어 안정성 검증이 더 필요합니다.



## Jetson 프레임 문제와 개선

강의실 CPU 실행에서 초기 속도는 약 **1.8 FPS**, YOLO 처리 시간은 **약 500ms대**, MediaPipe Pose는 **약 50~100ms**로 확인했습니다. 영상 분석을 순차 실행하는 동안 화면 갱신도 기다리던 것이 끊김의 원인이었습니다. 분석 크기를 줄이는 것만으로는 체감 개선이 확인되지 않았습니다.

현재 구현에는 다음 개선이 들어 있습니다.

- 미리보기와 영상 분석 분리: 작업이 끝나면 최신 프레임을 분석하고 대기 프레임은 쌓지 않습니다.
- YOLO 실행 후 0.35초 동안 사람 수·박스를 재사용하며 관절을 추적합니다.
- 한글 안내 패널을 최대 32개 캐시하고 FPS 표시를 1초 간격으로 갱신합니다.
- GStreamer의 해상도는 파이프라인에서 지정하고 카메라를 연 뒤 다시 설정하지 않습니다.
- Jetson에서는 창 상태 조회에 의존하지 않고 `Q` 또는 `Esc`로 종료합니다.


## 제출 자료

- [발표 자료 PDF](presentation/Smart_Photo_Director.pdf)
- [편집 가능한 PPT](presentation/Smart_Photo_Director.pptx)
- [시연 영상](presentation/ㅇㅇㅇ.mp4)

모델, 가상환경, 토큰 파일, 촬영 결과, 임시 파일, 외부 실행 도구는 `.gitignore`로 제외합니다. 모델은 위 준비 절차 또는 기존 강의 환경에서 별도로 준비합니다.

## 참고 자료

- 강의 `00_Initial-Setup.ipynb`, `05-LLM_and_Gemma.ipynb`, `06-Vision_LLM_Multimodal_Systems.ipynb`
- [Gemma GGUF 모델](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Piper](https://github.com/OHF-Voice/piper1-gpl)
- [한국어 Piper 모델](https://huggingface.co/rhasspy/piper-voices/tree/main/ko/ko_KR/kss/medium)
