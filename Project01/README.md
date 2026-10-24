# Project01: 초록색 공 추적 및 손 접근 경고 시스템

202235077 유현규

## 프로젝트 소개

`02_Computer-Vision.ipynb`에서 배운 영상 처리 기술을 활용하여 카메라 영상 속 초록색 공을 검출하고 추적합니다. 공을 위험 물체의 대용으로 사용하고 MediaPipe로 손을 인식하여, 접근 정도에 따라 `SAFE`, `CAUTION`, `DANGER`를 표시하는 시연 시스템입니다. 위험 상태에서는 빨간 테두리와 경고 문구를 표시하고 위험 구역 진입 횟수를 집계합니다.

## 제출 파일

```text
Project01/
├── README.md
├── 02_Computer_Vision_Project.py
└── hand_landmarker.task
```

- `02_Computer_Vision_Project.py`: 독립적으로 실행하는 프로젝트 코드
- `hand_landmarker.task`: MediaPipe 손 인식에 필요한 사전 학습 모델
- `README.md`: 의존성, 실행 방법 및 결과 설명

모델은 실행 파일과 같은 폴더에서 불러오므로 저장소의 다른 폴더나 노트북을 실행할 필요가 없습니다. 기본 실행에는 별도 입력 이미지나 영상 파일이 필요하지 않습니다.

## 의존성 및 준비물

- Python 3와 NumPy, OpenCV, MediaPipe
- 카메라와 초록색 공
- 화면을 표시할 수 있는 실행 환경

PC에서 실행에 사용할 Python으로 패키지를 설치합니다.

```sh
python -m pip install numpy opencv-contrib-python mediapipe
```

VS Code에서 선택한 Python과 터미널의 Python 버전을 동일하게 설정해야 합니다. Jetson에서는 JetPack에서 제공하는 OpenCV를 유지하고, 나머지 필요한 패키지만 추가 설치합니다.

- 모델 출처: [MediaPipe Hand Landmarker 모델]

## 실행 방법

저장소 최상위 폴더에서 실행합니다.

```sh
cd Project01
python 02_Computer_Vision_Project.py
```

기본값은 0번 카메라입니다. 초록색 공과 손을 카메라에 함께 비추면 검출·추적 결과와 이진 마스크가 각각 창에 표시됩니다.

다른 카메라 또는 좌우 반전 화면을 사용하는 예시:

```sh
python 02_Computer_Vision_Project.py --source 1
python 02_Computer_Vision_Project.py --mirror
```

영상 파일을 입력으로 사용하는 경우 해당 파일은 별도로 준비합니다.

```sh
python 02_Computer_Vision_Project.py --source "video.mp4"
```

Jetson CSI 카메라 실행:

```sh
python 02_Computer_Vision_Project.py --jetson
```

이 옵션은 GStreamer를 지원하는 Jetson 환경에서 사용합니다.

### USB 카메라용 YUY2 파이프라인

USB 카메라가 YUYV 형식을 지원하면, 코드의 `if args.jetson:` 안에 있는 `pipeline = (...)`을 아래 예시로 교체합니다. V4L2에서 표시하는 `YUYV`는 GStreamer에서 `YUY2`로 지정합니다.

```python
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
```

바로 아래의 카메라 생성 코드는 그대로 사용합니다.

```python
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

장치 경로, 해상도, 프레임 속도는 카메라가 지원하는 조합으로 지정해야 합니다. Jetson 터미널에서 다음 명령으로 확인합니다.

```sh
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

교체한 파이프라인을 선택하려면 다음 명령으로 실행합니다.

```sh
python 02_Computer_Vision_Project.py --jetson
```

현재 코드에서 `--jetson`은 지정된 GStreamer 파이프라인을 사용하는 옵션입니다. 옵션 없이 기본 카메라가 정상 작동한다면 파이프라인 교체 없이 실행해도 됩니다. 이 예시는 OpenCV의 GStreamer 지원이 필요하며, 실제 카메라에서의 동작은 별도로 확인해야 합니다.

## 내용

| 기술 | 적용 내용 |
| --- | --- |
| 이미지 조정 | 가로 960픽셀 이하로 축소, 선택적 좌우 반전 |
| 색공간 변환 | BGR → LAB, LAB → BGR, BGR → HSV 및 RGB |
| Image Enhancement | LAB의 L 채널에 CLAHE를 적용하여 명도 대비 개선 |
| 필터링 | Gaussian Blur로 영상 잡음 완화 |
| Color Segmentation | HSV 범위로 초록색 영역의 이진 마스크 생성 |
| Morphological Operation | Opening과 Closing으로 작은 잡음과 구멍 정리 |
| Contour Detection | 면적·원형도 조건을 만족하는 가장 큰 윤곽선 선택, 중심·반지름 계산 |
| 객체 추적 | 칼만 필터로 위치와 속도를 추정하고 짧은 미검출 구간의 위치 예측 |
| 신체 인식 | MediaPipe로 최대 두 손의 관절을 검출하고 연결선 표시 |
| 접근 경고 | 손 크기로 정규화한 거리에 따라 SAFE/CAUTION/DANGER 표시 및 위험 진입 횟수 집계 |

## 결과 및 조작 방법

- 파란색 원과 윤곽선: 현재 프레임에서 검출한 공
- 노란색 원과 궤적: 칼만 필터가 추정한 공의 위치와 이동 경로
- 손의 점과 연결선: 검출된 손 관절과 손가락 구조
- 상태별 색상의 선: 공과 정규화 거리 기준으로 가장 가까운 손바닥을 연결한 선
- `TRACKING`: 공이 검출되어 추적 중
- `PREDICTED (occluded)`: 공이 일시적으로 검출되지 않아 위치를 예측 중
- `NO BALL` / `NO HAND`: 공 또는 손이 검출되지 않음
- `SAFE` (초록색): 손이 공에서 충분히 떨어져 있음
- `CAUTION` (노란색): 손이 공에 접근 중
- `DANGER` (빨간색): 손이 공에 매우 가까움. 빨간 화면 테두리와 경고 문구 표시
- `UNKNOWN` (회색): 공 또는 손이 검출되지 않아 판정 불가. 예측 위치만 있을 때도 동일
- `Danger entries`: 위험 상태에 진입한 횟수. 위험 상태 유지 중에는 증가하지 않음
- `Surface gap / palm size`: 공 표면까지의 영상상 거리를 손바닥 크기로 나눈 값
- `q` 또는 `ESC`: 종료
- `r`: 추적 상태와 위험 진입 횟수 초기화

### 경고 판정 기준

정규화 거리 `gap_ratio = max(0, (손바닥 중심과 공 중심 사이 거리 - 공 반지름) / 손바닥 크기)`를 사용합니다. 손바닥 크기는 손목과 중지 시작 관절 사이의 픽셀 거리입니다. 양손이 보이면 정규화 거리가 더 작은 손을 기준으로 판정합니다.

- `gap_ratio <= 0.7`: DANGER 진입
- DANGER 진입 후에는 `gap_ratio >= 1.0`일 때 해제하여 경계 부근의 중복 집계를 완화
- 위험 상태가 아니고 `gap_ratio <= 2.0`: CAUTION
- `gap_ratio > 2.0`: SAFE

검출이 잠깐 끊기면 UNKNOWN을 표시하면서 이전 위험 진입 상태는 보존합니다. 따라서 위험 상태에서 검출이 끊겼다가 같은 위치에서 다시 검출되어도 횟수가 중복 증가하지 않습니다. 손이 해제 거리까지 멀어진 것이 관측된 뒤 다시 접근해야 새 진입으로 집계합니다. 각 기준값은 시연을 위한 경험적 설정이며 코드의 `ProximityWarning.update()`에서 조정할 수 있습니다.

## 참고 사항

공이 1초를 초과하여 검출되지 않으면 추적 상태를 해제합니다. 예측만 수행하는 구간에서는 손과의 근접 여부를 판정하지 않습니다.

SAFE/CAUTION/DANGER는 2차원 영상에서의 거리 판정이며 실제 접촉이나 3차원 거리를 의미하지 않습니다. 실제 안전장치로 사용할 수 없으며, SAFE 표시는 현실의 안전을 보장하지 않습니다. 조명, 배경의 초록색 물체, 공의 가려짐에 따라 검출 결과가 달라질 수 있습니다. 필요하면 코드의 HSV 범위를 조절하거나 `--min-area` 옵션으로 최소 검출 면적을 설정합니다.

합성 영상으로 공 검출, 비원형 영역 제외, 추적, 미검출 시 초기화 및 재검출 동작을 확인했습니다. 실제 카메라와 MediaPipe 손 인식의 통합 동작은 별도 확인이 필요합니다.
