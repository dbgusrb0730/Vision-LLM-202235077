# Project01: 초록색 공 추적 및 손과의 관계 시각화

202235077 유현규

## 프로젝트 소개

`02_Computer-Vision.ipynb`에서 배운 영상 처리 기술을 활용하여 카메라 영상 속 초록색 공을 검출하고 추적합니다. MediaPipe로 손을 인식하고, 공과 손바닥 사이의 화면상 거리에 따라 `NEAR` 또는 `FAR`를 표시합니다.

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
| 관계 시각화 | 공과 가장 가까운 손바닥 사이 거리 및 NEAR/FAR 표시 |

## 결과 및 조작 방법

- 파란색 원과 윤곽선: 현재 프레임에서 검출한 공
- 노란색 원과 궤적: 칼만 필터가 추정한 공의 위치와 이동 경로
- 손의 점과 연결선: 검출된 손 관절과 손가락 구조
- 자홍색 선: 검출한 공의 중심과 가장 가까운 손바닥 중심을 연결한 선
- `TRACKING`: 공이 검출되어 추적 중
- `PREDICTED (occluded)`: 공이 일시적으로 검출되지 않아 위치를 예측 중
- `NO BALL` / `NO HAND`: 공 또는 손이 검출되지 않음
- `NEAR` / `FAR`: 공 반지름과 손바닥 크기를 기준으로 화면상 근접 여부를 판정하며 거리도 픽셀 단위로 표시
- `q` 또는 `ESC`: 종료
- `r`: 추적 상태 초기화

## 참고 사항

공이 1초를 초과하여 검출되지 않으면 추적 상태를 해제합니다. 예측만 수행하는 구간에서는 손과의 근접 여부를 판정하지 않습니다.

NEAR/FAR는 2차원 영상에서의 거리 판정이며 실제 접촉이나 3차원 거리를 의미하지 않습니다. 조명, 배경의 초록색 물체, 공의 가려짐에 따라 검출 결과가 달라질 수 있습니다. 필요하면 코드의 HSV 범위를 조절하거나 `--min-area` 옵션으로 최소 검출 면적을 설정합니다.

합성 영상으로 공 검출, 비원형 영역 제외, 추적, 미검출 시 초기화 및 재검출 동작을 확인했습니다. 실제 카메라와 MediaPipe 손 인식의 통합 동작은 별도 확인이 필요합니다.
