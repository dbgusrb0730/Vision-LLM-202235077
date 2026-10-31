# 261025 실습 과제: GPU 가속과 객체 탐지 후처리

## 1. 과제 개요

이번 과제는 CPU와 GPU의 연산 방식 차이를 확인하고, 이미지 분류 모델의 학습 과정과 객체 탐지의 후처리 알고리즘을 구현하는 것을 목표로 한다.

`03_DL-and-GPU.ipynb`에서는 NumPy와 CuPy를 이용한 이미지 연산, MNIST의 GPU 학습, CIFAR-10의 CNN 학습을 다룬다. `04_DL-Object-Detection.ipynb`에서는 Bounding Box를 이용한 IoU 계산과 중복 박스를 제거하는 NMS를 구현한다.


## 2. 파일 구성

| 파일 | 역할 |
| --- | --- |
| `Daily_Assignment03.py` | MNIST 학습 및 시각화, CuPy 이미지 연산, MNIST·CIFAR-10 GPU 학습 |
| `Daily_Assignment04.py` | 사과 영역 검출, Bounding Box 좌표 변환, IoU, Confidence Thresholding, NMS |
| `README.md` | 구현 내용, 실행 방법, 모델 정보 및 확인 결과 |
| `../src/outputs/03_DL_and_GPU/cifar10_cuda/CIFAR10CNN.pth` | 학습한 CIFAR-10 모델 |

이미지와 데이터셋은 저장소 루트의 `src` 폴더를 사용한다.

## 3. Daily_Assignment03.py 구현 내용

### 3.1. MNIST 모델 학습과 성능 확인

MNIST의 28×28 흑백 손글씨 이미지를 0~9의 숫자로 분류하는 `SimpleCNN`을 사용한다. 모델은 Convolution, ReLU, Max Pooling을 두 차례 적용한 뒤 Fully Connected Layer를 통해 클래스별 점수 10개를 출력한다.

학습은 다음 순서로 진행한다.

1. 이미지와 정답을 CPU 또는 GPU로 이동한다.
2. 이전 Batch의 Gradient를 초기화한다.
3. 순전파로 예측 점수를 구하고 Cross-Entropy Loss를 계산한다.
4. 역전파로 Gradient를 계산하고 Optimizer로 파라미터를 갱신한다.
5. 테스트 데이터로 Loss와 Accuracy를 계산한다.

`train_one_epoch()`는 학습을, `evaluate()`는 평가를 담당한다. 평가할 때는 `model.eval()`과 `torch.no_grad()`를 사용하여 파라미터를 갱신하지 않는다. Epoch 반복과 지표 기록은 `fit()`에서 수행한다.

**16) 모델 성능 확인 TODO**에서는 Epoch별 Train/Test Loss와 Accuracy 그래프를 작성했다. Accuracy는 100을 곱해 백분율로 표시한다. 학습 데이터와 테스트 데이터의 지표를 함께 표시하여 학습이 진행되는 양상과 일반화 성능의 차이를 확인할 수 있도록 했다.

**17) 학습된 모델의 Feature Map 확인 TODO**에서는 테스트 이미지 한 장의 중간 출력을 시각화했다. Conv1의 채널 8개, Conv2의 채널 16개를 표시하고, `conv1 → relu1 → pool1 → conv2 → relu2 → pool2`의 각 층에서 채널 4개씩 비교한다. 색상은 노트북의 요구에 맞춰 `viridis`를 사용했다.

### 3.2. CuPy를 이용한 이미지 잔상 효과

입력 이미지로 `seagull.jpg`를 사용한다. 이미지를 흑백으로 변환하고 1280×720으로 조정한 뒤 픽셀값을 0~1 범위의 `float32` 배열로 만든다.

잔상 효과는 이미지를 오른쪽으로 이동시킨 결과에 가중치를 곱해 누적하는 방식으로 구현했다. 이동 횟수는 기본 32회이며, 이동 거리가 멀수록 가중치가 작아진다. 가중치의 합은 1이 되도록 정규화했다. 이미지 왼쪽 바깥으로 벗어나는 위치는 누적에서 제외한다.

- `motion_trail_np()`: CPU에서 출력 픽셀을 하나씩 순회하며 가중합을 계산한다.
- `motion_trail_cp()`: 이동 거리만 반복하고, 각 이동 단계에서는 배열 슬라이싱과 CuPy 연산으로 전체 영역을 처리한다.

CuPy는 CUDA를 이용해 GPU에서 배열 연산을 수행한다. 이 실습은 같은 잔상 효과를 CPU의 픽셀 반복과 GPU의 배열 연산으로 구현했을 때 처리 시간이 어떻게 달라지는지 비교하기 위한 것이다. 측정된 배속에는 하드웨어 차이뿐 아니라 구현 방식의 차이도 반영된다.

### 3.3. CuPy를 이용한 Convolution2D

`convolution2d()`와 `convolution2d_cp()`는 이미지 영역과 커널의 원소별 곱을 합산하고 Bias를 더한다. 노트북의 CNN 연산 방식에 따라 커널을 뒤집지 않으며, Stride와 Padding을 반영해 출력 크기를 계산한다.

CPU 버전은 출력 위치마다 이미지 영역을 잘라 계산한다. GPU 버전은 커널의 위치만 반복하고, 각 위치에 대응하는 전체 출력 영역을 CuPy 배열 연산으로 계산한다.

비교에는 3×3 세로·가로 Edge 필터를 사용하며 Padding은 1, Bias는 -0.2로 설정했다. CPU와 GPU의 원시 출력이 일치하는지 확인한 뒤 ReLU를 적용한 이미지를 화면에 표시한다.

### 3.4. CPU와 GPU 처리 시간 비교

`benchmark_array()`에서 다음 항목을 각각 측정한다.

| 항목 | 의미 |
| --- | --- |
| CPU 시간 | NumPy 구현의 처리 시간 |
| Upload 시간 | `cp.asarray()`로 데이터를 GPU에 복사하는 시간 |
| GPU 시간 | CuPy 구현의 처리 시간 |
| Download 시간 | `cp.asnumpy()`로 GPU 결과를 CPU에 가져오는 시간 |
| Compute speedup | CPU 시간 ÷ GPU 연산 시간 |
| Total speedup | CPU 시간 ÷ (Upload + GPU 연산 + Download 시간) |

실제 측정 전에 작은 배열로 Warm-up을 수행한다. GPU 연산은 비동기적으로 실행될 수 있으므로 시간 측정 경계에서 `synchronize()`를 호출하여 연산이 끝날 때까지 기다린다. CPU와 GPU 결과는 `np.testing.assert_allclose()`로 허용 오차 내에서 일치하는지 검사한다.

배속은 `8.00x`와 같은 배수로 출력한다. GPU 종류, 이미지 크기, 데이터 전송 시간 등에 영향을 받으므로 특정 배속을 정답으로 가정하지 않는다.

### 3.5. MNIST main_gpu() 구현

`main_gpu()`는 CUDA 사용 가능 여부를 확인한 뒤 `run_training()`에 GPU 장치를 전달한다. `cifar`의 기본값이 `False`이므로 MNIST 데이터셋과 `SimpleCNN`을 사용한다.

CPU와 GPU 모두 같은 모델 구조, Seed, 학습 데이터 순서를 사용한다. 기본 설정은 3 Epoch, 학습 Batch 크기 64, 평가 Batch 크기 1000, Adam Optimizer, 학습률 0.001이다. 학습 후 CPU와 GPU의 전체 시간을 비교한다.

학습 시간에는 각 Epoch의 학습과 평가가 포함되며, Warm-up과 그래프 출력 및 모델 저장 시간은 제외한다.

### 3.6. CIFAR-10 모델 학습과 저장

CIFAR-10은 32×32 RGB 이미지를 10개 클래스로 분류하는 데이터셋이다. `run_training(device, cifar=True)`를 호출하면 MNIST 대신 CIFAR-10의 데이터, 모델, 학습 설정을 사용한다.

학습 데이터에는 `RandomCrop`과 `RandomHorizontalFlip`을 적용하여 데이터의 위치와 방향을 변화시킨다. 학습·테스트 데이터에는 동일한 채널 평균과 표준편차로 정규화를 적용한다.

모델은 다음과 같이 구성했다.

```text
입력: 3×32×32
→ ConvBlock(3, 64)
→ ConvBlock(64, 128)
→ ConvBlock(128, 256)
→ Adaptive Average Pooling
→ Flatten
→ Dropout(0.3)
→ Linear(256, 10)
```

각 ConvBlock은 `Conv2d → BatchNorm2d → ReLU`를 두 번 적용하고 Max Pooling으로 공간 크기를 줄인다.

| 학습 항목 | 설정 |
| --- | --- |
| Epoch | 5 |
| 학습 / 평가 Batch 크기 | 32 / 128 |
| 손실 함수 | CrossEntropyLoss |
| Optimizer | AdamW |
| 학습률 | 0.001 |

학습 후 Loss·Accuracy 그래프를 표시하고 모델을 `.pth`로 저장한다. 저장 파일에는 모델 이름, 모델과 Optimizer의 상태, Epoch, 테스트 정확도, 클래스 이름, 정규화 정보가 포함된다. 이 Python 코드로 새로 저장할 때는 Epoch별 학습 기록인 `history`도 포함한다.

저장한 모델은 다시 불러와 동일한 입력에 대한 출력이 저장 전과 일치하는지 검사한다. CIFAR-10 테스트 이미지 40장에는 실제 클래스, 예측 클래스, 예측 확률을 표시하고 오답을 빨간색으로 구분한다.

## 4. Daily_Assignment04.py 구현 내용

### 4.1. Bounding Box와 기존 작성 코드

노트북에 작성했던 사과 Color Segmentation, Contour 추출, Bounding Box 코드도 포함했다. 빨간색의 HSV 범위를 두 구간으로 나누어 Binary Mask를 만들고 Gaussian Blur, Opening, Closing을 적용한다.

가장 큰 Contour를 선택해 원본 이미지에 표시한 뒤 `cv2.boundingRect()`로 이를 감싸는 사각형을 구한다. 반환된 `(x, y, w, h)`는 `(x, y, x+w, y+h)`로 변환하여 화면에 그린다.

### 4.2. IoU 구현

`convert_bbox_coord()`는 Bounding Box 좌표 형식을 변환하고, `calculate_iou()`는 두 박스의 겹침 정도를 계산한다.

```text
IoU = 교집합 면적 / 합집합 면적
합집합 면적 = 박스 A 면적 + 박스 B 면적 - 교집합 면적
```

교집합의 좌측 상단은 두 박스의 시작 좌표 중 큰 값, 우측 하단은 끝 좌표 중 작은 값으로 구한다. 겹치지 않는 경우 너비나 높이가 음수가 될 수 있으므로 `max(0, 값)`으로 처리한다. 합집합 면적이 0이면 0을 반환하여 0으로 나누는 오류를 방지한다.

`two_apples1.png`부터 `two_apples3.png`까지 빨간 사과와 초록 사과의 Bounding Box를 추출하고 각 IoU를 출력한다.

### 4.3. Confidence Thresholding과 NMS 구현

`prune_low_conf()`는 신뢰도가 임계값 이상인 박스와 해당 점수만 남긴다. 기존 코드에서 점수 목록 전체를 추가하던 부분은 각 박스에 대응하는 `score`를 추가하도록 수정했다.

`apply_nms()`는 다음 순서로 중복 박스를 제거한다.

1. `np.argsort(scores)[::-1]`로 점수가 높은 순서의 인덱스를 구한다.
2. 가장 높은 점수의 박스를 최종 결과에 추가한다.
3. 선택한 박스와 나머지 박스의 IoU를 계산한다.
4. IoU가 임계값보다 큰 박스를 제거한다.
5. 남은 박스에 대해 같은 과정을 반복한다.

Confidence Thresholding 결과를 NMS의 입력으로 전달하며, 결과를 그릴 때도 남은 박스와 점수를 함께 사용한다. 기본 Confidence 임계값은 0.5, IoU 임계값은 0.2이다.

NMS 예시는 추가 이미지 다운로드 없이 `seagull.jpg` 위에 무작위 박스 20개를 생성한다. 직접 구현한 결과와 `cv2.dnn.NMSBoxes()` 결과를 비교하며, 원본 박스·Confidence 필터링·직접 구현한 NMS·OpenCV NMS를 나란히 표시한다.

## 5. 실행 방법

### 실행 환경

| 파일 | 필요한 라이브러리 및 장치 |
| --- | --- |
| Assignment03 | NumPy, Matplotlib, Pillow, PyTorch, torchvision |
| Assignment03 GPU 학습 | NVIDIA GPU, 드라이버, CUDA를 지원하는 PyTorch |
| Assignment03 CuPy 실습 | NVIDIA CUDA 환경과 호환되는 CuPy |
| Assignment04 | NumPy, Matplotlib, OpenCV |


저장소 루트에서 다음 명령으로 실행한다.

```powershell
python 261025/Daily_Assignment03.py
python 261025/Daily_Assignment04.py
```


실행 순서는 MNIST CPU 학습, MNIST GPU 학습, CuPy 비교, CIFAR-10 GPU 학습이다.  데이터셋은 `src/datasets`에서 불러오며, 없으면 `download=True`에 의해 다운로드한다.

CIFAR-10을 이미 학습했다면 `RUN_CIFAR10 = False`로 설정해 재학습을 생략할 수 있다. 이 설정은 해당 단계 전체를 건너뛰는 것이며 기존 모델의 추론만 실행하는 기능은 아니다. `True`로 실행하면 새로 학습하고 같은 경로의 모델 파일을 덮어쓴다.

그래프와 Feature Map은 모든 실행 단계가 끝난 뒤 `plt.show()`로 표시한다. 그림 파일은 저장하지 않으며, 모델 파일만 저장한다. Assignment04는 이미지 표시와 계산 결과 출력만 수행한다.

## 6. 모델 저장 위치

모델은 저장소 루트 기준 다음 위치에 저장한다.

```text
src/outputs/03_DL_and_GPU/
├── mnist_cpu/SimpleCNN.pth
├── mnist_cuda/SimpleCNN.pth
└── cifar10_cuda/CIFAR10CNN.pth
```
