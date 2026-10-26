# TODO 실습 실행

```powershell
python 03_DL_and_GPU_todos.py
```

기본 실행은 MNIST CPU 학습(3 Epoch), 성능 그래프, Feature Map, 모델 저장 및 불러오기입니다. 학습과 저장이 끝나면 그래프 창을 화면에 표시합니다.

GPU 실습은 Python 파일 상단에서 실행할 항목을 `True`로 바꾸세요.

- `RUN_MNIST_GPU`: MNIST GPU 학습 및 CPU/GPU 시간 비교
- `RUN_CUPY`: CuPy 잔상 효과 및 Convolution 비교
- `RUN_CIFAR10`: CIFAR-10 GPU 학습(5 Epoch)

노트북이 NVIDIA GPU가 아니어서 CPU 환경에서 검증했습니다. 
