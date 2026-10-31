"""03_DL-and-GPU.ipynb의 TODO 실습."""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "src/outputs/03_DL_and_GPU"
DATA_ROOT = ROOT / "src/datasets"
MNIST_EPOCHS = 3
CIFAR10_EPOCHS = 5

RUN_MNIST_GPU = True
RUN_CUPY = True
RUN_CIFAR10 = True

IMAGE_PATH = ROOT / "src/images/seagull.jpg"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TRAIL_LENGTH = 32
LAYER_NAMES = ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


class SimpleCNN(nn.Module):
    """MNIST 모델."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        features = {}
        features["conv1"] = self.conv1(x)
        features["relu1"] = F.relu(features["conv1"])
        features["pool1"] = self.pool(features["relu1"])
        features["conv2"] = self.conv2(features["pool1"])
        features["relu2"] = F.relu(features["conv2"])
        features["pool2"] = self.pool(features["relu2"])
        x = torch.flatten(features["pool2"], start_dim=1)
        logits = self.fc2(F.relu(self.fc1(x)))
        return (logits, features) if return_features else logits


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 128), ConvBlock(128, 256)
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, x):
        x = self.global_average_pool(self.features(x))
        return self.classifier(torch.flatten(x, start_dim=1))


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = correct_count = sample_count = 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct_count += (logits.argmax(1) == labels).sum().item()
        sample_count += images.size(0)
    if not sample_count:
        raise ValueError("학습 데이터가 비어 있습니다.")
    return total_loss / sample_count, correct_count / sample_count


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = correct_count = sample_count = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct_count += (logits.argmax(1) == labels).sum().item()
        sample_count += images.size(0)
    if not sample_count:
        raise ValueError("평가 데이터가 비어 있습니다.")
    return total_loss / sample_count, correct_count / sample_count


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다."
            "PyTorch를 확인하세요."
        )
    return torch.device("cuda")


# ============================================================
# 16) 모델 성능 확인 TODO
# ============================================================
def plot_history(history):
    epoch_axis = range(1, len(history["train_loss"]) + 1)

    # TODO: train_loss_history 그래프 생성
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_axis, history["train_loss"], marker="o", label="Train loss")

    # TODO: test_loss_history 그래프 생성
    ax.plot(epoch_axis, history["test_loss"], marker="o", label="Test loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="CNN Loss")
    ax.set_xticks(list(epoch_axis))
    ax.grid()
    ax.legend()
    fig.tight_layout()

    # TODO: train_accuracy_history 그래프 생성 (백분율)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_axis, np.array(history["train_accuracy"]) * 100,
            marker="o", label="Train accuracy")

    # TODO: test_accuracy_history 그래프 생성 (백분율)
    ax.plot(epoch_axis, np.array(history["test_accuracy"]) * 100,
            marker="o", label="Test accuracy")
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)", title="CNN Accuracy")
    ax.set_xticks(list(epoch_axis))
    ax.grid()
    ax.legend()
    fig.tight_layout()


def show_feature_maps(feature_tensor, title, max_maps=8, cmap="viridis"):
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")
    maps = feature_tensor[0].detach().cpu()
    count = min(max_maps, maps.shape[0])
    fig, axes = plt.subplots((count + 3) // 4, 4, figsize=(12, 3 * ((count + 3) // 4)), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index < count:
            ax.imshow(maps[index], cmap=cmap)
            ax.set_title(f"Channel {index}")
        ax.axis("off")
    fig.suptitle(title)
    return fig


# ============================================================
# 17) 학습된 모델의 Feature Map 확인 TODO
# ============================================================
@torch.no_grad()
def plot_trained_features(model, dataset, device):
    """F-17: Conv1 8개, Conv2 16개, 6개 Layer에서 4개씩 비교."""
    # TODO: 이미지 선택
    image, true_label = dataset[0]
    # TODO: 역전파(파라미터 수정) 없이 순전파 수행 (@torch.no_grad() 적용)
    model.eval()
    logits, features = model(image.unsqueeze(0).to(device), return_features=True)
    prediction = logits.argmax(1).item()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image.squeeze(0), cmap="gray")
    ax.set_title(f"True: {true_label}, Prediction: {prediction}")
    ax.axis("off")
    fig.tight_layout()
    # TODO: Conv1 Feature Maps 8개 시각화 (cmap="viridis")
    fig = show_feature_maps(features["conv1"], "Trained conv1 Feature Maps", 8)
    fig.tight_layout()

    # TODO: Conv2 Feature Maps 16개 시각화 (cmap="viridis")
    fig = show_feature_maps(features["conv2"], "Trained conv2 Feature Maps", 16)
    fig.tight_layout()

    # TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
    fig, axes = plt.subplots(6, 4, figsize=(12, 16))
    for row, name in enumerate(LAYER_NAMES):
        maps = features[name][0].cpu()
        for column in range(4):
            axes[row, column].imshow(maps[column], cmap="viridis")
            axes[row, column].set_title(f"{name}, channel {column}")
            axes[row, column].axis("off")
    fig.tight_layout()


# ============================================================
# 3) GPU 기반 MNIST 데이터셋 학습 TODO (I 섹션)
# 12) CNN 모델 학습 TODO (J 섹션: CIFAR-10)
# ============================================================
def fit(model, train_loader, test_loader, optimizer, device, epochs):
    criterion = nn.CrossEntropyLoss()
    history = {key: [] for key in ("train_loss", "train_accuracy", "test_loss", "test_accuracy")}
    # Warm-up은 학습 파라미터/BatchNorm 통계를 변경하지 않는다.
    if device.type == "cuda":
        sample, _ = next(iter(test_loader))
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                model(sample[: min(64, len(sample))].to(device))
    synchronize(device)
    # 12) CNN 모델 학습 TODO: 전체 학습 시간 측정 시작
    start = time.perf_counter()
    for epoch in range(epochs):
        synchronize(device)
        epoch_start = time.perf_counter()
        # 12) CNN 모델 학습 TODO: 각 Epoch에서 학습 및 평가 진행, 지표 기록
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        for key, value in zip(history, (train_loss, train_accuracy, test_loss, test_accuracy)):
            history[key].append(value)
        synchronize(device)
        # 12) CNN 모델 학습 TODO: 각 Epoch 시간 계산 및 출력
        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | Time: {elapsed:.2f}s",
            flush=True,
        )
    synchronize(device)
    # 12) CNN 모델 학습 TODO: 전체 학습 시간 계산 및 출력
    total_time = time.perf_counter() - start
    print(f"Total {device.type.upper()} training/evaluation time: {total_time:.2f}s")
    return history, total_time


def make_loaders(cifar, device):
    if cifar:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = test_transform = transforms.ToTensor()
    dataset_type = datasets.CIFAR10 if cifar else datasets.MNIST
    train = dataset_type(str(DATA_ROOT), train=True, transform=train_transform, download=True)
    test = dataset_type(str(DATA_ROOT), train=False, transform=test_transform, download=True)
    kwargs = dict(num_workers=0, pin_memory=device.type == "cuda")
    # 고정 generator로 CPU/GPU 비교에서 같은 순서의 학습 Batch를 사용한다.
    train_loader = DataLoader(
        train, batch_size=32 if cifar else 64, shuffle=True,
        generator=torch.Generator().manual_seed(42), **kwargs,
    )
    test_loader = DataLoader(test, batch_size=128 if cifar else 1000, shuffle=False, **kwargs)
    return train_loader, test_loader


@torch.no_grad()
def plot_cifar_predictions(model, dataset, device):
    model.eval()
    fig, axes = plt.subplots(4, 10, figsize=(20, 10))
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= len(dataset):
            continue
        image, label = dataset[index]
        probs = model(image.unsqueeze(0).to(device)).softmax(1)[0]
        predicted = probs.argmax().item()
        ax.imshow((image * std + mean).permute(1, 2, 0).clamp(0, 1))
        ax.set_title(
            f"True: {CIFAR10_CLASSES[label]}\nPred: {CIFAR10_CLASSES[predicted]}\n"
            f"Conf: {probs[predicted].item() * 100:.1f}%",
            color="red" if predicted != label else "black", fontsize=9,
        )
    fig.tight_layout()


def run_training(device, cifar=False):
    seed_everything()
    train_loader, test_loader = make_loaders(cifar, device)
    model_type = CIFAR10CNN if cifar else SimpleCNN
    model = model_type().to(device)
    optimizer_type = torch.optim.AdamW if cifar else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=0.001)
    epochs = CIFAR10_EPOCHS if cifar else MNIST_EPOCHS
    name = "cifar10" if cifar else "mnist"
    directory = OUTPUT / f"{name}_{device.type}"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{name.upper()} | Device: {device} | Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")
    history, elapsed = fit(model, train_loader, test_loader, optimizer, device, epochs)
    plot_history(history)
    if not cifar:
        plot_trained_features(model, test_loader.dataset, device)
    # F-18 / J-14: 원본 모델 파일을 덮어쓰지 않고 별도 결과 폴더에 저장한다.
    checkpoint = {
        "model_name": model_type.__name__, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "epochs": epochs,
        "test_accuracy": history["test_accuracy"][-1], "history": history,
        "class_names": list(CIFAR10_CLASSES) if cifar else [str(i) for i in range(10)],
    }
    if cifar:
        checkpoint.update(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    model_path = directory / f"{model_type.__name__}.pth"
    torch.save(checkpoint, model_path)
    loaded = model_type().to(device)
    saved = torch.load(model_path, map_location=device, weights_only=True)
    loaded.load_state_dict(saved["model_state_dict"])
    loaded.eval()
    model.eval()
    image, _ = test_loader.dataset[0]
    with torch.no_grad():
        batch = image.unsqueeze(0).to(device)
        torch.testing.assert_close(model(batch), loaded(batch))
    if cifar:
        plot_cifar_predictions(loaded, test_loader.dataset, device)
    print(f"Saved and reload verified: {model_path}")
    return elapsed


# ============================================================
# 3) GPU 기반 MNIST 데이터셋 학습 TODO (I 섹션)
# ============================================================
def main_gpu():
    """I-3: GPU 기반 MNIST 학습."""
    return run_training(require_cuda())


# ============================================================
# 4) CuPy를 활용한 배열 연산 GPU 가속 1 – 이미지 잔상 효과 구현 TODO
# TODO: NumPy를 사용하여 잔상 효과 함수 정의
# ============================================================
def motion_trail_np(image, weights):
    """G-4: CPU에서 출력 픽셀을 하나씩 계산하고, 왼쪽 빈 영역은 0으로 처리."""
    # TODO : 출력 배열 생성
    output = np.zeros_like(image, dtype=np.float32)
    for output_y in range(image.shape[0]):
        for output_x in range(image.shape[1]):
            value = 0.0
            # TODO : 입력 x 좌표를 shift하고 유효한 픽셀만 가중합으로 누적
            for shift in range(min(len(weights), output_x + 1)):
                value += image[output_y, output_x - shift] * weights[shift]
            # TODO : 누적한 값을 출력 픽셀에 저장
            output[output_y, output_x] = value
    return output


def get_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy가 없습니다.") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy에서 사용할 CUDA GPU가 없습니다.")
    return cp


# 4) 이미지 잔상 효과 구현 TODO: CuPy를 사용하여 잔상 효과 함수 정의
def motion_trail_cp(image, weights):
    """이동 횟수는 가중치 개수로 결정하여 짧은 warm-up 배열도 지원."""
    cp = get_cupy()
    # TODO : GPU 출력 배열 생성
    output = cp.zeros_like(image, dtype=cp.float32)
    for shift in range(min(len(weights), image.shape[1])):
        # TODO : 입력/출력 영역을 slicing하여 오른쪽으로 이동
        input_region = image[:, :image.shape[1] - shift]
        output_region = output[:, shift:]
        # TODO : 해당 영역 전체를 GPU 병렬 연산으로 누적
        output_region += input_region * weights[shift]
    return output


def convolution_shape(image, kernel, stride, padding):
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("이미지와 커널은 2차원이어야 합니다.")
    if stride < 1 or padding < 0:
        raise ValueError("stride는 양수, padding은 0 이상이어야 합니다.")
    height = (image.shape[0] + 2 * padding - kernel.shape[0]) // stride + 1
    width = (image.shape[1] + 2 * padding - kernel.shape[1]) // stride + 1
    if min(height, width) < 1:
        raise ValueError("커널이 패딩된 이미지보다 큽니다.")
    return height, width


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    height, width = convolution_shape(image, kernel, stride, padding)
    padded = np.pad(image, padding, mode="constant")
    output = np.zeros((height, width), dtype=np.float32)
    kh, kw = kernel.shape
    for y in range(height):
        for x in range(width):
            region = padded[y * stride:y * stride + kh, x * stride:x * stride + kw]
            output[y, x] = np.sum(region * kernel) + bias
    return output


# ============================================================
# 5) CuPy를 활용한 배열 연산 GPU 가속 2 – Convolution 연산 TODO
# TODO: CuPy를 사용하여 2차원 convolution 함수 정의
# ============================================================
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """G-5: 커널 위치만 반복하고 모든 출력 픽셀은 GPU 배열 연산으로 계산."""
    cp = get_cupy()
    height, width = convolution_shape(image, kernel, stride, padding)
    padded = cp.pad(image, padding, mode="constant")
    output = cp.zeros((height, width), dtype=cp.float32)
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            region = padded[ky:ky + height * stride:stride, kx:kx + width * stride:stride]
            output += region * kernel[ky, kx]
    return output + bias


# 4) 이미지 잔상 효과 구현 / 5) Convolution 연산 TODO: CPU/GPU 실행 및 비교
def benchmark_array(name, image, kernels, cpu_operation, gpu_operation):
    """업로드/연산/다운로드 시간을 분리하고 실제 수치 일치를 검사한다."""
    cp = get_cupy()
    start = time.perf_counter()
    # 5) Convolution 연산 TODO: convolution2d 함수로 CPU 처리 및 시간 계산
    cpu_outputs = [cpu_operation(image, kernel) for kernel in kernels]
    cpu_time = time.perf_counter() - start
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    # 4) 이미지 잔상 효과 구현 TODO: NumPy 배열을 CuPy 배열로 변환
    image_cp = cp.asarray(image)
    kernels_cp = [cp.asarray(kernel) for kernel in kernels]
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - start
    # 5) Convolution 연산 TODO: GPU 처리 파이프라인 중 warm-up
    for kernel in kernels_cp:
        gpu_operation(cp.zeros((64, 64), dtype=cp.float32), kernel)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    # 5) Convolution 연산 TODO: convolution을 GPU로 처리하고 시간 계산
    gpu_outputs = [gpu_operation(image_cp, kernel) for kernel in kernels_cp]
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start
    start = time.perf_counter()
    # 4) 이미지 잔상 효과 구현 TODO: output_cp를 CPU 메모리로 복사
    gpu_outputs_np = [cp.asnumpy(output) for output in gpu_outputs]
    cp.cuda.Stream.null.synchronize()
    download_time = time.perf_counter() - start
    # 5) Convolution 연산 TODO: CPU/GPU 결과 비교 및 출력
    for cpu, gpu in zip(cpu_outputs, gpu_outputs_np):
        np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-5)
    total = upload_time + gpu_time + download_time
    max_error = max(float(np.abs(a - b).max()) for a, b in zip(cpu_outputs, gpu_outputs_np))
    print(f"\n{name}")
    print(f"CPU: {cpu_time:.4f}s")
    print(f"Upload: {upload_time:.4f}s | GPU: {gpu_time:.4f}s | Download: {download_time:.4f}s")
    print(f"Compute speedup: {cpu_time / gpu_time:.2f}x | Total speedup: {cpu_time / total:.2f}x")
    print(f"Max absolute error: {max_error:.8f}")
    fig, axes = plt.subplots(len(kernels), 3, figsize=(15, 5 * len(kernels)), squeeze=False)
    for row, (cpu, gpu) in enumerate(zip(cpu_outputs, gpu_outputs_np)):
        for ax, values, title in zip(axes[row], (image, cpu, gpu), ("Input", "CPU", "GPU")):
            # 5) Convolution 연산 TODO: CPU/GPU 결과에 ReLU 적용
            ax.imshow(np.maximum(values, 0), cmap="gray")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()


def run_cupy():
    get_cupy()  # CPU의 오래 걸리는 픽셀 반복 전에 GPU 환경 확인
    with Image.open(IMAGE_PATH) as source:
        image = np.asarray(source.convert("L").resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BOX), dtype=np.float32) / 255
    weights = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)
    weights /= weights.sum()
    benchmark_array("motion_trail", image, [weights], motion_trail_np, motion_trail_cp)
    dx = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
    dy = dx.T.copy()
    benchmark_array(
        "convolution", image, [dx, dy],
        lambda im, k: convolution2d(im, k, bias=-0.2, padding=1),
        lambda im, k: convolution2d_cp(im, k, bias=-0.2, padding=1),
    )


def main():
    # F-16~18: CPU 학습 후 성능 그래프, Feature Map, 모델 저장/불러오기
    cpu_time = run_training(torch.device("cpu"))

    # I-3: MNIST GPU 학습 및 CPU/GPU 시간 비교
    if RUN_MNIST_GPU:
        gpu_time = main_gpu()
        print(f"CPU: {cpu_time:.2f}s | GPU: {gpu_time:.2f}s | Speedup: {cpu_time / gpu_time:.2f}x")

    # G-4~5: CuPy 잔상 효과 및 Convolution
    if RUN_CUPY:
        run_cupy()

    # J-12: CIFAR-10 GPU 학습
    if RUN_CIFAR10:
        run_training(require_cuda(), cifar=True)

    plt.show()


if __name__ == "__main__":
    main()
