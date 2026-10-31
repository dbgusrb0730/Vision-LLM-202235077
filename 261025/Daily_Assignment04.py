"""04_DL-Object-Detection.ipynb의 Bounding Box, IoU, NMS TODO 실습."""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = ROOT / "src/images"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.2


def read_image(name):
    image = cv2.imread(str(IMAGE_ROOT / name))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_ROOT / name}")
    return image


def draw_bbox(image, box, color, text=""):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(image, text, (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ============================================================
# B. Intersection over Union (IoU) TODO
# ============================================================

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


# TODO: 2) IoU 계산 함수 정의
def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)
    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    intersection_area = w * h
    area_a = max(0, xa_2 - xa_1) * max(0, ya_2 - ya_1)
    area_b = max(0, xb_2 - xb_1) * max(0, yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area

    # TODO: ZeroDivisionError 주의
    return intersection_area / union_area if union_area > 0 else 0.0


# ============================================================
# A. Bounding Box TODO
# ============================================================
def show_apple_bbox():
    apple_bgr = read_image("apple.png")
    apple_rgb = cv2.cvtColor(apple_bgr, cv2.COLOR_BGR2RGB)

    # TODO: Color Segmentation을 통해 Binary Mask 생성
    apple_hsv = cv2.cvtColor(apple_rgb, cv2.COLOR_RGB2HSV)
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])

    # TODO: Gaussian Blur 적용
    apple_blur = cv2.GaussianBlur(apple_hsv, (15, 15), 0)

    # TODO: Binary Mask 생성 (빨강의 두 HSV 범위를 합침)
    apple_mask = cv2.inRange(apple_blur, red_lower1, red_upper1)
    apple_mask |= cv2.inRange(apple_blur, red_lower2, red_upper2)

    # TODO: 필요하다면 Morphological Operation 적용
    kernel = np.ones((5, 5), dtype=np.uint8)
    apple_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_OPEN, kernel)
    apple_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_CLOSE, kernel)

    # TODO: Binary Mask를 사용하여 Contour 추출
    contours, _ = cv2.findContours(apple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("사과 이미지에서 빨간색 Contour를 찾지 못했습니다.")

    # TODO: 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # TODO: Contour를 추출하여 원본 이미지 위에 overlay
    largest_mask = np.zeros_like(apple_mask)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, cv2.FILLED)
    mask_color = np.zeros_like(apple_rgb)
    mask_color[:, :, 1] = largest_mask
    overlay = cv2.addWeighted(apple_rgb, 1.0, mask_color, 0.35, 0)
    cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)

    # TODO: cv2.rectangle() 함수에 사용할 수 있는 좌표로 변환
    bbox_xywh = cv2.boundingRect(largest_contour)
    bbox_xyxy = convert_bbox_coord(bbox_xywh)
    bbox_image = apple_rgb.copy()
    draw_bbox(bbox_image, bbox_xyxy, (0, 255, 0), "Apple")
    print(f"Apple BBox: xywh={bbox_xywh}, xyxy={bbox_xyxy}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, image, title in zip(axes, (apple_rgb, apple_mask, overlay, bbox_image),
                                 ("Original", "Binary Mask", "Contour Overlay", "Bounding Box")):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()


# B. Intersection over Union (IoU): 노트북의 사과 이미지 3장으로 결과 확인
def get_contour(image_bgr):
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([25, 10, 30]), np.array([100, 255, 255]))
    contours = []
    for mask in (red_mask, green_mask):
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            raise ValueError("빨간 사과 또는 초록 사과의 Contour를 찾지 못했습니다.")
        contours.append(max(found, key=cv2.contourArea))
    return contours[0], contours[1]


def show_iou():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for index, ax in enumerate(axes, start=1):
        image_bgr = read_image(f"two_apples{index}.png")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        red_contour, green_contour = get_contour(image_bgr)
        red_box = convert_bbox_coord(cv2.boundingRect(red_contour))
        green_box = convert_bbox_coord(cv2.boundingRect(green_contour))
        iou = calculate_iou(red_box, green_box)
        draw_bbox(image_rgb, red_box, (255, 0, 0), "Red Apple")
        draw_bbox(image_rgb, green_box, (0, 255, 0), "Green Apple")
        print(f"two_apples{index}: IoU = {iou:.4f}")
        ax.imshow(image_rgb)
        ax.set_title(f"IoU: {iou:.4f}")
        ax.axis("off")
    fig.tight_layout()


# ============================================================
# C. Non-Maximum Suppression (NMS) TODO
# ============================================================

# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거
def prune_low_conf(boxes, scores, conf_threshold=0.5):
    pruned_boxes = []
    pruned_scores = []
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            # 전체 scores 목록이 아니라 현재 box의 score를 저장한다.
            pruned_scores.append(score)
    return pruned_boxes, pruned_scores


# TODO: NMS 함수 정의
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환
def apply_nms(boxes, scores, iou_threshold=0.5):
    nms_boxes = []
    nms_scores = []
    # 원본 scores의 인덱스를 구한 뒤 뒤집어 신뢰도가 높은 순서로 선택한다.
    indices = np.argsort(scores)[::-1]
    while len(indices) > 0:
        current = int(indices[0])
        nms_boxes.append(boxes[current])
        nms_scores.append(scores[current])
        remaining = []
        for index in indices[1:]:
            if calculate_iou(boxes[current], boxes[index]) <= iou_threshold:
                remaining.append(int(index))
        indices = np.array(remaining, dtype=int)
    return nms_boxes, nms_scores


# C. NMS: OpenCV 내장 함수로 직접 구현한 결과와 비교
def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    boxes, scores = prune_low_conf(boxes, scores, conf_threshold)
    if not boxes:
        return [], []
    boxes_xywh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
    # Confidence Thresholding은 위에서 이미 적용했으므로 여기서는 NMS만 수행한다.
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, iou_threshold)
    indices = np.asarray(indices).reshape(-1).tolist()
    return [boxes[i] for i in indices], [scores[i] for i in indices]


def show_nms():
    random.seed(42)
    # 추가 다운로드 없이 저장소의 이미지 위에 무작위 BBox를 생성한다.
    image = cv2.cvtColor(read_image("seagull.jpg"), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    boxes = []
    scores = []
    for _ in range(20):
        w = random.randint(max(1, width // 5), max(1, width // 2))
        h = random.randint(max(1, height // 3), max(1, height * 2 // 3))
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        boxes.append((x, y, x + w, y + h))
        scores.append(min(1.0, max(0.0, random.gauss(0.6, 0.1))))

    # Confidence Thresholding 결과를 NMS 입력으로 전달한다.
    pruned_boxes, pruned_scores = prune_low_conf(boxes, scores, CONF_THRESHOLD)
    nms_boxes, nms_scores = apply_nms(pruned_boxes, pruned_scores, IOU_THRESHOLD)
    cv2_boxes, cv2_scores = apply_nms_cv2(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    print(f"NMS: original={len(boxes)}, confidence filter={len(pruned_boxes)}, "
          f"custom={len(nms_boxes)}, OpenCV={len(cv2_boxes)}")
    print(f"Custom/OpenCV results match: {nms_boxes == cv2_boxes and nms_scores == cv2_scores}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    stages = ((boxes, scores), (pruned_boxes, pruned_scores),
              (nms_boxes, nms_scores), (cv2_boxes, cv2_scores))
    titles = ("Original Boxes", "Confidence Thresholding", "Custom NMS", "OpenCV NMS")
    for ax, (stage_boxes, stage_scores), title in zip(axes, stages, titles):
        result = image.copy()
        for box, score in zip(stage_boxes, stage_scores):
            draw_bbox(result, box, (0, 255, 0), f"{score:.2f}")
        ax.imshow(result)
        ax.set_title(f"{title}: {len(stage_boxes)}")
        ax.axis("off")
    fig.tight_layout()


def main():
    show_apple_bbox()
    show_iou()
    show_nms()
    plt.show()


if __name__ == "__main__":
    main()
