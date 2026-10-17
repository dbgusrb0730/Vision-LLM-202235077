import cv2
import numpy as np
import matplotlib.pyplot as plt


basketball = cv2.imread("../src/images/basketball_crop.jpg")
if basketball is None:
    raise FileNotFoundError("Check ../src/images/basketball_crop.jpg")
basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(basketball_lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_enhanced, a, b))
basketball_enhanced_rgb = cv2.cvtColor(
    basketball_enhanced_lab, cv2.COLOR_LAB2RGB
)

lower_basketball_lab = np.array([70, 130, 70], dtype=np.uint8)
upper_basketball_lab = np.array([180, 230, 185], dtype=np.uint8)
mask_enhanced = cv2.inRange(
    basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
final_ball_mask = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(
    final_ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

largest_mask = np.zeros_like(final_ball_mask)
final_img = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(largest_mask, [largest_contour], -1, 255, cv2.FILLED)

    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 0] = largest_mask
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, 0.15, 0)
    cv2.drawContours(final_img, [largest_contour], -1, (255, 0, 0), 2)
else:
    print("No contours found in the final ball mask.")

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1), plt.imshow(basketball_rgb), plt.title("Original")
plt.subplot(2, 3, 2), plt.imshow(basketball_enhanced_rgb), plt.title("LAB + CLAHE")
plt.subplot(2, 3, 3), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Binary Mask")
plt.subplot(2, 3, 4), plt.imshow(final_ball_mask, cmap="gray"), plt.title("Opening + Closing")
plt.subplot(2, 3, 5), plt.imshow(largest_mask, cmap="gray"), plt.title("Selected Region")
plt.subplot(2, 3, 6), plt.imshow(final_img), plt.title("Selected Contour")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
