"""
generate_dataset.py
────────────────────────────────────────────────
Sinh ảnh dataset từ tất cả ảnh trong 1 thư mục.
Áp dụng tổ hợp nhiều loại augmentation:
  • Góc nhìn / phối cảnh (perspective)
  • Xoay (rotate)
  • Lật (flip)
  • Độ sáng / tương phản / bão hòa (HSV)
  • Làm mờ (Gaussian / Motion blur)
  • Nhiễu (Gaussian noise)
  • Crop & zoom
  • Đổ bóng vùng (shadow patch)
  • Nhiệt độ màu (warm / cool tone)
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path

# ============================================================
#  CẤU HÌNH
# ============================================================
INPUT_DIR        = "data 13 vị vua/LĂNG VUA TỰ ĐỨC"         # Thư mục ảnh gốc
OUTPUT_DIR       = "data 13 vị vua/LĂNG VUA TỰ ĐỨC_dataset" # Thư mục lưu kết quả
NUM_PER_IMAGE    = 67                # Số ảnh sinh ra từ mỗi ảnh gốc
OUTPUT_SIZE      = (640, 640)         # W × H ảnh đầu ra
JPEG_QUALITY     = 93                 # Chất lượng lưu ảnh
 
# Ngưỡng augmentation
PERSPECTIVE_STR  = 0.25    # 0 = không méo, 0.4 = méo nhiều
ROTATE_RANGE     = (-30, 30)
FLIP_PROB        = 0.4
BRIGHTNESS_RANGE = (0.45, 1.55)
CONTRAST_RANGE   = (0.7, 1.4)
SAT_RANGE        = (0.6, 1.5)
BLUR_PROB        = 0.4
MOTION_BLUR_PROB = 0.2      # blur kiểu rung tay
NOISE_PROB       = 0.35
SHADOW_PROB      = 0.3      # đổ bóng cục bộ
COLOR_SHIFT_PROB = 0.4      # dịch nhiệt độ màu warm/cool
CROP_PROB        = 0.4      # crop ngẫu nhiên rồi resize lại
# ============================================================


# ─── Các hàm augmentation ────────────────────────────────────

def aug_perspective(img, strength):
    h, w = img.shape[:2]
    s = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    delta = np.float32([
        [random.uniform(-s, s) * w, random.uniform(-s, s) * h],
        [random.uniform(-s, s) * w, random.uniform(-s, s) * h],
        [random.uniform(-s, s) * w, random.uniform(-s, s) * h],
        [random.uniform(-s, s) * w, random.uniform(-s, s) * h],
    ])
    dst = src + delta
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def aug_rotate(img, angle_range):
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def aug_flip(img, prob):
    if random.random() < prob:
        return cv2.flip(img, 1)
    return img


def aug_hsv(img, brightness_range, contrast_range, sat_range):
    """Điều chỉnh Hue-Saturation-Value một cách độc lập"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(*sat_range), 0, 255)
    # Value (brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(*brightness_range), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Contrast (multiply around mid)
    alpha = random.uniform(*contrast_range)
    img = np.clip(128 + alpha * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)
    return img


def aug_gaussian_blur(img, prob, max_k=7):
    if random.random() < prob:
        k = random.choice([3, 5, 7][:max_k // 2 + 1])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def aug_motion_blur(img, prob):
    """Giả lập rung tay khi chụp"""
    if random.random() < prob:
        size = random.choice([5, 7, 9, 11])
        angle = random.uniform(0, 360)
        k = np.zeros((size, size))
        k[size // 2, :] = 1
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
        k = cv2.warpAffine(k, M, (size, size))
        k = k / k.sum()
        return cv2.filter2D(img, -1, k)
    return img


def aug_noise(img, prob):
    if random.random() < prob:
        std = random.uniform(5, 25)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def aug_shadow(img, prob):
    """Đổ bóng hình chữ nhật ngẫu nhiên lên một góc ảnh"""
    if random.random() < prob:
        h, w = img.shape[:2]
        x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
        x2, y2 = x1 + random.randint(w // 4, w // 2), y1 + random.randint(h // 4, h // 2)
        x2, y2 = min(x2, w), min(y2, h)
        factor = random.uniform(0.35, 0.65)
        shadow = img.copy()
        shadow[y1:y2, x1:x2] = (shadow[y1:y2, x1:x2] * factor).astype(np.uint8)
        alpha = random.uniform(0.5, 0.85)
        return cv2.addWeighted(img, 1 - alpha, shadow, alpha, 0)
    return img


def aug_color_shift(img, prob):
    """Dịch chuyển nhiệt độ màu: warm (+R-B) hoặc cool (+B-R)"""
    if random.random() < prob:
        b, g, r = cv2.split(img.astype(np.float32))
        if random.random() < 0.5:  # warm
            r = np.clip(r + random.uniform(10, 40), 0, 255)
            b = np.clip(b - random.uniform(5, 25), 0, 255)
        else:  # cool
            b = np.clip(b + random.uniform(10, 40), 0, 255)
            r = np.clip(r - random.uniform(5, 25), 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)
    return img


def aug_crop_zoom(img, prob):
    """Crop ngẫu nhiên rồi resize lại về kích thước gốc"""
    if random.random() < prob:
        h, w = img.shape[:2]
        crop_ratio = random.uniform(0.65, 0.92)
        cw, ch = int(w * crop_ratio), int(h * crop_ratio)
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        img = img[y:y+ch, x:x+cw]
        img = cv2.resize(img, (w, h))
    return img


def augment_one(src):
    """Pipeline đầy đủ — thứ tự quan trọng"""
    img = src.copy()

    img = aug_perspective(img, PERSPECTIVE_STR)
    img = aug_rotate(img, ROTATE_RANGE)
    img = aug_flip(img, FLIP_PROB)
    img = aug_crop_zoom(img, CROP_PROB)
    img = aug_hsv(img, BRIGHTNESS_RANGE, CONTRAST_RANGE, SAT_RANGE)
    img = aug_color_shift(img, COLOR_SHIFT_PROB)
    img = aug_shadow(img, SHADOW_PROB)
    img = aug_gaussian_blur(img, BLUR_PROB)
    img = aug_motion_blur(img, MOTION_BLUR_PROB)
    img = aug_noise(img, NOISE_PROB)

    img = cv2.resize(img, OUTPUT_SIZE)
    return img


# ─── Chạy chính ──────────────────────────────────────────────

def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    sources = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in ext]

    if not sources:
        print(f"❌ Không có ảnh trong '{INPUT_DIR}'")
        return

    total_plan = len(sources) * NUM_PER_IMAGE
    print(f"📂 {len(sources)} ảnh gốc × {NUM_PER_IMAGE} biến thể = {total_plan} ảnh đầu ra")
    print(f"📁 Lưu vào: '{OUTPUT_DIR}/'\n")

    total_saved = 0

    for src_path in sorted(sources):
        src = cv2.imread(str(src_path))
        if src is None:
            print(f"  ⚠️  Bỏ qua (đọc lỗi): {src_path.name}")
            continue

        base = src_path.stem
        print(f"  🖼️  [{base}] {src.shape[1]}×{src.shape[0]}px → sinh {NUM_PER_IMAGE} ảnh...")

        for i in range(NUM_PER_IMAGE):
            result = augment_one(src)
            fname  = f"{base}_aug{i+1:04d}.jpg"
            out    = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(out, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            total_saved += 1

        print(f"     ✅ Xong {NUM_PER_IMAGE} ảnh")

    print(f"\n🎉 Hoàn tất! Tổng {total_saved} ảnh → '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    generate()
