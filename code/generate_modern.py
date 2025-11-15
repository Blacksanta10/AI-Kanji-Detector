from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import random

# Hiragana characters
hiragana_chars = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を",
    "ん"
]

img_size = 28
samples_per_class = 6000
# Example: reserve some fonts for validation (unseen during training)
all_fonts = [
    "./font/ipag.ttf",
    "./font/KleeOne-Regular.ttf",
    "./font/NotoSansJP-Regular.ttf",
    "./font/SlacksideOne-Regular.ttf",
    "./font/TekitouPoem.ttf",
    "./font/ZenKurenaido-Regular.ttf",
    "./font/Kaorigel-Z9YK.ttf",
    "./font/Yomogi-Regular.ttf",
    "./font/HinaMincho-Regular.ttf",
    "./font/GL-CurulMinamoto.ttf",
    "./font/AoyagiSosekiFont2.otf",
    "./font/KouzanMouhituFontOTF.otf",
]

random.seed(42)
random.shuffle(all_fonts)
val_ratio = 0.25
split_idx = int(len(all_fonts) * (1 - val_ratio))
train_fonts = all_fonts[:split_idx]
val_fonts   = all_fonts[split_idx:]

def pick_font(split="train"):
    fonts = train_fonts if split == "train" else val_fonts
    return random.choice(fonts)

font_size_range = (64, 80)  # Reduced to avoid clipping

train_ratio = 0.8
train_spc = int(samples_per_class * train_ratio)  # samples per class for train
val_spc   = samples_per_class - train_spc         # samples per class for val

train_images, train_labels = [], []
val_images,   val_labels   = [], []

for idx, char in enumerate(hiragana_chars):
    print(f"Generating for: {char} (index {idx})")
    # ---------- TRAIN SAMPLES (train-only fonts) ----------
    for i in range(train_spc):
        canvas_size = 64
        img = Image.new("L", (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(img)

        font_size = random.randint(*font_size_range)
        font_choice = pick_font(split="train")
        if os.path.basename(font_choice) == "SlacksideOne-Regular.ttf":
            font_size = int(font_size * 1.4)
        font = ImageFont.truetype(font_choice, font_size)

        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top
        x = (canvas_size - w) // 2 - left
        y = (canvas_size - h) // 2 - top
        draw.text((x, y), char, font=font, fill=255)

        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img = np.array(img)

        # elastic + perspective augmentations
        a_num = random.uniform(0.01, 0.02)
        s_num = random.uniform(0.005, 0.01)
        alpha = img_size * a_num
        sigma = img_size * s_num
        random_state = np.random.RandomState(None)

        dx = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (15, 15), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (15, 15), sigma) * alpha
        x_coords, y_coords = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # perspective warp
        delta = 5
        pts1 = np.float32([[0,0],[img_size,0],[0,img_size],[img_size,img_size]])
        pts2 = np.float32([
            [random.randint(0, delta),                random.randint(0, delta)],
            [img_size - random.randint(0, delta),     random.randint(0, delta)],
            [random.randint(0, delta),                img_size - random.randint(0, delta)],
            [img_size - random.randint(0, delta),     img_size - random.randint(0, delta)]
        ])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M_persp, (img_size, img_size), borderValue=0)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        train_images.append(img.astype(np.uint8))
        train_labels.append(idx)

    # ---------- VAL/TEST SAMPLES (val-only fonts) ----------
    for i in range(val_spc):
        canvas_size = 64
        img = Image.new("L", (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(img)

        font_size = random.randint(*font_size_range)
        font_choice = pick_font(split="val")  # NOTE: val fonts only
        if os.path.basename(font_choice) == "SlacksideOne-Regular.ttf":
            font_size = int(font_size * 1.4)
        font = ImageFont.truetype(font_choice, font_size)

        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top
        x = (canvas_size - w) // 2 - left
        y = (canvas_size - h) // 2 - top
        draw.text((x, y), char, font=font, fill=255)

        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img = np.array(img)

        # same elastic + perspective
        a_num = random.uniform(0.01, 0.02)
        s_num = random.uniform(0.005, 0.01)
        alpha = img_size * a_num
        sigma = img_size * s_num
        random_state = np.random.RandomState(None)

        dx = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (15, 15), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (15, 15), sigma) * alpha
        x_coords, y_coords = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        delta = 5
        pts1 = np.float32([[0,0],[img_size,0],[0,img_size],[img_size,img_size]])
        pts2 = np.float32([
            [random.randint(0, delta),                random.randint(0, delta)],
            [img_size - random.randint(0, delta),     random.randint(0, delta)],
            [random.randint(0, delta),                img_size - random.randint(0, delta)],
            [img_size - random.randint(0, delta),     img_size - random.randint(0, delta)]
        ])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M_persp, (img_size, img_size), borderValue=0)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        val_images.append(img.astype(np.uint8))
        val_labels.append(idx)

# Convert to numpy arrays
train_images = np.array(train_images, dtype=np.uint8)
train_labels = np.array(train_labels, dtype=np.int32)
val_images   = np.array(val_images, dtype=np.uint8)
val_labels   = np.array(val_labels, dtype=np.int32)

# Shuffle within each split
train_idx = np.arange(len(train_images)); np.random.shuffle(train_idx)
val_idx   = np.arange(len(val_images));   np.random.shuffle(val_idx)

train_images = train_images[train_idx]; train_labels = train_labels[train_idx]
val_images   = val_images[val_idx];     val_labels   = val_labels[val_idx]

# Save datasets
os.makedirs("./data/hiragana_final", exist_ok=True)
np.savez("./data/hiragana_final/hiragana-train-imgs.npz", train_images)
np.savez("./data/hiragana_final/hiragana-train-labels.npz", train_labels)
np.savez("./data/hiragana_final/hiragana-test-imgs.npz", val_images)
np.savez("./data/hiragana_final/hiragana-test-labels.npz", val_labels)

print(f"✅ Dataset created: Train {train_images.shape}, Test {val_images.shape}, Classes: {len(hiragana_chars)}")

data = np.load("./data/hiragana_final/hiragana-train-imgs.npz")
train_images = data["arr_0"]  # Default key if you didn't name it

import matplotlib.pyplot as plt
plt.imshow(train_images[0], cmap="gray")
plt.show()

