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
    "わ", "ゐ", "ゑ", "を",
    "ん", "ゝ"
]

img_size = 28
samples_per_class = 1000
font_path = "./font/ipag.ttf"  # Confirmed working
font_size_range = (28, 31)  # Reduced to avoid clipping

all_images, all_labels = [], []

for idx, char in enumerate(hiragana_chars):
    print(f"Generating for: {char} (index {idx})")
    for i in range(samples_per_class):
        # Create blank image
        img = Image.new("L", (img_size, img_size), color=0)  # Black background
        draw = ImageDraw.Draw(img)
        font_size = random.randint(*font_size_range)
        font = ImageFont.truetype(font_path, font_size)

        # Calculate text size and center
        bbox = font.getbbox(char)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max((img_size - w) // 16, 0)
        y = max((img_size - h) // 16, 0)

        # Draw text (black on white)
        draw.text((x, y), char, font=font, fill=255)

        # Convert to NumPy
        img = np.array(img)

        a_num = random.uniform(0.01, 0.02)
        s_num = random.uniform(0.005, 0.01)

        # Apply elastic deformation
        alpha = img_size * a_num
        sigma = img_size * s_num
        random_state = np.random.RandomState(None)
        dx = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (15,15), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (15,15), sigma) * alpha
        x_coords, y_coords = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        # Apply perspective warp
        delta = 4
        pts1 = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])
        pts2 = np.float32([[random.randint(0, delta), random.randint(0, delta)],
                           [img_size - random.randint(0, delta), random.randint(0, delta)],
                           [random.randint(0, delta), img_size - random.randint(0, delta)],
                           [img_size - random.randint(0, delta), img_size - random.randint(0, delta)]])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M_persp, (img_size, img_size), borderValue=0)

        img = cv2.GaussianBlur(img, (3, 3), 0)

        all_images.append(img)
        all_labels.append(idx)


# Convert to numpy arrays
all_images = np.array(all_images, dtype=np.uint8)
all_labels = np.array(all_labels, dtype=np.int32)

# Shuffle before splitting
indices = np.arange(len(all_images))
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

# Split into train/test
split_idx = int(len(all_images) * 0.8)
train_images, test_images = all_images[:split_idx], all_images[split_idx:]
train_labels, test_labels = all_labels[:split_idx], all_labels[split_idx:]

# Save datasets
os.makedirs("./data/hiragana_final", exist_ok=True)
np.savez("./data/hiragana_final/hiragana-train-imgs.npz", train_images)
np.savez("./data/hiragana_final/hiragana-train-labels.npz", train_labels)
np.savez("./data/hiragana_final/hiragana-test-imgs.npz", test_images)
np.savez("./data/hiragana_final/hiragana-test-labels.npz", test_labels)

print(f"✅ Dataset created: Train {train_images.shape}, Test {test_images.shape}, Classes: {len(hiragana_chars)}")

data = np.load("./data/hiragana_final/hiragana-train-imgs.npz")
train_images = data["arr_0"]  # Default key if you didn't name it

import matplotlib.pyplot as plt
plt.imshow(train_images[0], cmap="gray")
plt.show()

