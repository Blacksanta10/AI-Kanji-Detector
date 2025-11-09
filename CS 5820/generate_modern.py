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

img_size = 112
samples_per_class = 1000
font_path = "./font/ipag.ttf"  # Confirmed working
font_size_range = (115, 115)  # Reduced to avoid clipping

train_images, train_labels = [], []
test_images, test_labels = [], []

for idx, char in enumerate(hiragana_chars):
    print(f"Generating for: {char} (index {idx})")
    for i in range(samples_per_class):
        # Create blank image
        img = Image.new("L", (img_size, img_size), color=0)  # Black background
        draw = ImageDraw.Draw(img)
        font_size = random.randint(*font_size_range)
        font = ImageFont.truetype(font_path, font_size)

        # Verify glyph exists
        if not font.getmask(char).getbbox():
            print(f"⚠ Glyph missing for {char}")
            continue

        # Calculate text size and center
        bbox = font.getbbox(char)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max((img_size - w) // 16, 0)
        y = max((img_size - h) // 16, 0)

        # Draw text (black on white)
        draw.text((x, y), char, font=font, fill=255)

        # Convert to NumPy
        img = np.array(img)

        # Apply thinning effect using morphological erosion
        r_num = random.randint(1, 5)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=r_num)

        a_num = random.uniform(0.02, 0.06)
        s_num = random.uniform(0.01, 0.025)

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
        #delta = 5
        #pts1 = np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])
        #pts2 = np.float32([[random.randint(0, delta), random.randint(0, delta)],
        #                   [img_size - random.randint(0, delta), random.randint(0, delta)],
        #                   [random.randint(0, delta), img_size - random.randint(0, delta)],
        #                   [img_size - random.randint(0, delta), img_size - random.randint(0, delta)]])
        #M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        #img = cv2.warpPerspective(img, M_persp, (img_size, img_size), borderValue=255
        #)

        # Add random noise
        noise = np.random.randint(0, 40, (img_size, img_size), dtype=np.uint8)
        img = cv2.subtract(img, noise)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Split into train/test
        if i < int(samples_per_class * 0.8):
            train_images.append(img)
            train_labels.append(idx)
        else:
            test_images.append(img)
            test_labels.append(idx)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Shuffle datasets
perm_train = np.random.permutation(len(train_images))
train_images = train_images[perm_train]
train_labels = train_labels[perm_train]

perm_test = np.random.permutation(len(test_images))
test_images = test_images[perm_test]
test_labels = test_labels[perm_test]

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

