import numpy as np
from PIL import Image
import os

# Paths to your dataset
imgs_path = "./data/K49/K49-train-imgs.npz"
labels_path = "./data/K49/K49-train-labels.npz"

# Load data
imgs_npz = np.load(imgs_path)
labels_npz = np.load(labels_path)

images = imgs_npz[imgs_npz.files[0]]  # or imgs_npz["images"] if named
labels_array = labels_npz[labels_npz.files[0]]  # or labels_npz["labels"]

# Hiragana labels
labels = [
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

# Directory to save samples
output_dir = "./sample_kuzushiji"
os.makedirs(output_dir, exist_ok=True)

# Extract 10 random samples
for i in range(10):
    idx = np.random.randint(0, len(images))
    img_np = images[idx]
    label_idx = labels_array[idx]
    label_char = labels[label_idx]
    img_pil = Image.fromarray(img_np.astype(np.uint8), mode="L")
    img_pil.save(os.path.join(output_dir, f"{label_char}_{idx}.png"))

print(f"✅ Saved 10 samples to {output_dir}")

