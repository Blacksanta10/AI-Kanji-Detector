import cv2
import torch
import torch.nn as nn
import threading
import time
import numpy as np
from PIL import Image
from torchvision import transforms

# =============================
# Labels for Kuzushiji-49
# =============================
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
    "わ", "ゐ", "ゑ", "を",
    "ん", "ゝ"
]

torch.set_num_threads(1)

# =============================
# Device Detection
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

# =============================
# CNN Model
# =============================
class KanjiCNN(nn.Module):
    def __init__(self, num_classes=49):
        super(KanjiCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = KanjiCNN().to(device)
model.load_state_dict(torch.load("best_kanji_cnn_model.pth", map_location=device))
model.eval()

# =============================
# Transform (Same as Training)
# =============================
# Load your training images
imgs_npz = np.load("./data/hiragana_final/hiragana-train-imgs.npz")
images = imgs_npz[imgs_npz.files[0]]  # Assuming first key contains image data

# Compute mean and std across all pixels
mean = images.mean() / 255.0  # Normalize to [0,1] range if original is 0-255
std = images.std() / 255.0

inference_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# =============================
latest_frame = None
prediction_text = "Loading..."
lock = threading.Lock()
running = True

# =============================
# Inference Thread (Corrected)
# =============================
def extract_and_preprocess(frame, target_size=28):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invert colors
    inverted = cv2.bitwise_not(gray)

    # OTSU threshold
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # Fallback: use entire image
        crop = thresh
    else:
        # Largest contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        margin = 60
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = w + margin * 2
        h = h + margin * 2
        crop = thresh[y:y+h, x:x+w]

    # Make square crop
    h, w = crop.shape
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half_side = side // 2
    x1 = max(cx - half_side, 0)
    y1 = max(cy - half_side, 0)
    x2 = min(cx + half_side, thresh.shape[1])
    y2 = min(cy + half_side, thresh.shape[0])
    crop = thresh[y1:y2, x1:x2]

    # Adaptive stroke adjustment
    white_ratio = np.sum(crop == 255) / (crop.shape[0] * crop.shape[1])
    kernel = np.ones((2, 2), np.uint8)
    if white_ratio < 0.05:
        crop = cv2.dilate(crop, kernel, iterations=1)
    elif white_ratio > 0.25:
        crop = cv2.erode(crop, kernel, iterations=1)

    # Resize to target size
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Add slight blur and noise
    resized = cv2.GaussianBlur(resized, (3, 3), 0)

    noise = np.random.normal(0, 2, resized.shape).astype(np.float32)
    resized = resized.astype(np.float32) + noise
    resized = np.clip(resized, 0, 255).astype(np.uint8)

    cv2.imwrite("test_image.jpg", resized)

    return Image.fromarray(resized), white_ratio

def inference_loop():
    global latest_frame, prediction_text, running
    while running:
        if latest_frame is not None:
            with lock:
                frame_copy = latest_frame.copy()

            # Preprocess image
            img_pil, white_ratio = extract_and_preprocess(frame_copy)
            tensor = inference_transform(img_pil).unsqueeze(0).to(device)

            # Debug: Save processed image
            print(f"White ratio: {white_ratio:.2f}")

            # Model prediction
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            # Top-3 predictions
            top3 = torch.topk(probs, 3)
            top_preds = [(labels[idx], probs[idx].item() * 100) for idx in top3.indices]

            # Confidence threshold
            main_label, main_conf = top_preds[0]
            if main_conf < 50:
                prediction_text = f"Uncertain: {main_label} ({main_conf:.1f}%)"
            else:
                prediction_text = f"{main_label} ({main_conf:.1f}%)"

            # Print debug info
            print(f"[Prediction] {prediction_text}")
            print("Top-3:", ", ".join([f"{lbl}: {conf:.1f}%" for lbl, conf in top_preds]))

        time.sleep(0.5)

thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

# =============================
# Main Video Loop
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    running = False
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()

    cv2.imshow('Kanji Live Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
