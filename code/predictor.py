import cv2
import torch
import torch.nn as nn
import threading
import time
import numpy as np
from collections import deque
from PIL import Image
from torchvision import transforms

# =============================
# Labels for Kuzushiji-46
# =============================
labels = [
    "あ","い","う","え","お",
    "か","き","く","け","こ",
    "さ","し","す","せ","そ",
    "た","ち","つ","て","と",
    "な","に","ぬ","ね","の",
    "は","ひ","ふ","へ","ほ",
    "ま","み","む","め","も",
    "や","ゆ","よ",
    "ら","り","る","れ","ろ",
    "わ","を","ん"
]

torch.set_num_threads(1)

# =============================
# Device & model
# =============================
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using {device}")

class KanjiCNN(nn.Module):
    def __init__(self, num_classes=46):
        super().__init__()
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
            nn.Linear(16*7*7, 64),
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
# Transforms (eval)
# =============================
# Compute mean/std from train set for normalization (same as training)
imgs_npz = np.load("./data/hiragana_final/hiragana-train-imgs.npz")
images = imgs_npz[imgs_npz.files[0]]
mean = images.mean() / 255.0
std  = images.std()  / 255.0
del imgs_npz, images

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# =============================
# ROI extraction (deterministic)
# =============================
def extract_and_preprocess(frame_bgr, target_size=28, pad_px=20, expect_white_on_black=True):
    """
    Returns a PIL image ready for transforms and an optional debug dict.
    Deterministic: no random noise or random morphology.
    """
    h_img, w_img = frame_bgr.shape[:2]

    # 1) Gray and binarize: we want foreground as white
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # If glyphs are dark on light background, invert first so threshold yields white foreground
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        crop = bw
    else:
        # union top-2 if needed (handle separate dot)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        xs, ys, xe, ye = [], [], [], []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            xs.append(x); ys.append(y); xe.append(x+w); ye.append(y+h)

        x_min = max(0, min(xs) - pad_px)
        y_min = max(0, min(ys) - pad_px)
        x_max = min(w_img, max(xe) + pad_px)
        y_max = min(h_img, max(ye) + pad_px)

        if x_min >= x_max or y_min >= y_max:
            crop = bw
        else:
            crop = bw[y_min:y_max, x_min:x_max]

    # 3) Make square with constant background (black)
    h, w = crop.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)  # black background
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    padded[y0:y0+h, x0:x0+w] = crop

    # 4) Single mild blur to denoise; avoid dilation at inference
    padded = cv2.GaussianBlur(padded, (3, 3), 0)

    # 5) Resize once to target
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 6) Ensure polarity matches training (white glyph on black background)
    if not expect_white_on_black:
        resized = cv2.bitwise_not(resized)

    cv2.imwrite("test_image.jpg", resized)
    # 7) Convert for torchvision
    pil_img = Image.fromarray(resized)
    return pil_img, {"bbox": (x_min, y_min, x_max, y_max) if len(contours) else None}

# =============================
# Temporal smoothing (EMA on logits)
# =============================
EMA_ALPHA = 0.6  # higher = more smoothing; try 0.6–0.8
ema_logits = None
last_infer_ts = 0.0
MIN_INFER_DT = 0.05  # seconds (cap at ~20 FPS inference)

latest_frame = None
prediction_text = "Loading..."
topk_display = 5
lock = threading.Lock()
running = True

def inference_loop():
    global latest_frame, prediction_text, ema_logits, last_infer_ts, running
    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        # Throttle to avoid excess CPU/GPU usage
        now = time.time()
        if now - last_infer_ts < MIN_INFER_DT:
            time.sleep(0.005)
            continue
        last_infer_ts = now

        with lock:
            frame = latest_frame.copy()

        # ROI & preprocess
        pil_img, _dbg = extract_and_preprocess(frame, target_size=28, pad_px=20, expect_white_on_black=True)
        tensor = inference_transform(pil_img).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            logits = model(tensor)  # shape [1, C]
            # EMA smoothing
            if ema_logits is None:
                ema_logits = logits.detach().clone()
            else:
                ema_logits = EMA_ALPHA * ema_logits + (1.0 - EMA_ALPHA) * logits

            probs = torch.softmax(ema_logits[0], dim=0)
            topk = torch.topk(probs, k=topk_display)
            top_preds = [(labels[idx], probs[idx].item() * 100) for idx in topk.indices]


        main_label, main_conf = top_preds[0]

        if main_conf < 60.0:
            prediction_text = f"Uncertain: {main_label} ({main_conf:.1f}%)"
        else:
            prediction_text = f"{main_label} ({main_conf:.1f}%)"

        # Optional: print occasionally
        print("[Prediction]", prediction_text)

        time.sleep(0.5)
        # no sleep here; loop paced by MIN_INFER_DT

thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

# =============================
# Main Video Loop (with overlay)
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    running = False
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()

    cv2.imshow('Hiragana Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
