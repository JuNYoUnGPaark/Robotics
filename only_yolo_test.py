# Library Import
import os
import random
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# environment Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(r"yolov8m_200e.pt")

img_dir = Path(r"WIDER_FACE_YOLO/train/images")

out_dir = Path("result_images")
out_dir.mkdir(exist_ok=True)

# Select 20 Images Random
all_imgs = list(img_dir.rglob("*.jpg"))
selected_imgs = random.sample(all_imgs, 20)

latencies = []

if device == 'cuda':
    torch.cuda.synchronize()
t0 = time.perf_counter()

# inference and save results
for img_path in selected_imgs:
    img = cv2.imread(str(img_path))

    # per-image timer
    if device == 'cuda':
        torch.cuda.synchronize()
    t_img0 = time.perf_counter()

    results = model.predict(source=img, conf=0.5, device=0 if device=='cuda' else 'cpu', verbose=False)

    if device == 'cuda':
        torch.cuda.synchronize()
    t_img1 = time.perf_counter()
    latencies.append((t_img1 - t_img0) * 1000.0)  # ms

    for r in results:
        for box in r.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    save_path = out_dir / img_path.name
    cv2.imwrite(str(save_path), img)
    print("저장:", save_path)

# ---- Total timer end ----
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.perf_counter()

# Summary
total_sec = t1 - t0
n = len(selected_imgs)
avg_ms = sum(latencies) / n
fps = n / total_sec if total_sec > 0 else float('inf')

print("\n=== Inference Summary ===")
print(f"총 이미지 수       : {n}")
print(f"총 소요 시간       : {total_sec:.3f} s")
print(f"이미지당 평균 시간 : {avg_ms:.2f} ms")
print(f"처리 FPS           : {fps:.2f} frames/s")
print(f"(장치: {device})")