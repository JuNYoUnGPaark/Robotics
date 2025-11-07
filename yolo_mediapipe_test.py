# polygon_masking.py
from pathlib import Path
import random, time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp

# -------------------------
# 0) 환경/모델
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO(r"yolov8m_200e.pt")      # 얼굴 전용 가중치 경로로 교체 가능
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,          # 이미지 단건 처리에 적합
    max_num_faces=1,                 # 1명만 (다중 처리 가능하면 조정)
    refine_landmarks=True            # 눈/입 정밀 랜드마크 강화
)

# FaceMesh 영역별 연결셋
FM = mp.solutions.face_mesh
REGION_CONN = {
    "lips":           FM.FACEMESH_LIPS,
    "left_eye":       FM.FACEMESH_LEFT_EYE,
    "right_eye":      FM.FACEMESH_RIGHT_EYE,
    "left_eyebrow":   FM.FACEMESH_LEFT_EYEBROW,
    "right_eyebrow":  FM.FACEMESH_RIGHT_EYEBROW,
    "nose":           FM.FACEMESH_NOSE,     # 코 전체 (광범위)
    # 참고: FACE_OVAL(외곽)도 있음 — 필요 시 참고용
}

# -------------------------
# 1) 유틸 함수
# -------------------------
def connections_to_hull(connections, pts_468):
    """
    connections: {(i, j), ...} 형태의 FaceMesh 연결셋
    pts_468: [(U,V), ...] 468개 (원본 이미지 좌표계)
    return: hull(n,1,2) or None
    """
    idxs = set()
    for i, j in connections:
        idxs.add(i); idxs.add(j)
    if not idxs:
        return None
    sel = np.array([[pts_468[k][0], pts_468[k][1]] for k in idxs], dtype=np.float32)
    if sel.shape[0] < 3:
        return None
    hull = cv2.convexHull(sel.astype(np.int32))
    return hull

def draw_overlay(img, mask, color=(0, 0, 255), alpha=0.35):
    over = img.copy()
    over[mask > 0] = color
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)

# -------------------------
# 2) 입력/출력 폴더
# -------------------------
img_dir = Path(r"WIDER_FACE_YOLO/train/images")  # 원본 이미지 폴더
out_dir = Path("result_masks")
out_dir.mkdir(exist_ok=True)

# 랜덤 샘플 선택
all_imgs = list(img_dir.rglob("*.jpg"))
assert len(all_imgs) >= 20, "이미지가 20장 이상 있어야 합니다."
samples = random.sample(all_imgs, 20)

# -------------------------
# 3) 배치 처리
# -------------------------
if device == 'cuda':
    torch.cuda.synchronize()
t0 = time.perf_counter()

for idx, img_path in enumerate(samples, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] 읽기 실패: {img_path}")
        continue
    H, W = img.shape[:2]

    # 3-1) 얼굴 bbox (YOLO)
    res = yolo.predict(img, conf=0.4, device=0 if device=='cuda' else 'cpu', verbose=False)[0]
    if len(res.boxes) == 0:
        print(f"[SKIP] 얼굴 미검출: {img_path.name}")
        continue
    # 가장 큰 박스 선택(또는 conf 기준)
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    b = boxes[areas.argmax()].astype(int)
    x1, y1, x2, y2 = map(int, b[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        print(f"[SKIP] 잘못된 박스: {img_path.name}")
        continue
    roi = img[y1:y2, x1:x2]

    # 3-2) FaceMesh (ROI 기준 → 원본 좌표로 역투영)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    fm = mp_mesh.process(rgb)
    if not fm.multi_face_landmarks:
        print(f"[SKIP] FaceMesh 실패: {img_path.name}")
        continue
    lm = fm.multi_face_landmarks[0].landmark

    pts_468 = []
    for p in lm:
        U = x1 + p.x * w
        V = y1 + p.y * h
        pts_468.append((float(U), float(V)))  # 원본 좌표계

    # 3-3) 금지영역 마스크 만들기
    mask = np.zeros((H, W), np.uint8)

    # 하드 금지: 눈(좌/우), 입
    hard_regions = ["left_eye", "right_eye", "lips"]
    # 소프트 금지(옵션): 코, 눈썹 (필요시 추가)
    soft_regions = ["nose", "left_eyebrow", "right_eyebrow"]

    # 하드 금지 채우기
    for name in hard_regions:
        hull = connections_to_hull(REGION_CONN[name], pts_468)
        if hull is not None:
            cv2.fillPoly(mask, [hull], 1)

    # 소프트 금지는 색을 달리 보이게(옵션) — 일단 같은 마스크로 합침
    for name in soft_regions:
        hull = connections_to_hull(REGION_CONN[name], pts_468)
        if hull is not None:
            cv2.fillPoly(mask, [hull], 1)

    # 3-4) 안전 마진(버퍼) — 얼굴 bbox 크기 비례로 2~3% 정도
    margin = max(2, int(0.02 * max(w, h)))           # 픽셀 단위
    k = 2 * margin + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 3-5) 시각화(오버레이) & 저장
    vis = draw_overlay(img, mask, color=(0, 0, 255), alpha=0.35)

    # 디버그: 박스/폴리곤 윤곽선도 같이 그리기
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

    save_base = out_dir / (img_path.stem + "_mask")
    cv2.imwrite(str(save_base.with_suffix(".png")), mask*255)     # 바이너리 마스크
    cv2.imwrite(str(save_base.with_suffix(".jpg")), vis)          # 오버레이
    print(f"[{idx:02d}/{len(samples)}] 저장:", save_base.with_suffix(".jpg").name)

if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"\n✅ 완료! 총 {len(samples)}장, 소요: {(t1 - t0):.2f}s")

