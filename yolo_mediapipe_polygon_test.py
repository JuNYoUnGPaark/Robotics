# face_oval_final_one.py
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
import mediapipe as mp

IMG = Path("test_img.jpg")
WTS = Path("yolov8m_200e.pt")
OUT_OVER = Path("test_img_overlay.jpg")
OUT_MASK = Path("test_img_mask.png")

assert IMG.exists(), f"이미지 없음: {IMG}"
assert WTS.exists(), f"가중치 없음: {WTS}"

device = 0 if torch.cuda.is_available() else "cpu"
yolo = YOLO(str(WTS))
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
FM = mp.solutions.face_mesh

REGION_CONN = {
    "lips":           FM.FACEMESH_LIPS,
    "left_eye":       FM.FACEMESH_LEFT_EYE,
    "right_eye":      FM.FACEMESH_RIGHT_EYE,
    "left_eyebrow":   FM.FACEMESH_LEFT_EYEBROW,
    "right_eyebrow":  FM.FACEMESH_RIGHT_EYEBROW,
    "nose":           FM.FACEMESH_NOSE,
}
FACE_OVAL = FM.FACEMESH_FACE_OVAL  # 얼굴 외곽

def hull_from(conns, pts):
    ids = set()
    for i, j in conns: ids.add(i); ids.add(j)
    if len(ids) < 3: return None
    sel = np.array([[pts[k][0], pts[k][1]] for k in ids], dtype=np.int32)
    return cv2.convexHull(sel)

img = cv2.imread(str(IMG))
H, W = img.shape[:2]

# 1) YOLO 얼굴 bbox
r = yolo.predict(img, conf=0.4, device=device, imgsz=640, verbose=False)[0]
if len(r.boxes) == 0:
    raise SystemExit("얼굴 미검출")
boxes = r.boxes.xyxy.cpu().numpy()
areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
x1,y1,x2,y2 = boxes[areas.argmax()].astype(int)
x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(W-1,x2), min(H-1,y2)
w,h = x2-x1, y2-y1
roi = img[y1:y2, x1:x2]

# 2) FaceMesh (ROI→원본 좌표)
fm = mp_mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
if not fm.multi_face_landmarks:
    raise SystemExit("FaceMesh 실패")
lm = fm.multi_face_landmarks[0].landmark
pts = [(x1 + p.x*w, y1 + p.y*h) for p in lm]  # 468개 (원본 좌표)

# 3) 얼굴 외곽 마스크 (목/어깨 제거)
face_mask = np.zeros((H, W), np.uint8)
oval = hull_from(FACE_OVAL, pts)
if oval is not None:
    cv2.fillPoly(face_mask, [oval], 255)
else:
    # 폴백: bbox 타원 근사
    center = (int(x1+w/2), int(y1+h*0.55))
    axes = (int(w*0.48), int(h*0.62))
    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)

# 4) 금지 영역(눈/입/코/눈썹) + 안전 마진
forbid = np.zeros((H, W), np.uint8)
for name in REGION_CONN:
    hull = hull_from(REGION_CONN[name], pts)
    if hull is not None:
        cv2.fillPoly(forbid, [hull], 255)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))  # ≈8px 마진
forbid = cv2.dilate(forbid, kernel, 1)

# 5) 최종 허용 마스크 = 얼굴외곽 ∩ bbox − 금지영역
bbox_mask = np.zeros((H, W), np.uint8)
cv2.rectangle(bbox_mask, (x1,y1), (x2,y2), 255, -1)
allow = cv2.bitwise_and(face_mask, bbox_mask)
allow = cv2.bitwise_and(allow, cv2.bitwise_not(forbid))  # 허용=255

# 6) 저장
overlay = img.copy()
overlay[forbid>0] = (0, 0, 255)                   # 금지=빨강
overlay[(allow>0) & (forbid==0)] = (0, 255, 0)    # 허용=초록(얼굴 내부)
overlay = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imwrite(str(OUT_OVER), overlay)       # 사람이 보는 확인용
cv2.imwrite(str(OUT_MASK), allow)         # 알고리즘용(허용=255, 나머지=0)
print("saved:", OUT_OVER, OUT_MASK)
