import cv2, numpy as np, torch, mediapipe as mp, random
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

IMG, WTS = Path("test_img.jpg"), Path("yolov8m_200e.pt")
device = 0 if torch.cuda.is_available() else "cpu"
yolo = YOLO(str(WTS))
FM = mp.solutions.face_mesh
mesh = FM.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def hull_from(conns, pts):
    ids=set()
    for i,j in conns: ids.add(i); ids.add(j)
    if len(ids)<3: return None
    sel=np.array([[pts[k][0],pts[k][1]] for k in ids],dtype=np.int32)
    return cv2.convexHull(sel)

# --- 1. 얼굴+마스크 ---
img = cv2.imread(str(IMG)); H,W=img.shape[:2]
r = yolo.predict(img, conf=0.4, device=device, imgsz=640, verbose=False)[0]
boxes=r.boxes.xyxy.cpu().numpy(); areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
x1,y1,x2,y2=boxes[areas.argmax()].astype(int)
roi=img[y1:y2, x1:x2]; w,h=x2-x1, y2-y1

res=mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
lm=res.multi_face_landmarks[0].landmark
pts=[(x1+p.x*w, y1+p.y*h) for p in lm]

REG={"lips":FM.FACEMESH_LIPS,"left_eye":FM.FACEMESH_LEFT_EYE,"right_eye":FM.FACEMESH_RIGHT_EYE,
     "left_eyebrow":FM.FACEMESH_LEFT_EYEBROW,"right_eyebrow":FM.FACEMESH_RIGHT_EYEBROW,"nose":FM.FACEMESH_NOSE}
forbid=np.zeros((H,W),np.uint8)
for k in REG:
    h_=hull_from(REG[k],pts)
    if h_ is not None: cv2.fillPoly(forbid,[h_],255)
forbid=cv2.dilate(forbid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)),1)

oval=hull_from(FM.FACEMESH_FACE_OVAL,pts)
face=np.zeros((H,W),np.uint8); cv2.fillPoly(face,[oval],255)
allow=cv2.bitwise_and(face,cv2.bitwise_not(forbid))

# --- 2. 후보 좌표 샘플링 ---
step=20; margin=10
dist=cv2.distanceTransform((allow>0).astype(np.uint8),cv2.DIST_L2,5)
cands=[(x,y) for y in range(y1+step//2,y2,step) for x in range(x1+step//2,x2,step) if allow[y,x]==255 and dist[y,x]>margin]

# --- 3. 랜덤 Z 붙여서 3D ---
Z_MIN,Z_MAX=20,50  # mm 범위
points_3d=[(u,v,random.uniform(Z_MIN,Z_MAX)) for (u,v) in cands]
points_3d=np.array(points_3d)

# --- 4. matplotlib 시각화 ---
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c=points_3d[:,2], cmap="viridis", s=30)

ax.set_xlabel("u (px)")
ax.set_ylabel("v (px)")
ax.set_zlabel("Z (mm, simulated)")
ax.set_title("Simulated 3D Candidate Points")

plt.show()
