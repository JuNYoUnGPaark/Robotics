import numpy as np
import json

# -------------------------
# 1) extrinsic 로드 (R: rotation, t: translation)
# -------------------------
with open("cam_to_robot_manual.json", "r") as f:
    extr = json.load(f)

R = np.array(extr["R"], dtype=np.float64)      # rotation ( 단위 없음 )
t_mm = np.array(extr["t"], dtype=np.float64)   # translation (mm 단위)

# -------------------------
# 2) 카메라 기준 마커 로드
# -------------------------
with open("red_markers_3d.json", "r") as f:
    data = json.load(f)

markers = data["markers_3d"]

robot_points = []

for m in markers:
    # Pc: meter → mm 변환
    Pc_m = np.array([m["X_m"], m["Y_m"], m["Z_m"]], dtype=np.float64)
    Pc_mm = Pc_m * 1000.0

    # 로봇 좌표계 변환
    Pr_mm = R @ Pc_mm + t_mm

    robot_points.append(Pr_mm.tolist())

print("\n=== Robot Coordinates (mm) ===")
for i, p in enumerate(robot_points):
    print(f"P{i+1}:", p)
