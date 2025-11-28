import json
import numpy as np

INPUT_JSON  = "scan_points_with_normals.json"       # 지금 파일
OUTPUT_JSON = "scan_points_with_normals_est.json"   # 법선까지 다시 계산한 결과

# ============================
# 1. JSON 로드
# ============================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    points = json.load(f)

# 포인트가 3D 좌표(X,Y,Z)를 가지고 있다고 가정
positions = np.array([[p["X_m"], p["Y_m"], p["Z_m"]] for p in points], dtype=float)

# ============================
# 2. 각 점마다 주변 이웃을 모아서 평면 법선 추정
# ============================

# 이웃 반경 [m] (예: 5 mm 안쪽 이웃만 사용)
NEIGHBOR_RADIUS = 0.005
R2 = NEIGHBOR_RADIUS ** 2

for i, p in enumerate(points):
    center = positions[i]

    # (자기 자신 제외) 반경 R 안의 이웃 인덱스들 찾기
    diff = positions - center        # [N,3]
    dist2 = np.sum(diff**2, axis=1)
    neighbor_idx = np.where((dist2 > 0) & (dist2 < R2))[0]

    if len(neighbor_idx) < 3:
        # 이웃이 너무 적으면 그냥 기본값 유지 (카메라 정면 방향)
        nx, ny, nz = 0.0, 0.0, 1.0
    else:
        # 이웃 포인트들로부터 평면의 법선 추정 (SVD)
        neighbor_pts = positions[neighbor_idx]
        centroid = neighbor_pts.mean(axis=0)
        A = neighbor_pts - centroid   # [K,3]

        # SVD로 최소제곱 평면 법선 구하기
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        normal = vh[-1]               # 마지막 행 벡터가 평면의 법선 방향

        # 정규화
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            nx, ny, nz = 0.0, 0.0, 1.0
        else:
            normal = normal / norm

            # RealSense 기준으로 카메라+Z 방향을 (0,0,1)로 보고,
            # 항상 "카메라 쪽"을 향하도록 부호를 맞춰줌
            if normal[2] < 0:
                normal = -normal

            nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])

    # JSON에 덮어쓰기
    p["nx"] = nx
    p["ny"] = ny
    p["nz"] = nz

# ============================
# 3. 결과 저장
# ============================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(points, f, ensure_ascii=False, indent=2)

print(f"Done. Estimated normals saved to {OUTPUT_JSON}")
