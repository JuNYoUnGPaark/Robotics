import json
import numpy as np

# =======================================
# 1. 카메라 → 로봇 Base 변환행렬 설정 (★ 이 부분만 실제 값으로 바꿔주면 됨)
# =======================================

# 예시) 카메라 좌표계에서 로봇 Base 좌표계로 가는 3x3 회전행렬
# 실제 값은 캘리브레이션해서 얻은 값으로 교체해야 함
R_cam_to_base = np.array([
    [1.0, 0.0, 0.0],   # 예시: 단위행렬 (회전 없음)
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=float)

# 예시) 카메라 원점이 로봇 Base 좌표에서 (0.3m, 0.1m, 0.5m)에 있다고 가정
# 실제 실험 환경에 맞게 (tx, ty, tz)를 m 단위로 바꾸기
t_cam_to_base = np.array([0.30, 0.10, 0.50], dtype=float)   # [m]

# mm 단위로도 쓰고 싶으면 마지막에만 변환하면 됨.


def transform_point_cam_to_base(p_cam):
    """
    p_cam: np.array shape (3,)  [X, Y, Z] in camera frame (meters)
    return: np.array shape (3,) [X, Y, Z] in robot base frame (meters)
    """
    return R_cam_to_base @ p_cam + t_cam_to_base


def transform_normal_cam_to_base(n_cam):
    """
    n_cam: np.array shape (3,) [nx, ny, nz] in camera frame
    return: np.array shape (3,) [nx, ny, nz] in robot base frame
    (법선은 방향벡터이므로 회전만 적용, 평행이동은 적용X)
    """
    n_base = R_cam_to_base @ n_cam
    # 혹시 수치오차로 길이가 1이 아니게 되면 다시 정규화
    norm = np.linalg.norm(n_base) + 1e-8
    return n_base / norm


# =======================================
# 2. JSON 읽어서 변환 적용
# =======================================

input_path  = "scan_points_with_normals.json"          # 기존 카메라 기준 파일
output_path = "scan_points_robot_frame.json"           # 로봇 기준으로 변환한 파일

with open(input_path, "r", encoding="utf-8") as f:
    cam_points = json.load(f)

robot_points = []

for p in cam_points:
    # ------------- 카메라 좌표계 포인트 / 법선 -------------
    Xc = float(p["X_m"])
    Yc = float(p["Y_m"])
    Zc = float(p["Z_m"])
    ncx = float(p.get("nx", 0.0))
    ncy = float(p.get("ny", 0.0))
    ncz = float(p.get("nz", 1.0))

    p_cam = np.array([Xc, Yc, Zc], dtype=float)
    n_cam = np.array([ncx, ncy, ncz], dtype=float)

    # ------------- 로봇 Base 좌표계로 변환 -------------
    p_base = transform_point_cam_to_base(p_cam)    # [Xb, Yb, Zb] (m)
    n_base = transform_normal_cam_to_base(n_cam)   # [nxb, nyb, nzb]

    Xb, Yb, Zb = p_base.tolist()
    nxb, nyb, nzb = n_base.tolist()

    # mm 단위도 같이 넣고 싶으면 여기서 곱하기 1000
    Xb_mm = Xb * 1000.0
    Yb_mm = Yb * 1000.0
    Zb_mm = Zb * 1000.0

    # ------------- 출력용 dict 구성 -------------
    out = {
        # 원본 정보(카메라 기준)도 남겨두면 디버깅에 좋음
        "u": p["u"],
        "v": p["v"],
        "depth_m": p["depth_m"],
        "X_cam_m": Xc,
        "Y_cam_m": Yc,
        "Z_cam_m": Zc,
        "nx_cam": ncx,
        "ny_cam": ncy,
        "nz_cam": ncz,

        # 로봇 Base 기준 위치 (m / mm)
        "X_base_m": Xb,
        "Y_base_m": Yb,
        "Z_base_m": Zb,
        "X_base_mm": Xb_mm,
        "Y_base_mm": Yb_mm,
        "Z_base_mm": Zb_mm,

        # 로봇 Base 기준 법선 방향
        "nx_base": nxb,
        "ny_base": nyb,
        "nz_base": nzb,
    }

    robot_points.append(out)

# =======================================
# 3. 변환된 JSON 저장
# =======================================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(robot_points, f, ensure_ascii=False, indent=2)

print(f"[INFO] 변환된 포인트 {len(robot_points)}개를 '{output_path}'에 저장했습니다.")
