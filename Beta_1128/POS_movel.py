import json
import numpy as np
from pathlib import Path
'''
각 점을 Doosan M0609 TCP 타겟으로 보내서

P = (X_robot_m, Y_robot_m, Z_robot_m)

방향은 n = (nx_robot, ny_robot, nz_robot)을 이용해서
TCP orientation 구성 → movel() 같은 걸로 접근
'''
# ==========================
# 0. 입력/출력 파일 경로
# ==========================
INPUT_JSON  = "scan_points_with_normals.json"   # 카메라 기준 JSON
OUTPUT_JSON = "scan_points_robot_frame.json"    # 로봇 기준 JSON (새로 저장)


# ==========================
# 1. 카메라 → 로봇 변환 행렬 정의
#    (실제 프로젝트에서 이 값만 캘리브레이션해서 교체하면 됨)
# ==========================

def get_cam_to_robot_transform():
    """
    카메라 좌표계에서 로봇 베이스 좌표계로 가는
    3x3 회전행렬 R, 3x1 평행이동 t 를 반환.
    
    - 지금은 예시 값!!  (대충 카메라가 로봇 앞쪽에 있고 살짝 아래에서 바라보는 상황 가정)
    - 실제로는 hand–eye calibration 해서 나온 R, t 값으로 교체해야 함.
    """

    # 예시: z축은 거의 동일, 카메라 x,y가 로봇 x,y에 약간 섞여 있는 정도
    # (그냥 기본 단위행렬 + 약간의 회전 예시, 나중에 전부 바꾸면 됨)
    # R = I 로 두고 시작해도 됨.
    R = np.eye(3)

    # 예시: 로봇 베이스 원점 기준으로 카메라 위치 (m)
    # 예를 들어, 카메라가 로봇 앞쪽 0.4m, 위쪽 0.8m 지점에 있다면:
    t = np.array([0.4, 0.0, 0.8])   # [tx, ty, tz]  (meter)

    return R, t


# ==========================
# 2. 변환 함수들
# ==========================

def transform_point_cam_to_robot(p_cam, R, t):
    """
    p_cam: (3,) numpy array, [X, Y, Z] in camera frame
    R: (3,3) rotation matrix
    t: (3,)  translation vector

    return: (3,) numpy array, [Xr, Yr, Zr] in robot base frame
    """
    return R @ p_cam + t


def transform_normal_cam_to_robot(n_cam, R):
    """
    n_cam: (3,) numpy array, [nx, ny, nz] in camera frame
    R: (3,3) rotation matrix

    return: (3,) numpy array, [nx_r, ny_r, nz_r] in robot base frame
    (단위 벡터 유지)
    """
    n_r = R @ n_cam
    # 혹시라도 수치오차로 norm이 1이 아니게 되면 다시 정규화
    norm = np.linalg.norm(n_r) + 1e-8
    return n_r / norm


# ==========================
# 3. 메인 변환 로직
# ==========================

def main():
    in_path  = Path(INPUT_JSON)
    out_path = Path(OUTPUT_JSON)

    if not in_path.exists():
        raise FileNotFoundError(f"입력 JSON 파일을 찾을 수 없습니다: {in_path.resolve()}")

    # 1) JSON 로드
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] 입력 포인트 개수: {len(data)}")

    # 2) 변환 행렬 가져오기
    R, t = get_cam_to_robot_transform()

    # 3) 각 포인트에 대해 변환 적용
    out_data = []
    for p in data:
        X, Y, Z = p["X_m"], p["Y_m"], p["Z_m"]
        nx, ny, nz = p["nx"], p["ny"], p["nz"]

        p_cam = np.array([X, Y, Z], dtype=float)
        n_cam = np.array([nx, ny, nz], dtype=float)

        p_robot = transform_point_cam_to_robot(p_cam, R, t)
        n_robot = transform_normal_cam_to_robot(n_cam, R)

        out_data.append({
            # 원본 픽셀/깊이 정보는 그대로 유지 (원하면 삭제해도 됨)
            "u": p["u"],
            "v": p["v"],
            "depth_m": p["depth_m"],

            "X_cam_m": float(X),
            "Y_cam_m": float(Y),
            "Z_cam_m": float(Z),

            "nx_cam": float(nx),
            "ny_cam": float(ny),
            "nz_cam": float(nz),

            # 로봇 좌표계 기준 위치/법선
            "X_robot_m": float(p_robot[0]),
            "Y_robot_m": float(p_robot[1]),
            "Z_robot_m": float(p_robot[2]),

            "nx_robot": float(n_robot[0]),
            "ny_robot": float(n_robot[1]),
            "nz_robot": float(n_robot[2]),
        })

    # 4) JSON 저장
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 변환 완료. 결과를 {out_path.name} 에 저장했습니다.")
    print("      (나중에 R, t를 실제 캘리브레이션 값으로 교체하면 바로 재사용 가능)")


if __name__ == "__main__":
    main()
