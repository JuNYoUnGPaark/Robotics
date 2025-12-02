import numpy as np
import json

def compute_rigid_transform(points_cam, points_robot):
    """
    points_cam: Nx3, 카메라 좌표계 (예: meter)
    points_robot: Nx3, 로봇 좌표계 (예: meter, base 기준)
    """
    A = np.asarray(points_cam, dtype=np.float64)
    B = np.asarray(points_robot, dtype=np.float64)
    assert A.shape == B.shape
    N = A.shape[0]
    assert N >= 3, "최소 3개 이상의 점이 필요합니다."

    # 중심 제거
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB

    # SVD로 회전 구하기
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # det(R) < 0면 reflection 제거
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 평행이동 t
    t = cB - R @ cA
    return R, t

def transform_cam_to_robot(point_cam, R, t):
    p = np.asarray(point_cam, dtype=np.float64)
    return R @ p + t

if __name__ == "__main__":
    # 예시: 여기다가 직접 측정한 값들 채워 넣으면 됨

    # 1) 카메라 기준 3D 좌표들 (Xc, Yc, Zc), 단위: meter
    #    → D435로 계산한 값들 그대로
    cam_points = [
        [0.0489526427009792,  0.15243016530065234, 0.29500001668930054],   # P1
        [-0.028822555660774223, 0.09873652250468935, 0.31300002336502075],   # P2
        [0.11401178964780033,  0.0711101525420211, 0.3020000159740448],   # P3
        [-0.03881912001421954, 0.06398809709676474, 0.30900001525878906],
        [0.03367187184766111, -0.055543415788112026, 0.2880000174045563]
    ]

    # 2) 같은 점들에 해당하는 로봇 베이스 기준 좌표 (Xr, Yr, Zr)
    #    로봇에서 mm로 읽어왔다면 /1000.0 해서 meter로 변환해서 넣기
    robot_points = [
        [386.26, -5.41, 145.27],
        [429.62, 71.79, 121.12],
        [420.2, 91.74, 115.44],
        [363.03, 122.87, 133.36],
        [327.42, 70.23, 125.61]
    ]

    R, t = compute_rigid_transform(cam_points, robot_points)
    print("=== R (3x3 회전 행렬) ===")
    print(R)
    print("\n=== t (이동 벡터) ===")
    print(t)

    # JSON으로 저장해두면, 다른 코드에서 바로 로드해서 사용 가능
    extrinsic = {
        "R": R.tolist(),
        "t": t.tolist()
    }
    with open("cam_to_robot_manual.json", "w", encoding="utf-8") as f:
        json.dump(extrinsic, f, indent=2)
    print("\n[INFO] cam_to_robot_manual.json 저장 완료")

    # 테스트: 임의의 새로운 카메라 점 하나 변환해보기
    test_cam = np.array([0.100, 0.000, 0.370])
    test_robot = transform_cam_to_robot(test_cam, R, t)
    print("\n[TEST] 카메라 점", test_cam, "→ 로봇 좌표", test_robot)
