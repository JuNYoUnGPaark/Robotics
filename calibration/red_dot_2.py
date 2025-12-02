import cv2
import numpy as np
import mediapipe as mp
import time
import json

import pyrealsense2 as rs  # Intel RealSense SDK


# --------------------------
# 0. 얼굴/예외 영역 정의
# --------------------------

# 얼굴 외곽(oval) 인덱스
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# 눈/코/입 중심용 인덱스 (평균을 써서 대략 중심점 잡기)
LEFT_EYE_CENTER_IDX  = [33, 133]
RIGHT_EYE_CENTER_IDX = [362, 263]
NOSE_CENTER_IDX      = [1, 4]
MOUTH_CENTER_IDX     = [13, 14]

# 눈/코/입 제외할 반경 (픽셀 단위, 필요하면 조절)
EXCLUDE_EYE_RADIUS   = 35
EXCLUDE_NOSE_RADIUS  = 30
EXCLUDE_MOUTH_RADIUS = 40


def get_point_mean(landmarks, idx_list, w, h):
    """FaceMesh 랜드마크들의 평균 픽셀 좌표 (u, v)를 구하는 유틸"""
    pts = []
    for idx in idx_list:
        lm = landmarks.landmark[idx]
        u = int(lm.x * w)
        v = int(lm.y * h)
        pts.append((u, v))
    if not pts:
        return None
    mx = int(sum(p[0] for p in pts) / len(pts))
    my = int(sum(p[1] for p in pts) / len(pts))
    return mx, my


# --------------------------
# 1. 빨간 마커 탐지 (HSV 마스크)
# --------------------------
def find_red_markers(frame_bgr):
    """
    BGR 이미지를 입력받아,
    HSV 마스크로 빨간색 영역을 찾아
    각 덩어리(마커)의 중심 좌표 리스트를 반환.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 빨간색 범위
    lower_red1 = np.array([0,   150, 120])
    upper_red1 = np.array([8,   255, 255])
    lower_red2 = np.array([172, 150, 120])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 잡음 제거용 morphological operation
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 컨투어(빨간 영역) 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:   # 마커 크기에 따라 20~50 사이에서 조절
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        red_points.append((cx, cy))

    return red_points, mask


# --------------------------
# 2. 메인 함수 (D435 사용)
# --------------------------
def main():
    # ==============
    # RealSense 초기화
    # ==============
    pipeline = rs.pipeline()
    config = rs.config()

    # 컬러 + 뎁스 스트림 활성화
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 파이프라인 시작
    profile = pipeline.start(config)

    # depth를 color에 align (픽셀 좌표 맞추기)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 카메라 내참수(Depth intrinsics) 가져오기
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_stream.get_intrinsics()
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(f"[INFO] Depth intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ==============
    # MediaPipe FaceMesh 초기화
    # ==============
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            # ==============
            # 1) D435에서 프레임 받기
            # ==============
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # numpy 배열로 변환
            frame = np.asanyarray(color_frame.get_data())  # BGR
            h, w, _ = frame.shape

            # ==============
            # 2) FaceMesh로 "얼굴 전체" + 눈/코/입 중심 구하기
            # ==============
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            # 얼굴 마스크 (얼굴 전체에서 눈/코/입 제외 영역)
            face_mask = np.zeros((h, w), dtype=np.uint8)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # 2-1) 얼굴 외곽 poly
                face_oval_poly = []
                for idx in FACE_OVAL_IDX:
                    lm = face_landmarks.landmark[idx]
                    u = int(lm.x * w)
                    v = int(lm.y * h)
                    face_oval_poly.append((u, v))

                if len(face_oval_poly) >= 3:
                    oval_np = np.array(face_oval_poly, np.int32)
                    # debug: 얼굴 외곽 라인 표시
                    cv2.polylines(frame, [oval_np], True, (0, 255, 0), 1)
                    # 마스크에 얼굴 영역 채우기
                    cv2.fillPoly(face_mask, [oval_np], 255)

                # 2-2) 눈/코/입 중심점
                left_eye_center  = get_point_mean(face_landmarks, LEFT_EYE_CENTER_IDX,  w, h)
                right_eye_center = get_point_mean(face_landmarks, RIGHT_EYE_CENTER_IDX, w, h)
                nose_center      = get_point_mean(face_landmarks, NOSE_CENTER_IDX,      w, h)
                mouth_center     = get_point_mean(face_landmarks, MOUTH_CENTER_IDX,     w, h)

                # debug: 중심점 찍기
                for c, col in [
                    (left_eye_center,  (0, 255, 255)),
                    (right_eye_center, (0, 255, 255)),
                    (nose_center,      (0, 165, 255)),
                    (mouth_center,     (255, 255, 0)),
                ]:
                    if c is not None:
                        cv2.circle(frame, c, 3, col, -1)

                # 2-3) 얼굴 마스크에서 눈/코/입 영역은 0으로 지워버리기
                def erase_circle(center, radius):
                    if center is None:
                        return
                    cx_i, cy_i = center
                    cv2.circle(face_mask, (cx_i, cy_i), radius, 0, -1)

                erase_circle(left_eye_center,  EXCLUDE_EYE_RADIUS)
                erase_circle(right_eye_center, EXCLUDE_EYE_RADIUS)
                erase_circle(nose_center,      EXCLUDE_NOSE_RADIUS)
                erase_circle(mouth_center,     EXCLUDE_MOUTH_RADIUS)

            # ==============
            # 3) HSV로 빨간 마커 탐지
            # ==============
            red_points, red_mask = find_red_markers(frame)

            face_markers = []
            others = []

            # ==============
            # 4) 마커를 “얼굴 내부(눈/코/입 제외)” vs 그 외로 분류 + depth 측정
            # ==============
            for (cx_p, cy_p) in red_points:
                depth_m = float(depth_frame.get_distance(cx_p, cy_p))  # meter 단위

                # 얼굴 마스크 값 확인 (255면 얼굴, 0이면 제외)
                if 0 <= cy_p < h and 0 <= cx_p < w and face_mask[cy_p, cx_p] == 255:
                    face_markers.append((cx_p, cy_p, depth_m))
                    # 얼굴 위의 마커: 연두색 + depth 표시
                    cv2.circle(frame, (cx_p, cy_p), 7, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"F {depth_m:.3f}m",
                        (cx_p + 5, cy_p - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    others.append((cx_p, cy_p, depth_m))
                    # 그 외: 보라색
                    cv2.circle(frame, (cx_p, cy_p), 6, (255, 0, 255), -1)

            # ==============
            # 5) 화면 표시
            # ==============
            cv2.putText(
                frame,
                "Press 's' to SAVE 3D red markers, 'q' to quit",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("D435 Color + Face Mask + Red Markers", frame)
            # 디버그용:
            # cv2.imshow("Red Mask", red_mask)
            # cv2.imshow("Face Mask", face_mask)

            key = cv2.waitKey(1) & 0xFF

            # -------------------------
            # ★ 특정 시점: 's' 키 → 3D 좌표 저장
            # -------------------------
            if key == ord('s'):
                if len(face_markers) == 0:
                    print("[WARN] 현재 얼굴에서 빨간 점이 감지되지 않았습니다. 저장 안 함.")
                else:
                    markers_3d = []
                    for (u, v, d_m) in face_markers:
                        # pinhole 모델로 카메라 3D 좌표 계산
                        Z = d_m
                        X = (u - cx) / fx * Z
                        Y = (v - cy) / fy * Z
                        markers_3d.append({
                            "u": int(u),
                            "v": int(v),
                            "depth_m": float(Z),
                            "X_m": float(X),
                            "Y_m": float(Y),
                            "Z_m": float(Z),
                        })

                    save_payload = {
                        "timestamp": time.time(),
                        "markers_3d": markers_3d
                    }

                    with open("red_markers_3d.json", "w", encoding="utf-8") as f:
                        json.dump(save_payload, f, ensure_ascii=False, indent=2)

                    print("[SAVE] red_markers_3d.json 저장 완료 (현재 프레임의 얼굴 빨간 점 3D 좌표)")

            # 종료
            if key == ord('q'):
                break

    finally:
        face_mesh.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
