import socket
import json
import math

HOST = "192.168.137.254"
PORT = 200

# with_calibraion.py에서 만들어준 파일 이름 그대로 사용
# (scan_points_robot_frame.json 안 구조: x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad, pose 등) :contentReference[oaicite:0]{index=0}
JSON_PATH = "scan_points_robot_frame.json"
MAX_POINTS = None   # None이면 JSON 전체 사용, 숫자 넣으면 그 개수까지만 사용


def load_coords_from_json(path, max_points=None):
    """
    scan_points_robot_frame.json을 읽어서
    Doosan에 보낼 "x,y,z,rx,ry,rz" 문자열 리스트로 변환.
    - x,y,z: mm (JSON 그대로 사용)
    - rx,ry,rz: rad → deg 변환
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    coords = []
    for i, p in enumerate(data):
        if max_points is not None and i >= max_points:
            break

        # --- 위치(mm) ---
        if "x_mm" in p:
            x = p["x_mm"]
            y = p["y_mm"]
            z = p["z_mm"]
        elif "pose" in p and len(p["pose"]) >= 3:
            # pose = [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
            x, y, z = p["pose"][:3]
        else:
            # 형식 이상하면 스킵
            continue

        # --- 각도(rad → deg) ---
        if "rx_rad" in p:
            rx_rad = p["rx_rad"]
            ry_rad = p["ry_rad"]
            rz_rad = p["rz_rad"]
        elif "pose" in p and len(p["pose"]) >= 6:
            _, _, _, rx_rad, ry_rad, rz_rad = p["pose"][:6]
        else:
            rx_rad = ry_rad = rz_rad = 0.0

        rx_deg = rx_rad * 180.0 / math.pi
        ry_deg = ry_rad * 180.0 / math.pi
        rz_deg = rz_rad * 180.0 / math.pi

        # 필요에 따라 자릿수는 조절 가능
        coord_str = f"{x:.1f},{y:.1f},{z:.1f},{rx_deg:.3f},{ry_deg:.3f},{rz_deg:.3f}"
        coords.append(coord_str)

    print(f"[SERVER] JSON에서 좌표 {len(coords)}개 로드 완료")
    return coords


def start_server():
    # 1) JSON에서 좌표 미리 로드
    coords = load_coords_from_json(JSON_PATH, max_points=MAX_POINTS)
    if not coords:
        print("[SERVER] 보낼 좌표가 없습니다. JSON 파일을 확인하세요.")
        return

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[SERVER] 로봇 접속 대기 중... (PORT: {PORT})")

    conn, addr = server.accept()
    print(f"[SERVER] 로봇 접속됨 → {addr}")

    idx = 0  # 몇 번째 좌표를 보낼 차례인지

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print("[SERVER] 클라이언트 연결 종료")
                break

            msg = data.decode().strip()
            print(f"[FROM ROBOT] {msg}")

            # 로봇이 shot 요청
            if msg == "shot":
                # 인덱스 범위 체크
                if idx >= len(coords):
                    # 다 썼으면 마지막 포인트 반복 or break 선택
                    # 여기선 마지막 포인트 계속 반복
                    idx = len(coords) - 1

                coord = coords[idx]
                idx += 1

                send_msg = coord + "\r\n"
                conn.sendall(send_msg.encode("utf-8"))
                print(f"[TO ROBOT] {send_msg}")

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    conn.close()
    server.close()


if __name__ == "__main__":
    start_server()
