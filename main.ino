#include <Servo.h>

const int SERVO_PIN = 9;
const int REST_ANGLE = 0;
const int PRESS_ANGLE = 58;

// 서보가 이동하는 시간(왕복)을 충분히 크게 줌
const unsigned long MOVE_TIME = 400;    // 0→180 또는 180→0 가는 데 걸리는 시간(ms)
const unsigned long SETTLE_TIME = 200;  // 멈추고 진동 가라앉는 시간(ms)

Servo bloodServo;

void setup() {
  bloodServo.attach(SERVO_PIN);
  bloodServo.write(REST_ANGLE);

  Serial.begin(115200);
  delay(1000);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();      // 한 글자 읽기

    Serial.print("got: ");
    Serial.println(c);

    if (c == '1') {              // '1' 받으면만 동작
      pressOnce();
    }
  }
}

void pressOnce() {
  Serial.println("START");

  // 1) 0도 -> 180도
  bloodServo.write(PRESS_ANGLE);
  delay(MOVE_TIME);          // ★ 실제로 180도까지 가는 시간을 충분히 기다림

  // 2) 180도 -> 0도
  bloodServo.write(REST_ANGLE);
  delay(MOVE_TIME);          // ★ 다시 0도로 돌아오는 시간 기다림

  // 3) 진동/관성 가라앉는 시간
  delay(SETTLE_TIME);

  // 4) 이제서야 동작 완료 신호 보내기
  Serial.println("DONE");
}
