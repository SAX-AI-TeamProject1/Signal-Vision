# Last updated: 2026-07-13

"""웹캠 영상에서 MediaPipe HandLandmarker 랜드마크를 실시간 시각화하는 검증 스크립트.

카메라 입력 → 손 랜드마크 추출이 정상 동작하는지 확인한다.
mediapipe 0.10.35는 구형 solutions API가 제거되어 Tasks API(HandLandmarker)를 사용한다.
최초 실행 시 모델 파일(hand_landmarker.task, 약 8MB)을 models/ 폴더에 자동 다운로드한다.

실행: F5 또는 VSCode 태스크 "실행: 랜드마크 시각화" (종료: q 또는 ESC)

이 파일 자체는 학습/예측을 하지 않는 검증용 뷰어다. 모델 경로/다운로드/그리기 로직은
src/capture/extractor.py에 공용으로 정의되어 있고(이 파일이 그걸 가져다 쓰는 방향),
extractor.py는 데이터 수집(dataset/collect.py) → 학습(training/train.py) →
실시간 추론(inference/predict.py)까지 동일한 형식으로 이어진다.

학습/예측 알고리즘: 30프레임 랜드마크 시퀀스를 PyTorch로 구현한 LSTM(2층, hidden 64) 시퀀스
분류 모델로 학습·추론한다 — 수신호는 정지 자세가 아니라 시간에 따른 궤적(동적 제스처)이라
시계열을 다루는 RNN 계열을 선택했다 (근거: doc/ml-flow.md 3절, doc/design.md).
"""

import platform # OS(Windows/macOS/Linux) 판별 — 카메라 백엔드를 OS별로 분기하기 위함
import time # 프레임 타임스탬프(ms), FPS 계산, 경고 깜빡임 주기 계산

import cv2 # OpenCV : 카메라 입력, 프레임 그리기/표시

from src.capture.extractor import (
    HAND_CONNECTIONS,
    create_hand_landmarker,
    to_mp_image,
    draw_skeleton,
)

# 손 이탈 경고: 연속 미감지가 이 프레임 수를 넘으면 화면을 빨갛게 깜빡인다
WARN_AFTER_MISSES = 5


# 학습 데이터 수집 중 손이 프레임 밖으로 나가면 그 구간은 0으로 채워진 저품질 샘플이 되므로,
# 수집자가 바로 알아챌 수 있도록 화면에 강한 시각 경고를 준다.
class HandWarning:
    """손이 화면에서 사라지면 빨간 깜빡임 경고를 그린다. (시각화·수집·추론 공용)"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.misses = 0  # 연속 미검출 프레임 수

    def update(self, hand_result) -> None:
        # 이번 프레임에 손이 하나라도 잡히면 카운트 리셋, 아니면 누적
        self.misses = 0 if hand_result.hand_landmarks else self.misses + 1

    def draw(self, frame) -> None:
        if not self.enabled or self.misses < WARN_AFTER_MISSES:  # 임계치 미만이면 그리지 않음
            return
        h, w = frame.shape[:2]
        # 2Hz 깜빡임: 현재 시각(초)에 4를 곱해 반 주기(0.25초)마다 정수가 1씩 올라가게 만들고,
        # 그 정수가 짝수/홀수인지(%2)로 on/off를 토글한다 → 1초에 2번 켜짐/꺼짐 반복
        blink_on = int(time.monotonic() * 4) % 2 == 0
        if blink_on:
            # 반투명 빨간 오버레이로 화면 전체를 덮어 깜빡이는 느낌을 준다
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            # addWeighted(A, alpha, B, beta, gamma): 결과 = A*alpha + B*beta + gamma
            # 즉 빨간 오버레이 30% + 원본 프레임 70%를 섞어 "반투명하게 붉게 물든" 효과를 낸다
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 16)  # 두꺼운 빨간 테두리
        # 흰색 두꺼운 글씨 위에 빨간 얇은 글씨를 겹쳐 그려 어떤 배경에서도 잘 보이게 함
        cv2.putText(frame, "!! HAND OUT OF FRAME !!", (w // 2 - 260, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        cv2.putText(frame, "!! HAND OUT OF FRAME !!", (w // 2 - 260, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """OS별 백엔드로 웹캠을 연다."""
    system = platform.system()
    # design.md 규약: 카메라 백엔드는 OS별로만 분기하고, 그 외 경로/셸 로직은 플랫폼 무관하게 작성
    if system == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Darwin":  # macOS
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라 {index}번을 열 수 없습니다. "
            "다른 앱이 카메라를 사용 중인지, OS 카메라 권한이 허용되어 있는지 확인하세요."
        )
    return cap


def draw_landmarks(frame, hand_landmarks) -> None:
    """정규화 좌표(0~1)의 랜드마크를 프레임 위에 그린다."""
    h, w = frame.shape[:2]
    # MediaPipe 좌표는 0~1 정규화 값이므로, 가로/세로 픽셀 크기(w, h)를 곱해 화면 좌표로 환산한다.
    # 21개 관절을 순서(0~20)대로 딕셔너리에 담아 HAND_CONNECTIONS의 인덱스와 맞춘다.
    points = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_landmarks)}
    draw_skeleton(frame, points, HAND_CONNECTIONS, (0, 255, 0), (0, 0, 255))


def main() -> None:
    """랜드마크 추출 → 시각화 루프 (학습/추론과 무관, 검증 전용)."""
    landmarker = create_hand_landmarker()
    cap = open_camera()
    start = time.monotonic()
    prev_time = start
    warning = HandWarning()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            break

        # 셀피 뷰로 좌우 반전 후 MediaPipe 입력용 이미지로 변환
        frame = cv2.flip(frame, 1)
        mp_image = to_mp_image(frame)

        # 비디오 모드는 단조 증가하는 타임스탬프(ms)가 필요 —
        # time.monotonic()은 시스템 시계 변경의 영향을 받지 않는 경과 시간(초)이라
        # (현재 - 시작) * 1000으로 "영상 재생 기준 몇 ms 지났는지"를 얻는다.
        timestamp_ms = int((time.monotonic() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        for hand in result.hand_landmarks:  # 검출된 손(최대 2개)마다 랜드마크 그리기
            draw_landmarks(frame, hand)
        warning.update(result)
        warning.draw(frame)

        # FPS = 1초 ÷ 이번 프레임에 걸린 시간(초). 프레임 간 시간이 0이면(이론상 드묾) 나눗셈 대신 0으로 처리
        now = time.monotonic()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}  hands: {len(result.hand_landmarks)}  (q/ESC: quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Landmarks", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = ESC
            break

    # 창/카메라/네이티브 리소스 정리 (생략 시 다음 실행에서 카메라 점유 문제 발생 가능)
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
