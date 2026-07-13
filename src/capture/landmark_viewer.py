# Last updated: 2026-07-13

"""[역할] 웹캠 영상에서 MediaPipe HandLandmarker(손) + PoseLandmarker(상체 포즈) 랜드마크를
실시간으로 함께 그려 보여주는 검증용 스크립트. 카메라 → 손+포즈 랜드마크 추출 파이프라인이
정상 동작하는지 눈으로 확인하는 용도이며, 그 자체로 학습이나 예측을 하지는 않는다.

실행: F5 또는 VSCode 태스크 "실행: 랜드마크 시각화" (종료: q 또는 ESC)

[손 + 포즈를 같이 보는 이유] 실제 수신호 인식은 손(왼손 63 + 오른손 63) + 상체 포즈(24) =
150차원 특징을 함께 쓴다(src/capture/extractor.py의 FeatureExtractor, 설계 근거는
doc/design.md 참고). 이 뷰어도 같은 두 랜드마커를 그대로 써서, 실제 특징 추출이 보는 것과
동일한 화면을 검증 단계에서 확인할 수 있게 한다.

[사용 라이브러리와 이유]
- OpenCV(cv2): 카메라 입력·프레임 표시/그리기. OS 간 웹캠 접근을 통일해 주는 사실상 표준 라이브러리.
- MediaPipe: 구글이 만든, 이미지/영상에서 사람의 손·몸·얼굴 등을 인식하는 미리 학습된(pretrained)
  ML 모델 + 실행 파이프라인 모음. "카메라 프레임 → 관절 좌표" 변환을 직접 딥러닝 모델을 만들어
  학습시키지 않고도 라이브러리 호출 몇 줄로 끝낼 수 있게 해준다. Tasks API(HandLandmarker +
  PoseLandmarker)를 써서 손 21관절 + 상체 8관절(x, y, z) 랜드마크를 추출한다.
  mediapipe 0.10.35부터 구형 solutions API가 제거되어 Tasks API만 지원한다.
  최초 실행 시 모델 파일(hand_landmarker.task·pose_landmarker.task)을 models/ 폴더에 자동 다운로드한다.
  모델 경로/다운로드/랜드마커 생성/그리기 헬퍼는 src/capture/extractor.py에 공용으로 정의되어
  있고, 이 파일은 거기서 가져다 쓴다 (뷰어가 core를 참조하는 방향).

[처리 흐름]
카메라 프레임(BGR) 캡처 → 좌우 반전(셀피 뷰) → RGB 변환 후 mp.Image로 래핑
→ HandLandmarker.detect_for_video() / PoseLandmarker.detect_for_video()로 같은 프레임에서
  손(최대 2개) 21관절 + 상체 8관절을 각각 검출
→ 정규화 좌표(0~1)를 프레임 픽셀 크기로 환산해 손은 초록/빨강, 포즈는 하늘색/주황 뼈대로 그림
→ 연속 미검출 시 경고 오버레이(HandWarning, 손 기준), 화면에 FPS 표시

[전체 파이프라인의 학습/예측 알고리즘] 이 뷰어 자체는 분류를 하지 않지만, 참고로 전체
파이프라인은 30프레임 랜드마크 시퀀스를 PyTorch로 구현한 LSTM(2층, hidden 64) 시퀀스
분류 모델로 학습·추론한다 — 수신호는 정지 자세가 아니라 시간에 따른 궤적(동적 제스처)이라
시계열을 다루는 RNN 계열을 선택했다 (근거: doc/ml-flow.md 3절, doc/design.md).
"""

import platform # OS(Windows/macOS/Linux) 판별 — 카메라 백엔드를 OS별로 분기하기 위함
import time # 프레임 타임스탬프(ms), FPS 계산, 경고 깜빡임 주기 계산

import cv2 # OpenCV : 카메라 입력, 프레임 그리기/표시

from src.capture.extractor import (
    HAND_CONNECTIONS,
    create_hand_landmarker,
    create_pose_landmarker,
    draw_pose,
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
    hand_landmarker = create_hand_landmarker()
    pose_landmarker = create_pose_landmarker()
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
        # extractor.FeatureExtractor.detect()와 동일하게, 같은 프레임·타임스탬프를 손/포즈
        # 두 랜드마커에 각각 넣어 검출한다 (실제 특징 추출이 보는 것과 같은 화면을 확인하려는 목적).
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        for hand in hand_result.hand_landmarks:  # 검출된 손(최대 2개)마다 랜드마크 그리기
            draw_landmarks(frame, hand)
        draw_pose(frame, pose_result)  # 상체 포즈(어깨·팔꿈치·손목·골반) 그리기
        warning.update(hand_result)
        warning.draw(frame)

        # FPS = 1초 ÷ 이번 프레임에 걸린 시간(초). 프레임 간 시간이 0이면(이론상 드묾) 나눗셈 대신 0으로 처리
        now = time.monotonic()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        pose_detected = bool(pose_result.pose_landmarks)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}  hands: {len(hand_result.hand_landmarks)}  "
            f"pose: {'O' if pose_detected else 'X'}  (q/ESC: quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand + Pose Landmarks", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = ESC
            break

    # 창/카메라/네이티브 리소스 정리 (생략 시 다음 실행에서 카메라 점유 문제 발생 가능)
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()


if __name__ == "__main__":
    main()
