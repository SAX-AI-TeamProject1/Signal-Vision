# Last updated: 2026-07-27

'''
    [역할] 
    웹캠 영상에서 MediaPipe HandLandmarker(손) + PoseLandmarker(상체 포즈) 랜드마크를
    실시간으로 함께 그려 보여주는 검증용 스크립트. 카메라 → 손+포즈 랜드마크 추출 파이프라인이
    정상 동작하는지확인하는 용도.

    [사용 라이브러리와 이유]
    - OpenCV(cv2): 카메라 입력·프레임 표시/그리기. OS 간 웹캠 접근을 통일해 주는 사실상 표준 라이브러리.
    - MediaPipe: 구글이 만든, 이미지/영상에서 사람의 손·몸·얼굴 등을 인식하는 미리 학습된(pretrained)
    ML 모델 + 실행 파이프라인 모음. "카메라 프레임 → 관절 좌표" 변환을 직접 딥러닝 모델을 만들어
    학습시키지 않고도 라이브러리 호출 몇 줄로 끝낼 수 있게 해준다.

    [흐름]
    카메라 프레임(BGR) 캡처 → 좌우 반전(셀피 뷰) → RGB 변환 후 mp.Image로 래핑
    → HandLandmarker.detect_for_video() / PoseLandmarker.detect_for_video()로 같은 프레임에서
    손(최대 2개) 21관절 + 상체 8관절을 각각 검출
    → extractor.draw_detections()로 손은 초록/빨강, 포즈는 하늘색/주황 뼈대로 그림
    → 연속 미검출 시 경고 오버레이(HandWarning, 손 기준), 화면에 FPS 표시
'''

import time # 프레임 타임스탬프(ms), FPS 계산
import cv2 # OpenCV : 카메라 입력, 프레임 그리기/표시

from src.capture.extractor import (
    HandWarning,
    create_hand_landmarker,
    create_pose_landmarker,
    draw_detections,
    open_camera,
    to_mp_image,
)
from src.console import banner, step


def main() -> None:
    step("카메라/모델 초기화 중 (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있음)")
    hand_landmarker = create_hand_landmarker()
    pose_landmarker = create_pose_landmarker()
    cap = open_camera()
    start = time.monotonic()
    prev_time = start
    warning = HandWarning()
    ok_exit = True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            ok_exit = False
            break

        # 셀피 뷰로 좌우 반전 후 MediaPipe 입력용 이미지로 변환
        frame = cv2.flip(frame, 1)
        mp_image = to_mp_image(frame)

        # 비디오 모드는 단조 증가하는 타임스탬프(ms)가 필요
        # time.monotonic()은 시스템 시계 변경의 영향을 받지 않는 경과 시간(초)이라
        # (현재 - 시작) * 1000으로 "영상 재생 기준 몇 ms 지났는지"를 얻는다.
        timestamp_ms = int((time.monotonic() - start) * 1000)
        # extractor.FeatureExtractor.detect()와 동일하게, 같은 프레임·타임스탬프를 손/포즈
        # 두 랜드마커에 각각 넣어 검출한다 (실제 특징 추출이 보는 것과 같은 화면을 확인하려는 목적).
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        draw_detections(frame, hand_result, pose_result)  # 손 랜드마크 + 상체 포즈 뼈대 그리기
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

    # 창/카메라/네이티브 리소스 정리
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()
    banner(ok_exit, "정상 종료" if ok_exit else "카메라 프레임 읽기 실패로 종료")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        banner(False, f"초기화/실행 중 오류 발생: {exc}")
        raise
