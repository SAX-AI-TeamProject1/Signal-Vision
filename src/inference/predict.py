# Last updated: 2026-07-14

'''
실시간 수신호 추론

최근 30프레임 랜드마크를 슬라이딩 윈도우로 유지하며 매 프레임 분류하고, 
안정화 필터(신회도 임계값 + 연속 k회 동일 예측)을 통과한 신호만 확정.

확정 신호가 없다면 '인식 불가' - 안전 기본 값 정지

종료 q 또는 ESC

'''

import argparse # 커맨드라인 인자(--threshold, --consecutive 등) 파싱
import time # 타임스탬프(ms) 계산, 하트비트/유예 타이머 기준 시각
from collections import deque # 최근 num_frames개만 유지하는 슬라이딩 윈도우 버퍼
from pathlib import Path # OS 무관 경로 처리 (Windows/macOS 공용, design.md 규약)

import numpy as np # 윈도우 버퍼를 (num_frames, 150) 배열로 쌓고 안전 가드 비율 계산
import torch # 모델 로드·추론 (LSTM 분류기)

# 랜드마크 검출·프레임 처리는 여기서 재구현하지 않고 core 모듈(extractor.py)에서 그대로
# 가져다 쓴다 — collect.py·landmark_viewer.py와 동일한 헬퍼를 공유한다.
from src.capture.extractor import (
    HAND_DIM,
    FeatureExtractor,
    HandWarning,
    draw_detections,
    open_camera,
    process_frame,
)
from src.inference.hud import Hud # 확률 패널·한글 오버레이·창 표시 전담 (predict.py는 예측 로직만)
from src.inference.bridge import make_publisher
from src.training.train import SignLSTM

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "sign_classifier.pt"  # train.py가 저장하는 학습된 LSTM 가중치


def load_model(model_path: Path, device):
    '''체크포인트에서 모델·라벨 목록·시퀀스 길이를 복원한다.

    train.py가 저장한 .pt 파일은 가중치 텐서 하나가 아니라
    {"state_dict": ..., "config": {...}, "labels": [...]} 형태의 딕셔너리를 통째로 담고
    있다. 그래서 (1) 그 딕셔너리를 읽고(torch.load), (2) config에 적힌 구조(입력 차원 150·
    hidden 크기·LSTM 레이어 수)로 아직 학습되지 않은 빈 SignLSTM을 만든 뒤, (3) 그 안에
    state_dict(레이어별로 학습된 가중치 값)를 채워 넣는(load_state_dict) 순서로 복원한다.

    라벨은 코드에 하드코딩하지 않고 체크포인트(ckpt["labels"])에서 읽는다 — 학습 시점에
    asset/ 폴더를 스캔해 정해진 라벨 순서를 그대로 따라야 model(x)의 클래스 인덱스가 맞는다.
    '''
    # map_location=device: 학습은 GPU에서 했더라도, 지금 이 device(예: GPU 없는 맥이면 cpu)로
    # 텐서를 바로 올려서 로드한다 (안 하면 학습 때 쓴 장치가 없을 때 오류가 남)
    # weights_only=False: 가중치 텐서 외에 config/labels 같은 일반 파이썬 객체도 같이
    # 들어있는 체크포인트라 필요 (PyTorch 최신 버전은 기본값이 True라 명시해야 함)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SignLSTM(cfg["input_dim"], cfg["hidden_dim"], len(ckpt["labels"]),
                     num_layers=cfg["num_layers"]).to(device)  # .to(device): 모델 파라미터를 CPU/GPU 메모리로 옮김
    model.load_state_dict(ckpt["state_dict"])  # 레이어별 학습된 가중치 값을 방금 만든 빈 모델 구조에 채움
    model.eval()  # 추론 전용 모드 — 학습 때만 쓰는 드롭아웃 등을 끄고 항상 같은 출력이 나오게 고정
    return model, ckpt["labels"], cfg["frames"]


def main() -> None:
    '''웹캠 → 특징 추출 → 슬라이딩 윈도우 추론 → 안정화 필터 → 퍼블리시 메인 루프.

    프레임마다 (1) 사람이 안 잡히면 즉시 "인식 불가"로 안전 기본값 처리,
    (2) 모델 확률을 EMA로 평활한 뒤 임계값+연속 일치를 만족해야만 신호를 확정,
    (3) 확정 상태가 바뀌면 즉시 + 아니면 1초 하트비트로 계속 퍼블리시한다.
    '''
    parser = argparse.ArgumentParser(description="실시간 수신호 추론")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="확정에 필요한 최소 신뢰도")
    parser.add_argument("--consecutive", type=int, default=5,
                        help="확정에 필요한 연속 동일 예측 횟수")
    parser.add_argument("--ema", type=float, default=0.4,
                        help="확률 지수 평활 계수 (1.0 = 평활 끔) — 스파이크 노이즈 방어")
    parser.add_argument("--release-grace", type=float, default=1.0,
                        help="확정 해제 유예 시간(초) — 일시적 신뢰도 하락에 기계가 서지 않게")
    parser.add_argument("--publish", choices=["none", "udp", "rosbridge"], default="none",
                        help="확정 신호 외부 전달 방식 (기계 프로젝트 연동)")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5555)
    parser.add_argument("--rosbridge-url", default="ws://localhost:9090")
    parser.add_argument("--ros-topic", default="/hand_signal")
    args = parser.parse_args()

    # GPU(cuda)가 있으면 그쪽에서 추론(빠름), 없으면 CPU로 폴백 — 이후 모델과 입력 텐서를
    # 전부 이 device로 옮겨야(.to(device)) 서로 같은 메모리 공간에서 연산할 수 있다
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, num_frames = load_model(args.model, device)
    print(f"모델 로드: {args.model.name}  라벨: {labels}")
    print(f"안정화: 신뢰도 ≥ {args.threshold}, 연속 {args.consecutive}회 일치 시 확정")

    publisher = make_publisher(args.publish, udp_host=args.udp_host, udp_port=args.udp_port,
                               rosbridge_url=args.rosbridge_url, ros_topic=args.ros_topic)
    hud = Hud()  # 확률 패널·텍스트·창 표시 전담 (predict.py는 이 객체에만 그리기를 위임)
    extractor = FeatureExtractor()  # 손+포즈 랜드마커 → 150차원 특징 벡터
    cap = open_camera()
    t0 = time.monotonic()  # VIDEO 모드 타임스탬프(process_frame)의 기준 시각
    last_sent: tuple[str, float] = ("", 0.0)  # (마지막 전송 신호, 전송 시각) — 하트비트 주기 판단용

    window: deque[np.ndarray] = deque(maxlen=num_frames)  # 최근 num_frames개 특징 벡터 (슬라이딩 윈도우)
    streak_label: int | None = None  # 직전까지 연속으로 임계값을 넘긴 클래스 인덱스
    streak = 0  # streak_label이 연속으로 몇 번째 유지되고 있는지 (--consecutive와 비교)
    confirmed = "인식 불가"  # 화면/퍼블리시에 쓰는 현재 확정 신호 (라벨 문자열 또는 안전 기본값)
    confirmed_idx: int | None = None  # confirmed에 대응하는 라벨 인덱스 (None = 확정 없음)
    warning = HandWarning()  # 손이 연속으로 안 잡히면 화면에 이탈 경고
    smoothed: np.ndarray | None = None   # 확률 EMA 상태 (프레임 간 스파이크 완화, None=아직 초기화 전)
    low_conf_since: float | None = None  # 확정 해제 유예(히스테리시스) 타이머 시작 시각, None=유예 중 아님

    while True:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            break
        draw_detections(frame, hand_result, pose_result)
        warning.update(hand_result)
        warning.draw(frame)
        window.append(FeatureExtractor.vector(hand_result, pose_result))

        probs, top = None, None
        status = "버퍼 채우는 중..."
        if len(window) == num_frames:
            arr = np.stack(window)

            # 안전 가드: 윈도우 대부분에서 사람(포즈)이 안 잡히면 모델 판단을 신뢰하지 않는다
            pose_ratio = float((arr[:, HAND_DIM * 2 :] != 0).any(axis=1).mean())
            if pose_ratio < 0.3:
                streak_label, streak = None, 0
                smoothed, low_conf_since = None, None  # 사람이 사라지면 평활·유예 상태도 초기화
                if confirmed != "인식 불가":
                    confirmed = "인식 불가"
                    confirmed_idx = None
                    print("확정: 인식 불가 (사람 미감지) → 안전 기본값(정지)")
                now = time.monotonic()
                if last_sent[0] != "unknown" or now - last_sent[1] >= 1.0:
                    publisher.publish("unknown", 0.0)
                    last_sent = ("unknown", now)
                if hud.render(frame, labels, None, None, None, args.threshold,
                              header="확정: 인식 불가", header_color=(80, 80, 255),
                              status="사람 미감지", status_color=(80, 80, 255)):
                    break
                continue

            # --- 실제 예측 한 번 ---
            # arr: (num_frames=30, 150) — PyTorch 모델은 항상 "배치 축"을 기대하므로
            # arr[None]으로 맨 앞에 배치 차원 1개를 추가해 (1, 30, 150)으로 만든다
            # (한 번에 시퀀스 하나만 넣지만, 모델 입력 형태 규약은 지켜야 함).
            # from_numpy: numpy 배열을 메모리 복사 없이 torch 텐서로 감싸고, .to(device)로
            # 모델과 같은 장치(CPU/GPU)로 옮긴다 — 서로 다른 장치의 텐서끼리는 연산 불가.
            x = torch.from_numpy(arr[None]).to(device)
            with torch.no_grad():  
                # 추론만 할 뿐 역전파 안 하므로 그래디언트 추적을 꺼서 메모리·속도 절약
                # model(x): SignLSTM의 forward()가 실행되어 클래스별 "로짓"(정규화 안 된 점수)을
                # (1, num_classes) 모양으로 반환. softmax(dim=1)으로 클래스 축을 따라 합이 1이
                # 되는 확률로 변환하고(연속 확률 분포), [0]으로 배치 차원을 벗겨 (num_classes,)만 남긴다.
                # .cpu().numpy(): GPU에 있었을 수도 있는 텐서를 CPU로 내리고 (numpy는 GPU 메모리를 다룰 줄 모름)
                # numpy 배열로 바꾼 후 (EMA 평활, argmax 등)는 순수 numpy로 다룬다.
                raw_probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            # 확률 EMA 평활: 1프레임 스파이크가 판단에 직접 튀지 않게
            smoothed = raw_probs if smoothed is None else \
                args.ema * raw_probs + (1 - args.ema) * smoothed
            probs = smoothed
            top = int(probs.argmax())
            conf = float(probs[top])
            status = f"연속 일치 {min(streak, args.consecutive)}/{args.consecutive}"

            # 안정화 필터: 임계값 + 연속 일치 (확정은 깐깐하게)
            if conf >= args.threshold:
                low_conf_since = None
                streak = streak + 1 if top == streak_label else 1
                streak_label = top
                if streak >= args.consecutive and confirmed != labels[top]:
                    confirmed = labels[top]
                    confirmed_idx = top
                    print(f"확정: {confirmed}  (신뢰도 {conf:.2f})")
            else:
                # 해제는 유예를 두고 (히스테리시스) — 일시적 하락에 기계가 서지 않게.
                # 사람 미감지 가드(위)는 이 유예와 무관하게 즉시 해제된다.
                streak_label, streak = None, 0
                now = time.monotonic()
                if confirmed != "인식 불가":
                    if low_conf_since is None:
                        low_conf_since = now
                    remain = args.release_grace - (now - low_conf_since)
                    if remain > 0:
                        status = f"신뢰 회복 대기 {remain:.1f}s (유지: {confirmed})"
                    else:
                        confirmed = "인식 불가"
                        confirmed_idx = None
                        low_conf_since = None
                        print("확정: 인식 불가 (유예 초과) → 안전 기본값(정지)")

        # 확정 상태 전송: 상태가 바뀌면 즉시, 같으면 1초 주기 하트비트
        signal = labels[confirmed_idx] if confirmed_idx is not None else "unknown"
        now = time.monotonic()
        if signal != last_sent[0] or now - last_sent[1] >= 1.0:
            conf_out = float(probs[confirmed_idx]) if (probs is not None and confirmed_idx is not None) else 0.0
            publisher.publish(signal, conf_out)
            last_sent = (signal, now)

        header_color = (0, 220, 0) if confirmed_idx is not None else (80, 80, 255)
        if hud.render(frame, labels, probs, top, confirmed_idx, args.threshold,
                      header=f"확정: {confirmed}", header_color=header_color, status=status):
            break

    cap.release()
    hud.close()
    extractor.close()
    publisher.close()


if __name__ == "__main__":
    main()

