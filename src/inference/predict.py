# Last updated: 2026-07-27

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
from signal_vision.vision_hand.capture.extractor import (
    HAND_DIM,
    FeatureExtractor,
    HandWarning,
    LensHealthMonitor,
    choose_camera_index,
    draw_detections,
    open_camera,
    process_frame,
)
from signal_vision.vision_hand.inference.ui import Hud # 확률 패널·한글 오버레이·창 표시 전담 (predict.py는 예측 로직만)
from signal_vision.vision_hand.inference.publish import make_publisher
from signal_vision.vision_hand.training.train import SignLSTM

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


class SignalStabilizer:
    '''
    모델 확률을 안정화 필터(임계값 + 연속 일치 + 히스테리시스)에 통과시켜 확정 신호를 관리한다.

    "확정 상태"(confirmed/confirmed_idx)와 그걸 바꾸는 규칙 — 연속 N회 일치해야 확정,
    낮은 confidence는 유예를 두고 해제하되(분진 등 일시적 하락에 기계가 서지 않게)
    렌즈 블러가 실측으로 겹치면 유예 없이 즉시 해제 — 을 한 곳에 모아, main() 루프는
    매 프레임 update()/mark_person_absent() 호출만 하면 되게 한다.
    '''

    def __init__(self, threshold: float, consecutive: int, ema: float, release_grace: float) -> None:
        self.threshold = threshold
        self.consecutive = consecutive
        self.ema = ema
        self.release_grace = release_grace

        self.confirmed = "인식 불가"  # 화면/퍼블리시에 쓰는 현재 확정 신호 (라벨 문자열 또는 안전 기본값)
        self.confirmed_idx: int | None = None  # confirmed에 대응하는 라벨 인덱스 (None = 확정 없음)
        self.emergency = False  # 렌즈 블러가 실측된 상태 = 즉시정지+정비알림 (블러 해제/정상 신호 재확정 시 해제)
        self.status = "버퍼 채우는 중..."
        self.probs: np.ndarray | None = None  # 확률 EMA 상태 (HUD 패널·퍼블리시 confidence에 재사용)
        self.top: int | None = None  # 현재 1위 클래스 인덱스

        self._streak_label: int | None = None  # 직전까지 연속으로 임계값을 넘긴 클래스 인덱스
        self._streak = 0  # streak_label이 연속으로 몇 번째 유지되고 있는지 (--consecutive와 비교)
        self._low_conf_since: float | None = None  # 확정 해제 유예(히스테리시스) 타이머 시작 시각

    def _reset_to_unknown(self, reason: str) -> None:
        '''확정 신호를 안전 기본값(정지)으로 되돌린다. emergency 는 여기서 건드리지
        않는다 — emergency 는 오직 렌즈 블러 실측 여부(is_blurred)로만 결정되므로
        호출부에서 따로 세팅한다.'''
        self._streak_label, self._streak = None, 0
        self._low_conf_since = None
        if self.confirmed != "인식 불가":
            self.confirmed = "인식 불가"
            self.confirmed_idx = None
            print(f"확정: 인식 불가 ({reason}) → 안전 기본값(정지)")

    def mark_person_absent(self, is_blurred: bool) -> None:
        '''윈도우 대부분에서 사람(포즈)이 안 잡힐 때 — 유예 없이 즉시 해제.

        렌즈 블러가 실측되면(is_blurred) 사람 미감지 여부와 무관하게 EMERGENCY 로
        올린다 — 블러 단독으로도 즉시정지+정비알림을 띄우는 정책. 화면이 선명한데
        사람만 없는 건(관절 미검출 포함) 정상 운용이라 emergency 가 아니다.
        '''
        self.probs, self.top = None, None
        self._reset_to_unknown("렌즈 블러 [즉시]" if is_blurred else "사람 미감지")
        self.emergency = is_blurred
        self.status = "렌즈 블러 — 정비 필요" if is_blurred else "사람 미감지"

    def update(self, raw_probs: np.ndarray, labels: list[str], is_blurred: bool) -> None:
        '''실제 예측 확률 하나로 안정화 필터를 한 스텝 진행한다.

        렌즈 블러가 실측되면(is_blurred) 인식 결과와 무관하게 최우선으로
        즉시정지+EMERGENCY 로 처리하고 나머지 안정화 로직은 건너뛴다 — 블러 단독
        트리거 정책. 화면이 선명해지면(is_blurred=False) 아래 정상 경로로 흘러
        emergency 가 자동으로 해제된다.
        '''
        if is_blurred:
            self.probs, self.top = None, None
            self._reset_to_unknown("렌즈 블러 [즉시]")
            self.emergency = True
            self.status = "렌즈 블러 — 정비 필요"
            return

        # 여기부터는 화면이 선명한 정상 경로 — emergency 는 항상 해제 상태로 둔다.
        self.emergency = False
        self.probs = raw_probs if self.probs is None else \
            self.ema * raw_probs + (1 - self.ema) * self.probs
        self.top = int(self.probs.argmax())
        conf = float(self.probs[self.top])
        self.status = f"연속 일치 {min(self._streak, self.consecutive)}/{self.consecutive}"

        # 안정화 필터: 임계값 + 연속 일치 (확정은 깐깐하게)
        if conf >= self.threshold:
            self._low_conf_since = None
            self._streak = self._streak + 1 if self.top == self._streak_label else 1
            self._streak_label = self.top
            if self._streak >= self.consecutive and self.confirmed != labels[self.top]:
                self.confirmed = labels[self.top]
                self.confirmed_idx = self.top
                print(f"확정: {self.confirmed}  (신뢰도 {conf:.2f})")
        else:
            # 해제는 유예를 두고 (히스테리시스) — 일시적 하락에 기계가 서지 않게.
            self._streak_label, self._streak = None, 0
            now = time.monotonic()
            if self.confirmed != "인식 불가":
                if self._low_conf_since is None:
                    self._low_conf_since = now
                remain = self.release_grace - (now - self._low_conf_since)
                if remain > 0:
                    self.status = f"신뢰 회복 대기 {remain:.1f}s (유지: {self.confirmed})"
                else:
                    self._reset_to_unknown("유예 초과")


def main() -> None:
    '''웹캠 → 특징 추출 → 슬라이딩 윈도우 추론 → 안정화 필터 → 퍼블리시 메인 루프.

    프레임마다 (1) 사람이 안 잡히면 즉시 "인식 불가"로 안전 기본값 처리,
    (2) 모델 확률을 EMA로 평활한 뒤 임계값+연속 일치를 만족해야만 신호를 확정,
    (3) 렌즈 블러(김서림 등)가 실측으로 확인되면서 인식까지 흔들리면 유예 없이 즉시 해제,
    (4) 확정 상태가 바뀌면 즉시 + 아니면 1초 하트비트로 계속 퍼블리시한다.

    상태 관리(확정/streak/유예/emergency)는 SignalStabilizer가 전담하고, 이 루프는
    매 프레임 그 객체를 갱신하고 결과를 읽어 퍼블리시·HUD 렌더링만 한다.
    '''
    parser = argparse.ArgumentParser(description="실시간 수신호 추론")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=None,
                        help="사용할 카메라 인덱스 (생략하면 연결된 카메라 중에서 고르는 메뉴가 뜬다)")
    parser.add_argument("--hand-confidence", type=float, default=0.4,
                        help="손 검출 신뢰도 임계값 (기본 0.5). 낮출수록 장갑 등으로 윤곽이 흐릿해도 "
                             "더 잘 잡지만 오검출·지터가 늘어난다")
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
    extractor = FeatureExtractor(hand_confidence=args.hand_confidence)  # 손+포즈 랜드마커 → 150차원 특징 벡터
    cap = open_camera(choose_camera_index(args.camera))
    t0 = time.monotonic()  # VIDEO 모드 타임스탬프(process_frame)의 기준 시각
    last_sent: tuple[str, float] = ("", 0.0)  # (마지막 전송 신호, 전송 시각) — 하트비트 주기 판단용

    window: deque[np.ndarray] = deque(maxlen=num_frames)  # 최근 num_frames개 특징 벡터 (슬라이딩 윈도우)
    warning = HandWarning()  # 손이 연속으로 안 잡히면 화면에 이탈 경고
    lens_monitor = LensHealthMonitor()  # 렌즈 블러(김서림·물방울·분진) 진단
    stabilizer = SignalStabilizer(args.threshold, args.consecutive, args.ema, args.release_grace)

    while True:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            break
        # draw_detections보다 반드시 먼저 측정한다 — 스켈레톤 선/경고 오버레이가 그려진
        # 뒤에 재면 그 인공적인 고대비 선 때문에 선명도가 실측보다 크게 부풀려져(실측
        # 평균 +157%, 최대 +197%) 실제 렌즈 블러(김서림 등)를 못 잡을 수 있다.
        is_blurred = lens_monitor.update(frame)
        draw_detections(frame, hand_result, pose_result)
        warning.update(hand_result)
        warning.draw(frame)

        window.append(FeatureExtractor.vector(hand_result, pose_result))

        if len(window) == num_frames:
            arr = np.stack(window)

            # 안전 가드: 윈도우 대부분에서 사람(포즈)이 안 잡히면 모델 판단을 신뢰하지 않는다
            pose_ratio = float((arr[:, HAND_DIM * 2 :] != 0).any(axis=1).mean())
            if pose_ratio < 0.3:
                stabilizer.mark_person_absent(is_blurred)
            else:
                # --- 실제 예측 한 번 ---
                # arr: (num_frames=30, 150) — PyTorch 모델은 항상 "배치 축"을 기대하므로
                # arr[None]으로 맨 앞에 배치 차원 1개를 추가해 (1, 30, 150)으로 만든다
                # (한 번에 시퀀스 하나만 넣지만, 모델 입력 형태 규약은 지켜야 함).
                # from_numpy: numpy 배열을 메모리 복사 없이 torch 텐서로 감싸고, .to(device)로
                # 모델과 같은 장치(CPU/GPU)로 옮긴다 — 서로 다른 장치의 텐서끼리는 연산 불가.
                x = torch.from_numpy(arr[None]).to(device)
                with torch.no_grad():
                    # 추론만 할 뿐 역전파는 안 하므로 그래디언트 추적을 꺼서 메모리·속도 절약
                    # model(x): SignLSTM의 forward()가 실행되어 클래스별 "로짓"(정규화 안 된 점수)을
                    # (1, num_classes) 모양으로 반환. softmax(dim=1)으로 클래스 축을 따라 합이 1이
                    # 되는 확률로 변환하고(연속 확률 분포), [0]으로 배치 차원을 벗겨 (num_classes,)만 남긴다.
                    # .cpu().numpy(): GPU에 있었을 수도 있는 텐서를 CPU로 내리고 (numpy는 GPU 메모리를 다룰 줄 모름)
                    # numpy 배열로 바꾼 후 (EMA 평활, argmax 등)는 순수 numpy로 다룬다.
                    raw_probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
                stabilizer.update(raw_probs, labels, is_blurred)

        # 확정 상태 전송: 상태가 바뀌면 즉시, 같으면 1초 주기 하트비트
        signal = labels[stabilizer.confirmed_idx] if stabilizer.confirmed_idx is not None else "unknown"
        now = time.monotonic()
        if signal != last_sent[0] or now - last_sent[1] >= 1.0:
            conf_out = float(stabilizer.probs[stabilizer.confirmed_idx]) \
                if (stabilizer.probs is not None and stabilizer.confirmed_idx is not None) else 0.0
            publisher.publish(signal, conf_out)
            last_sent = (signal, now)

        header_color = (0, 220, 0) if stabilizer.confirmed_idx is not None else (80, 80, 255)
        if hud.render(frame, labels, stabilizer.probs, stabilizer.top, stabilizer.confirmed_idx, args.threshold,
                      header=f"확정: {stabilizer.confirmed}", header_color=header_color, status=stabilizer.status,
                      emergency=stabilizer.emergency):
            break

    cap.release()
    hud.close()
    extractor.close()
    publisher.close()


if __name__ == "__main__":
    main()
