# Last updated: 2026-07-13

"""[역할] 수신호 랜드마크 시퀀스를 라벨별로 모아 학습 데이터(asset/)로 저장하는 수집 스크립트
(자동 연속 수집 + 화면 클릭 버튼 UI). 카메라 앞에서 같은 동작을 여러 번 반복하면, 시퀀스
하나하나가 `src/training/train.py`가 읽어 들일 학습 샘플이 된다.

특징 형식은 여기서 만들지 않고 `src/capture/extractor.py`를 그대로 가져다 쓴다 — 양손 +
상체 포즈 150차원(FEATURE_DIM). 수집·학습·추론이 전부 같은 추출기를 거쳐야 형식이 어긋나지
않기 때문이다 (doc/design.md 참고).

사용법:
    python -m src.dataset.collect --label 멈춰 [--samples 30] [--frames 30]

조작 (화면 우상단 버튼 클릭 또는 키보드):
    START/PAUSE 버튼 (또는 SPACE) — 자동 수집 시작/일시정지.
        시작하면 "READY 카운트다운(기본 1.5초) → REC 녹화(30프레임) → 저장"이
        목표 개수를 채울 때까지 자동 반복된다. 카운트다운 동안 다음 동작을 준비하면 된다.
    QUIT 버튼 (또는 q/ESC) — 종료

품질 보호:
    손이 감지된 프레임이 절반 미만이면 저장하지 않고 SKIP 처리한다.
    (신호없음처럼 손이 없는 게 정상인 라벨은 --idle 로 필터를 끈다)

저장 형식:
    asset/<label>/<타임스탬프>.npy — shape (frames, 150)
"""

import argparse # 커맨드라인 인자(--label, --samples 등) 파싱
import time # 타임스탬프(ms), 저장 파일명에 쓸 현재 시각
from pathlib import Path # OS 무관 경로 처리 (Windows/macOS 공용, design.md 규약)

import cv2 # OpenCV : 카메라 입력, 버튼/텍스트/진행바 그리기, 마우스 클릭 이벤트
import numpy as np # 시퀀스를 (frames, 150) 배열로 쌓고 .npy로 저장하기 위한 수치 연산

# 모델 다운로드·랜드마커 생성·그리기·카메라 열기 등은 여기서 재구현하지 않고 core 모듈
# (extractor.py)에서 그대로 가져다 쓴다 — landmark_viewer.py와 동일한 헬퍼를 공유한다.
from src.capture.extractor import (
    HAND_DIM,
    FeatureExtractor,
    HandWarning,
    draw_detections,
    open_camera,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "asset"
MIN_DETECTED_RATIO = 0.5  # 손 감지 프레임이 이 비율 미만이면 SKIP

WINDOW = "Collect"
BTN_W, BTN_H, BTN_MARGIN = 150, 50, 10  # 버튼 하나의 가로/세로 크기와 버튼 사이 여백(픽셀)



# cv2 창엔 버튼 위젯이 없어서, 사각형을 직접 그리고 마우스 클릭 좌표가 그 사각형 안에
# 들어오는지 매 프레임 검사하는 방식으로 "클릭 가능한 버튼"을 흉내 낸다.
class ButtonBar:
    """창 우상단에 클릭 가능한 버튼을 그리고 클릭 이벤트를 받는다."""

    def __init__(self) -> None:
        self._clicked: str | None = None  # 마지막으로 클릭된 버튼 이름 (pop()으로 소비됨)
        self._rects: dict[str, tuple[int, int, int, int]] = {}  # 버튼 이름 → (x1, y1, x2, y2) 픽셀 사각형

    def attach(self, window: str) -> None:
        """OpenCV의 저수준 마우스 콜백을 이 창에 등록한다 (클릭 감지는 이 콜백을 통해서만 들어옴)."""
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, *_args) -> None:
        # cv2 콜백 시그니처가 고정이라 flags/param은 안 쓰지만 자리만 *_args로 받아둔다.
        if event == cv2.EVENT_LBUTTONDOWN:
            # 클릭 좌표(x, y)가 어느 버튼 사각형 안에 들어오는지 선형 탐색으로 찾는다 (버튼 몇 개뿐이라 충분).
            for name, (x1, y1, x2, y2) in self._rects.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._clicked = name

    def draw(self, frame, buttons: list[tuple[str, str, tuple[int, int, int]]]) -> None:
        """buttons: (이름, 표시 텍스트, BGR 색) 목록. 오른쪽부터 배치된다."""
        self._rects = {}  # 이번 프레임 버튼 배치로 클릭 판정 영역을 매번 새로 채움 (버튼 목록이 프레임마다 바뀔 수 있어서)
        w = frame.shape[1]
        x2 = w - BTN_MARGIN  # 화면 오른쪽 끝에서 시작해 왼쪽으로 하나씩 배치
        for name, label, color in buttons:
            x1 = x2 - BTN_W
            y1, y2 = BTN_MARGIN, BTN_MARGIN + BTN_H
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                frame, label, (x1 + 14, y1 + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            self._rects[name] = (x1, y1, x2, y2)
            x2 = x1 - BTN_MARGIN  # 다음 버튼은 이 버튼 왼쪽에 이어 붙인다

    def pop(self) -> str | None:
        """클릭된 버튼 이름을 반환하고 내부 상태는 비운다 (한 번 클릭 = 한 번만 반응하도록)."""
        clicked, self._clicked = self._clicked, None
        return clicked


def process_frame(cap, extractor: FeatureExtractor, t0: float):
    """프레임 1장을 읽어 검출까지 수행한다. 반환: (frame, 손결과, 포즈결과) 또는 (None,)*3."""
    ok, frame = cap.read()
    if not ok:
        return None, None, None
    frame = cv2.flip(frame, 1)  # 셀피 뷰 (landmark_viewer.py와 동일하게 좌우 반전)
    # 비디오 모드는 단조 증가하는 타임스탬프(ms)가 필요 — t0(수집 시작 시각) 기준 경과 시간
    timestamp_ms = int((time.monotonic() - t0) * 1000)
    hand_result, pose_result = extractor.detect(frame, timestamp_ms)
    return frame, hand_result, pose_result


def show(frame, hand_result, pose_result, text: str, color, bar: ButtonBar, buttons,
         progress: float | None = None,
         warning: HandWarning | None = None) -> str | None:
    """랜드마크·상태 텍스트·버튼을 그려 표시하고, 발생한 동작을 반환한다.

    progress가 주어지면(0~1) 화면 하단에 빨간 진행 바를 그린다 (녹화 진행 표시용).
    warning이 주어지면 손 미감지 상태를 갱신하고 이탈 경고를 그린다.
    반환: 'toggle' (START/PAUSE 버튼 또는 SPACE), 'quit' (QUIT 버튼 또는 q/ESC), None
    """
    draw_detections(frame, hand_result, pose_result)
    if warning is not None:
        warning.update(hand_result)
        warning.draw(frame)
    cv2.putText(frame, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    if progress is not None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 14), (int(w * progress), h), (0, 0, 255), -1)
    bar.draw(frame, buttons)
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    clicked = bar.pop()
    if key in (ord("q"), 27) or clicked == "quit":  # 27 = ESC
        return "quit"
    if key == ord(" ") or clicked == "toggle":
        return "toggle"
    return None


def hand_detected_ratio(seq: np.ndarray) -> float:
    """시퀀스에서 손(앞 126차원 = 왼손 63 + 오른손 63)이 감지된 프레임 비율.

    `seq[:, :HAND_DIM*2] != 0`은 (frames, 126) 크기의 True/False 배열이 되고,
    `.any(axis=1)`로 프레임마다 "손 126차원 중 하나라도 0이 아니면 손이 감지된 프레임"으로 판정한
    (frames,) 불리언 배열을 만든 다음, `.mean()`으로 그 비율(감지된 프레임 수 / 전체 프레임 수)을 낸다.
    """
    return float((seq[:, : HAND_DIM * 2] != 0).any(axis=1).mean())


def record_sequence(cap, extractor, num_frames: int, t0: float,
                    bar: ButtonBar | None = None,
                    status: str = "",
                    warning: HandWarning | None = None) -> tuple[np.ndarray, str | None]:
    """num_frames 프레임 동안 시퀀스를 녹화한다. 반환: (시퀀스 (num_frames, 150), 동작).

    녹화 진행은 프레임 숫자 대신 하단 진행 바로 표시한다 (저장 개수와 혼동 방지).
    """
    buffer = []
    action = None
    while len(buffer) < num_frames:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            continue
        buffer.append(FeatureExtractor.vector(hand_result, pose_result))
        if bar is not None:
            a = show(frame, hand_result, pose_result, f"{status}  REC", (0, 0, 255),
                     bar, [("quit", "QUIT", (0, 0, 180))],
                     progress=len(buffer) / num_frames, warning=warning)
            # 이 루프 도중 QUIT을 눌러도 즉시 끊지 않고, 프레임을 다 채운 뒤(while 조건 종료) 상위
            # main()에 action="quit"을 돌려줘서 거기서 정리하며 빠져나가게 한다. `a or action`은
            # "한 번이라도 quit이 감지되면 이후 매 프레임 None으로 덮어써지지 않게" 유지하는 역할.
            action = a or action
    return np.stack(buffer), action


def countdown(cap, extractor, t0: float, seconds: float, status: str,
              bar: ButtonBar, warning: HandWarning | None = None) -> str | None:
    """다음 녹화 전 준비 카운트다운. 발생한 동작을 반환한다."""
    action = None
    end = time.monotonic() + seconds  # 카운트다운 종료 시각을 미리 계산해 둠
    while time.monotonic() < end:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            continue
        remain = end - time.monotonic()
        a = show(frame, hand_result, pose_result, f"{status}  READY {remain:.1f}s",
                 (0, 200, 255), bar,
                 [("quit", "QUIT", (0, 0, 180)), ("toggle", "PAUSE", (0, 140, 200))],
                 warning=warning)
        action = a or action  # record_sequence와 동일하게 한 번 감지된 동작을 잃지 않도록 누적
        if action == "quit":  # quit은 카운트다운 도중이라도 즉시 빠져나감 (PAUSE와 달리 대기할 이유 없음)
            break
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="수신호 특징 시퀀스 수집 (자동 연속)")
    parser.add_argument("--label", required=True, help="수신호 라벨 (예: 멈춰)")
    parser.add_argument("--samples", type=int, default=30, help="수집할 시퀀스 개수")
    parser.add_argument("--frames", type=int, default=30, help="시퀀스당 프레임 수")
    parser.add_argument("--prep", type=float, default=1.5, help="녹화 전 준비 시간(초)")
    parser.add_argument("--idle", action="store_true",
                        help="손 감지 품질 필터 끄기 (신호없음처럼 손이 없어도 되는 라벨)")
    args = parser.parse_args()

    out_dir = DATA_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.npy")))
    print(f"라벨 '{args.label}' — 기존 {existing}개, 목표 +{args.samples}개")
    print("화면의 START 버튼(또는 SPACE): 자동 수집 시작/일시정지, QUIT(또는 q/ESC): 종료")

    extractor = FeatureExtractor()
    cap = open_camera()
    cv2.namedWindow(WINDOW)
    bar = ButtonBar()
    bar.attach(WINDOW)
    warning = HandWarning(enabled=not args.idle)  # idle 수집은 손이 없는 게 정상

    t0 = time.monotonic()
    saved = 0
    skipped = 0
    collecting = False  # False=대기 화면, True=자동 수집(카운트다운→녹화→저장 반복) 중

    # 상태 머신: collecting이 꺼져 있으면 대기 화면만 반복하고, 켜지면
    # "카운트다운 → 녹화 → (품질 검사 후) 저장" 한 사이클을 목표 개수(args.samples)만큼 반복한다.
    while saved < args.samples:
        status = f"[{args.label}] {existing + saved}/{existing + args.samples}"

        if not collecting:
            # 대기 화면: START 클릭(또는 SPACE)으로 자동 수집 시작
            frame, hand_result, pose_result = process_frame(cap, extractor, t0)
            if frame is None:
                print("프레임을 읽지 못했습니다. 종료합니다.")
                break
            action = show(frame, hand_result, pose_result,
                          status + "  SPACE=start  q=quit", (0, 255, 0), bar,
                          [("quit", "QUIT", (0, 0, 180)), ("toggle", "START", (0, 160, 0))],
                          warning=warning)
            if action == "quit":
                break
            if action == "toggle":
                collecting = True
            continue  # 대기 화면에서는 카운트다운/녹화로 안 내려가고 while 맨 위로 돌아가 다시 그림

        # 자동 수집: 카운트다운 → 녹화 → 저장, 반복
        action = countdown(cap, extractor, t0, args.prep, status, bar, warning)
        if action == "quit":
            break
        if action == "toggle":  # 카운트다운 중 PAUSE를 누르면 녹화까지 가지 않고 대기 화면으로 복귀
            collecting = False
            print("일시정지 — START(또는 SPACE)로 재개")
            continue

        seq, action = record_sequence(cap, extractor, args.frames, t0, bar, status, warning)
        ratio = hand_detected_ratio(seq)
        if not args.idle and ratio < MIN_DETECTED_RATIO:
            skipped += 1  # 손이 잘 안 보인 시퀀스는 저장하지 않고 버림 (저품질 학습 데이터 방지)
            print(f"SKIP (손 감지 {ratio * 100:.0f}% < 50%) — 손을 화면에 보이게 해주세요")
        else:
            # 파일명은 밀리초 타임스탬프 — 같은 라벨로 여러 번 실행해도 겹치지 않고 계속 쌓인다
            path = out_dir / f"{int(time.time() * 1000)}.npy"
            np.save(path, seq)
            saved += 1
            print(f"저장 {saved}/{args.samples}: {path.name}  손 감지 {ratio * 100:.0f}%")
        if action == "quit":  # 녹화 도중 QUIT을 눌렀으면 이번 시퀀스 저장/스킵 처리 후 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"완료 — 저장 {saved}개, 스킵 {skipped}개, 총 {existing + saved}개 ({out_dir})")


if __name__ == "__main__":
    main()
