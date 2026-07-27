# Last updated: 2026-07-27

'''
    collect.py
    
    수신호 랜드마크 시퀀스를 라벨별로 모아 학습 데이터(asset/)로 저장하는 수집 스크립트
    (자동 연속 수집 + 화면 클릭 버튼 UI). 카메라 앞에서 같은 동작을 여러 번 반복하면, 시퀀스
    하나하나가 `src/training/train.py`가 읽어 들일 학습 샘플이 된다.

    조작 (화면 우상단 버튼 클릭 또는 키보드):
        START/PAUSE 버튼 (또는 SPACE) — 자동 수집 시작/일시정지.
            시작하면 "READY 카운트다운(기본 1.5초) → REC 녹화(30프레임) → 저장"이
            목표 개수를 채울 때까지 자동 반복된다. 카운트다운 동안 다음 동작을 준비하면 된다.
        QUIT 버튼 (또는 q/ESC) — 종료

    품질 보호:
        손이 감지된 프레임이 절반 미만이면 저장하지 않고 SKIP 처리한다.
        (신호없음처럼 손이 없는 게 정상인 라벨은 --idle 로 필터를 끈다)

    저장 형식: asset/<label>/<타임스탬프>.npy — shape (frames, 150)
'''

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# core 모듈 landmark_viewer.py와 동일한 헬퍼를 공유.
from src.capture.extractor import (
    HAND_DIM,
    FeatureExtractor,
    HandWarning,
    open_camera,
    process_frame,
)
from src.dataset.ui import WINDOW, ButtonBar, show

DATA_DIR = Path(__file__).resolve().parents[2] / "asset"
MIN_DETECTED_RATIO = 0.5  # 손 감지 프레임이 이 비율 미만이면 SKIP

KNOWN_LABELS = ["stop", "slow", "come", "back", "left_go", "right_go", "idle"] # 신호수집 라벨

def choose_label(preset: str | None) -> str:
    '''label을 줬으면 그대로 쓰되, 알려진 목록(KNOWN_LABELS)에 없으면 오타일 수 있다고
    경고만 하고 진행한다 (새 신호 추가를 막지 않기 위해 차단하지는 않는다).
    '''
    if preset is not None:
        if preset not in KNOWN_LABELS:
            print(f"⚠ '{preset}'은 알려진 라벨 목록에 없습니다 — 오타가 아니라면 새 신호로 그대로 진행합니다")
        return preset

    print("라벨을 선택하세요:")
    for i, name in enumerate(KNOWN_LABELS, 1):
        print(f"  {i}. {name}")
    print(f"  {len(KNOWN_LABELS) + 1}. 직접 입력 (새 신호)")
    while True:
        choice = input("번호 또는 라벨명 입력: ").strip()
        if not choice:
            print("빈 값은 사용할 수 없습니다.")
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(KNOWN_LABELS):
                return KNOWN_LABELS[idx - 1]
            if idx == len(KNOWN_LABELS) + 1:
                custom = input("새 라벨 이름: ").strip()
                if custom:
                    return custom
                print("빈 값은 사용할 수 없습니다.")
                continue
            print("목록에 없는 번호입니다.")
            continue
        # 번호가 아니라 라벨명을 직접 타이핑한 경우 — 목록에 없어도 새 신호로 그대로 받아준다
        if choice not in KNOWN_LABELS:
            print(f"⚠ '{choice}'은 알려진 라벨 목록에 없습니다 — 오타가 아니라면 새 신호로 그대로 진행합니다")
        return choice


def hand_detected_ratio(seq: np.ndarray) -> float:
    '''
    시퀀스에서 손(왼손 63 + 오른손 63)이 감지된 프레임 비율.

    `seq[:, :HAND_DIM*2] != 0`은 (frames, 126) 크기의 True/False 배열이 되고,
    `.any(axis=1)`로 프레임마다 "손 126차원 중 하나라도 0이 아니면 손이 감지된 프레임"으로 판정한
    (frames,) 불리언 배열을 만든 다음, `.mean()`으로 그 비율(감지된 프레임 수 / 전체 프레임 수)을 낸다.
    '''
    return float((seq[:, : HAND_DIM * 2] != 0).any(axis=1).mean())


def record_sequence(cap, extractor, num_frames: int, t0: float,
                    bar: ButtonBar | None = None,
                    status: str = "",
                    warning: HandWarning | None = None) -> tuple[np.ndarray, str | None]:
    '''
    num_frames 프레임 동안 시퀀스를 녹화한다. 반환: (시퀀스 (num_frames, 150), 동작).
    녹화 진행은 프레임 숫자 대신 하단 진행 바로 표시한다 (저장 개수와 혼동 방지).
    '''
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
            action = a or action  # quit 감지돼도 즉시 끊지 않고, 프레임을 다 채운 뒤 main()에서 종료 처리
    return np.stack(buffer), action


def countdown(cap, extractor, t0: float, seconds: float, status: str,
              bar: ButtonBar, warning: HandWarning | None = None) -> str | None:
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
    parser.add_argument("--label", default=None,
                        help="수신호 라벨 (예: stop). 생략하면 번호로 고르는 메뉴가 뜬다")
    parser.add_argument("--samples", type=int, default=30, help="수집할 시퀀스 개수")
    parser.add_argument("--frames", type=int, default=30, help="시퀀스당 프레임 수")
    parser.add_argument("--prep", type=float, default=1.5, help="녹화 전 준비 시간(초)")
    parser.add_argument("--idle", action="store_true",
                        help="손 감지 품질 필터 끄기 (신호없음처럼 손이 없어도 되는 라벨)")
    args = parser.parse_args()

    label = choose_label(args.label)
    out_dir = DATA_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.npy")))
    print(f"라벨 '{label}' — 기존 {existing}개, 목표 +{args.samples}개")
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
        status = f"[{label}] {existing + saved}/{existing + args.samples}"

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
