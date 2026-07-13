# Last updated: 2026-07-13

"""[역할] 외부 공개 데이터셋인 Jester → 우리 학습 데이터(asset/) 형식으로 변환하는 스크립트
(doc/ml-flow.md 1-2절). **현재는 학습에 실제로 쓰이지 않는다** — Jester로 학습해봤더니
실제 웹캠에서는 예측이 크게 빗나가는 도메인 갭이 확인되어(동작 방식 차이, 셀피 뷰 좌우 반전
문제) 자체 수집(dataset/collect.py) 단독으로 전환했다. 그 실험을 재현하거나 향후 사전학습에
활용할 수 있도록 변환 도구로만 유지한다 (doc/design.md, doc/ml-flow.md 1-2절 참고).

Jester(20BN-Jester)는 웹캠 3인칭 동적 제스처 영상 데이터셋으로, 클립이
프레임 JPG 폴더(12fps, 약 36프레임)로 제공된다. 이 스크립트는 수신호에
해당하는 클래스만 골라 손+포즈 특징을 추출하고, 우리 학습 형식
asset/<라벨>/*.npy (30, 150) 으로 저장한다 (src/capture/extractor.py 참고).

사용법:
    python -m src.dataset.import_jester \
        --jester-dir <20bn-jester-v1 폴더> --labels-csv <jester-v1-train.csv> \
        [--max-per-class 100] [--out asset]

CSV 형식: 원본 "video_id;label" 또는 Kaggle 미러 "video_id,label,..." 모두 지원

클래스 매핑(기본값, --mapping 으로 JSON 파일 지정 가능):
    Stop Sign / Pushing Hand Away → 멈춰   (혼동 쌍 + 기계 행동 동일로 통합)
    Pulling Hand In               → 이리와
    Swiping Left                  → 왼쪽으로
    Swiping Right                 → 오른쪽으로
    No gesture / Doing other things → 신호없음

품질 보호: 손 감지 프레임이 절반 미만인 클립은 건너뛴다 (collect.py와 동일 기준 —
MIN_DETECTED_RATIO/hand_detected_ratio를 재구현하지 않고 그대로 가져다 쓴다).
"""

import argparse # 커맨드라인 인자(--jester-dir, --labels-csv 등) 파싱
import csv # Jester 라벨 CSV(video_id;label 또는 Kaggle 미러 형식) 읽기
import json # --mapping으로 커스텀 클래스 매핑 JSON을 넘길 때 파싱
import random # 라벨별 video_id 목록을 --max-per-class만큼 자르기 전에 섞기 위함
import time # 변환 소요 시간 측정, 클립 간 타임스탬프 계산
from pathlib import Path # OS 무관 경로 처리 (Windows/macOS 공용, design.md 규약)

import cv2 # Jester 클립의 JPG 프레임을 읽기 위함 (imread)
import numpy as np # 시퀀스를 (frames, 150) 배열로 다루고 리샘플링·저장하기 위한 수치 연산

from src.capture.extractor import FeatureExtractor
# collect.py의 품질 기준을 재구현하지 않고 그대로 재사용 — 자체 수집과 Jester 변환이
# "손 감지 부족 시 SKIP" 기준을 다르게 적용하면 두 데이터 소스의 품질이 어긋난다.
from src.dataset.collect import MIN_DETECTED_RATIO, hand_detected_ratio

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "asset"
TARGET_FRAMES = 30
FRAME_INTERVAL_MS = 83  # Jester 12fps
VIDEO_GAP_MS = 1000  # 클립 사이 타임스탬프 간격 (트래킹 연결 끊기용)

DEFAULT_MAPPING = {
    "Stop Sign": "멈춰",
    "Pushing Hand Away": "멈춰",  # '오지마'는 '멈춰'와 혼동 쌍 + 기계 행동이 같아 통합
    "Pulling Hand In": "이리와",
    "Swiping Left": "왼쪽으로",
    "Swiping Right": "오른쪽으로",
    "No gesture": "신호없음",
    "Doing other things": "신호없음",
}


def resample(seq: np.ndarray, target: int) -> np.ndarray:
    """시퀀스를 시간 축 선형 보간으로 target 프레임으로 리샘플링한다.

    Jester 클립은 12fps라 프레임 수가 클립마다 제각각(약 36개 안팎)인데, 우리 학습 형식은
    프레임 수가 정확히 30으로 고정(TARGET_FRAMES)이어야 한다. 그래서 원본 시퀀스 길이를
    0~1 구간에 균등 배치한 시간축(t_orig)과, 목표 길이(target)를 0~1에 균등 배치한 시간축
    (t_new)을 만든 뒤, 각 특징 차원(열)마다 `np.interp`로 t_new 시점의 값을 선형 보간해서
    "원본 동작의 타이밍 비율은 유지한 채" 프레임 수만 늘리거나 줄인다.
    """
    if len(seq) == target:  # 이미 목표 길이면 보간 없이 타입만 맞춰 반환
        return seq.astype(np.float32)
    t_orig = np.linspace(0.0, 1.0, len(seq))  # 원본 프레임 각각의 "진행률"(0=시작 ~ 1=끝)
    t_new = np.linspace(0.0, 1.0, target)  # 목표 프레임 각각의 진행률
    out = np.empty((target, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):  # 특징 차원(150개) 하나하나를 독립된 시계열로 보고 보간
        out[:, j] = np.interp(t_new, t_orig, seq[:, j])
    return out


def extract_video(extractor: FeatureExtractor, video_dir: Path, ts_ms: int) -> tuple[np.ndarray | None, int]:
    """클립 폴더의 프레임들에서 특징 시퀀스를 추출한다. 반환: (시퀀스 또는 None, 다음 타임스탬프).

    ts_ms를 호출자에게 돌려주고 다음 호출에 그대로 넘기게 한 이유: FeatureExtractor는 VIDEO 모드라
    호출마다 타임스탬프가 단조 증가해야 한다(MediaPipe 내부 트래킹 요구사항). 그래서 여러 클립을
    잇달아 처리해도 전체적으로 계속 증가하는 하나의 타임라인처럼 타임스탬프를 이어 붙인다.
    """
    frames = sorted(video_dir.glob("*.jpg"))
    if len(frames) < 5:  # 프레임이 너무 적은 클립은 의미 있는 시퀀스를 못 만드므로 건너뜀
        return None, ts_ms
    vecs = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        hand_result, pose_result = extractor.detect(img, ts_ms)
        vecs.append(FeatureExtractor.vector(hand_result, pose_result))
        ts_ms += FRAME_INTERVAL_MS  # Jester 12fps 기준 프레임 간 간격만큼 타임스탬프 전진
    # 클립이 끝나면 큰 간격(VIDEO_GAP_MS)을 더해, 다음 클립의 첫 프레임이 이전 클립의 마지막
    # 프레임과 "이어지는 동작"으로 트래킹되지 않게 끊어준다 (서로 무관한 클립이므로).
    ts_ms += VIDEO_GAP_MS
    if not vecs:
        return None, ts_ms
    return np.stack(vecs), ts_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Jester → asset/ 변환")
    parser.add_argument("--jester-dir", type=Path, required=True,
                        help="클립 폴더들이 들어있는 디렉토리 (20bn-jester-v1)")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="video_id;label 형식 CSV (jester-v1-train.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-class", type=int, default=100,
                        help="우리 라벨당 최대 변환 개수")
    parser.add_argument("--mapping", type=Path, default=None,
                        help="Jester 라벨 → 우리 라벨 매핑 JSON (기본 매핑 대체)")
    parser.add_argument("--idle-labels", nargs="*", default=["신호없음"],
                        help="손 감지 품질 필터를 면제할 라벨 (손이 없는 게 정상인 클래스)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mapping = DEFAULT_MAPPING
    if args.mapping:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    # CSV에서 매핑 대상 클립만 수집하고 라벨별로 섞는다.
    # 두 형식 지원: 원본 "video_id;label" (헤더 없음), Kaggle 미러 "video_id,label,..." (헤더 있음)
    by_label: dict[str, list[str]] = {}
    with open(args.labels_csv, newline="", encoding="utf-8") as fp:
        first = fp.readline()  # 첫 줄만 미리 읽어서 구분자/헤더 유무를 추정한 뒤 필요하면 되감는다
        delimiter = "," if "," in first else ";"  # Kaggle 미러는 쉼표, 원본 배포본은 세미콜론 구분
        has_header = "video_id" in first  # 헤더가 있으면(Kaggle 미러) 그 줄은 데이터가 아니므로 건너뛰어야 함
        if not has_header:
            fp.seek(0)  # 헤더가 없는 원본 형식은 방금 읽은 첫 줄도 실제 데이터이므로 파일 처음으로 되감음
        for row in csv.reader(fp, delimiter=delimiter):
            if len(row) < 2:
                continue
            video_id, jester_label = row[0].strip(), row[1].strip()
            our_label = mapping.get(jester_label)  # 매핑에 없는 Jester 클래스(관련 없는 제스처)는 제외
            if our_label:
                by_label.setdefault(our_label, []).append(video_id)
    rng = random.Random(args.seed)  # 시드 고정 — 같은 --seed로 실행하면 항상 같은 부분집합이 뽑히도록 재현성 확보
    for ids in by_label.values():
        rng.shuffle(ids)  # CSV에 나열된 순서(촬영 순서 등에 편향 가능) 대신 무작위 순서로 --max-per-class를 자름

    print("변환 대상:", {k: min(len(v), args.max_per_class) for k, v in by_label.items()})
    extractor = FeatureExtractor()
    ts_ms = 0
    t_start = time.monotonic()

    for our_label, ids in by_label.items():
        out_dir = args.out / our_label
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = skipped = 0
        for video_id in ids:
            if saved >= args.max_per_class:
                break
            video_dir = args.jester_dir / video_id
            if not video_dir.is_dir():
                continue
            seq, ts_ms = extract_video(extractor, video_dir, ts_ms)
            if seq is None:
                skipped += 1
                continue
            # idle 라벨은 손이 없는 게 정상이므로 감지율 필터를 적용하지 않는다
            if our_label not in args.idle_labels and hand_detected_ratio(seq) < MIN_DETECTED_RATIO:
                skipped += 1
                continue
            np.save(out_dir / f"jester_{video_id}.npy", resample(seq, TARGET_FRAMES))
            saved += 1
        print(f"{our_label}: 저장 {saved}개, 스킵 {skipped}개 (손 감지 부족/프레임 없음)")

    extractor.close()
    print(f"완료 ({time.monotonic() - t_start:.0f}초) → {args.out}")


if __name__ == "__main__":
    main()
