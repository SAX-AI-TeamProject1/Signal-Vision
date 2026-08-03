# Last updated: 2026-08-03

'''
신뢰도 로그(CSV) → 수신호별 평균 신뢰도 막대그래프 (발표 자료용).

`predict.py --log-probs`가 남긴 로그를 읽어, **한 수신호를 3초간 수행하는 동안
7개 라벨 각각의 평균 신뢰도**를 가로 막대로 그린다. 한 장이 답하는 질문은 두 가지다:

    1. 수행한 수신호를 제대로 인식했는가?   → 강조(파란) 막대가 얼마나 긴가
    2. 헷갈렸다면 무엇과 헷갈렸는가?        → 회색 막대 중 어느 것이 따라 올라왔는가

그래서 색은 "라벨마다 다른 색"이 아니라 **강조(emphasis)** 방식이다 — 수행한 라벨
하나만 파랑, 나머지는 회색. 라벨마다 색을 주면 정작 하고 싶은 이야기(정답 vs 혼동)가
7색 속에 묻힌다. 막대 길이가 이미 값을 말하므로 색은 "어느 것이 정답이었나"만 나른다.

실행:
    python -m src.tools.plot_probs                     # logs/의 CSV 전부 → logs/charts/
    python -m src.tools.plot_probs --raw               # EMA 평활 전 원본 확률로
    python -m src.tools.plot_probs --tags come stop    # 일부 수신호만
    python -m src.tools.plot_probs --show              # 저장 + 화면에 띄우기

출력 (기본 logs/charts/):
    sign_<태그>.png   수신호 하나당 한 장
    sign_all.png      전체를 한 장에 모은 비교용 (슬라이드 한 장짜리)
    summary.csv       그래프와 같은 수치의 표 (태그 × 라벨) — 색을 못 읽어도 값은 남는다

같은 태그의 CSV가 여러 개면 전부 합쳐(프레임 단위로) 평균을 낸다 — 같은 수신호를
여러 번 찍었다면 자동으로 누적된다.
'''

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_OUT_DIR = ROOT / "logs" / "charts"

# 색 — dataviz 팔레트의 "강조(1 hue + gray)" 조합.
# 파랑 #2a78d6은 흰 배경 대비 4.30:1, 회색 #c3c2b7과의 OKLab ΔE는 29.5로
# 일반 시야 분리 기준(15)을 크게 넘는다. 색이 나르는 정보는 "정답 라벨" 하나뿐이라
# 색각 이상에서도 밝기 차이만으로 구분되고, 굵은 축 라벨이 같은 정보를 한 번 더 준다.
SURFACE = "#fcfcfb"   # 그림 배경
ACCENT = "#2a78d6"    # 수행한 수신호 막대
MUTED = "#c3c2b7"     # 나머지 라벨 막대 (맥락)
GRID = "#e1e0d9"      # 격자 — 배경에서 한 단계만 떨어진 실선 헤어라인
INK = "#0b0b0b"       # 제목·강조 텍스트
TEXT2 = "#52514e"     # 부제·값 라벨
TEXT3 = "#898781"     # 축 눈금·임계값 주석

BAR_HEIGHT = 0.5      # 슬롯(1.0)의 절반만 채워 막대 사이에 공기를 남긴다
LABEL_MIN = 0.05      # 이 값 이상인 막대에만 숫자를 붙인다 (전부 붙이면 읽히지 않는다)


def apply_korean_font() -> None:
    '''matplotlib에 한글 폰트를 등록한다 (없으면 제목·라벨이 네모로 깨진다).

    OS별 폰트 경로 목록은 HUD(`src/inference/ui.py`)가 이미 갖고 있으므로 그대로
    재사용한다 — 같은 경로를 두 곳에 하드코딩하면 한쪽만 고쳐지기 때문.
    '''
    from matplotlib import font_manager

    from src.inference.ui import find_font

    path = find_font()
    if path is None:
        print("[plot] 한글 폰트를 찾지 못했습니다 — 제목이 깨져 보일 수 있습니다")
        return
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호가 한글 폰트에서 깨지는 것 방지


def read_log(path: Path, raw: bool) -> tuple[str, list[str], np.ndarray, float]:
    '''CSV 한 개를 읽어 (태그, 라벨 목록, 프레임별 확률 배열, 기록 길이)를 돌려준다.

    라벨 목록은 헤더의 `p_<라벨>` 열에서 뽑는다 — 모델이 학습된 라벨 순서가 그대로
    열 순서라, 클래스 목록을 여기에 다시 하드코딩하지 않아도 된다(design.md 규약).
    확률 칸이 빈 행(사람 미감지 등)은 평균에서 제외한다.
    '''
    prefix = "raw_" if raw else "p_"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        labels = [name[len("p_"):] for name in (reader.fieldnames or []) if name.startswith("p_")]
        if not labels:
            raise ValueError(f"{path.name}: p_<라벨> 열이 없습니다 — 확률 로그 CSV가 맞는지 확인하세요")
        tag, rows, last_elapsed = "", [], 0.0
        for row in reader:
            tag = tag or (row.get("tag") or "")
            last_elapsed = float(row.get("elapsed_s") or 0.0)
            values = [row.get(prefix + label, "") for label in labels]
            if all(v != "" for v in values):
                rows.append([float(v) for v in values])
    return tag or path.stem, labels, np.array(rows, dtype=float), last_elapsed


def collect(paths: list[Path], raw: bool) -> tuple[list[str], OrderedDict]:
    '''여러 CSV를 태그별로 묶어 평균 신뢰도를 계산한다.

    같은 태그의 파일이 여러 개면 프레임을 전부 이어 붙인 뒤 평균 — 촬영 횟수가
    달라도 프레임 수에 비례해 가중되므로 별도 가중치가 필요 없다.
    '''
    labels_ref: list[str] | None = None
    buckets: OrderedDict[str, dict] = OrderedDict()
    for path in sorted(paths):
        tag, labels, probs, duration = read_log(path, raw)
        if labels_ref is None:
            labels_ref = labels
        elif labels != labels_ref:
            raise ValueError(f"{path.name}: 라벨 구성이 다릅니다 {labels} ≠ {labels_ref} "
                             "— 같은 모델로 기록한 로그끼리만 함께 그릴 수 있습니다")
        if not len(probs):
            print(f"[plot] {path.name}: 확률이 있는 프레임이 없어 건너뜁니다")
            continue
        bucket = buckets.setdefault(tag, {"probs": [], "files": 0, "duration": 0.0})
        bucket["probs"].append(probs)
        bucket["files"] += 1
        bucket["duration"] += duration
    for bucket in buckets.values():
        stacked = np.concatenate(bucket["probs"])
        bucket["mean"] = stacked.mean(axis=0)
        bucket["frames"] = len(stacked)
    return labels_ref or [], buckets


def draw_panel(ax, labels: list[str], mean: np.ndarray, tag: str, threshold: float,
               *, compact: bool = False) -> None:
    '''패널 하나 — 라벨별 평균 신뢰도 가로 막대.

    라벨 순서는 모든 패널에서 동일하게 고정한다(값 크기순으로 정렬하지 않는다).
    여러 수신호의 그래프를 나란히 놓고 볼 때 같은 위치가 같은 라벨이어야 비교가 된다.
    '''
    y = np.arange(len(labels))[::-1]  # 첫 라벨이 맨 위에 오도록 뒤집는다
    colors = [ACCENT if label == tag else MUTED for label in labels]
    ax.barh(y, mean, height=BAR_HEIGHT, color=colors, linewidth=0, zorder=2)  # 막대가 격자·임계값선 위

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="both", length=0, colors=TEXT3,
                   labelsize=9 if compact else 11)
    for tick, label in zip(ax.get_yticklabels(), labels):
        if label == tag:  # 수행한 라벨은 색 말고 굵기로도 표시 — 색만으로 정보를 나르지 않게
            tick.set_color(INK)
            tick.set_fontweight("bold")
        else:
            tick.set_color(TEXT2)

    # 격자·축은 배경에서 한 단계만 떨어진 실선 헤어라인 (데이터보다 뒤로)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["left"].set_linewidth(0.8)

    # 확정 임계값 — 이 선을 넘어야 실제로 신호가 확정된다
    ax.axvline(threshold, color=TEXT3, linewidth=1.0, zorder=1)
    if not compact:
        ax.text(threshold, len(labels) - 0.35, f" 확정 임계값 {threshold:g}",
                color=TEXT3, fontsize=9, va="bottom", ha="left")

    # 값 라벨은 선별적으로 — 이야기가 되는 막대(정답 + 의미 있게 올라온 혼동)에만
    for yi, (label, value) in zip(y, zip(labels, mean)):
        if label != tag and value < LABEL_MIN:
            continue
        ax.text(min(value + 0.015, 0.995), yi, f"{value * 100:.0f}%",
                va="center", ha="left", fontsize=9 if compact else 10,
                color=INK if label == tag else TEXT2,
                fontweight="bold" if label == tag else "normal")


def panel_subtitle(labels: list[str], mean: np.ndarray, tag: str, bucket: dict,
                   compact: bool = False) -> str:
    '''"제대로 맞혔나 / 뭐랑 헷갈렸나"를 한 줄 문장으로 — 그래프를 읽기 전에 결론부터.

    compact(모아보기 패널)에서는 프레임 수·길이를 뺀다 — 좁은 패널에서 제목이 옆 칸을
    침범하기 때문. 뺀 수치는 개별 그래프와 summary.csv에 그대로 남는다.
    '''
    order = np.argsort(mean)[::-1]
    top_label, top_value = labels[order[0]], mean[order[0]]
    rival_idx = next((i for i in order if labels[i] != tag), None)
    if tag not in labels:  # 태그가 모델 라벨이 아닌 경우(파일명에서 온 태그 등)
        verdict = f"1위 {top_label} {top_value * 100:.0f}%"
    elif top_label == tag:
        verdict = f"정답 {tag} {mean[labels.index(tag)] * 100:.0f}%"
        if rival_idx is not None and mean[rival_idx] >= LABEL_MIN:
            verdict += f" · 최다 혼동 {labels[rival_idx]} {mean[rival_idx] * 100:.0f}%"
    else:
        # 수행한 라벨이 1위가 아닌 경우 — 무엇에 밀렸는지가 이 그래프의 핵심이 된다
        verdict = (f"오인식 — 1위 {top_label} {top_value * 100:.0f}% > "
                   f"수행 {tag} {mean[labels.index(tag)] * 100:.0f}%")
    if compact:
        return verdict
    took = f"{bucket['frames']}프레임 / {bucket['duration']:.1f}초"
    if bucket["files"] > 1:
        took += f" (파일 {bucket['files']}개 합산)"
    return f"{took} · {verdict}"


def save_single(out_dir: Path, labels: list[str], tag: str, bucket: dict,
                threshold: float, dpi: int, source: str) -> Path:
    '''수신호 하나짜리 그래프를 PNG로 저장한다.'''
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    draw_panel(ax, labels, bucket["mean"], tag, threshold)
    fig.suptitle(f"'{tag}' 수신호 — 라벨별 평균 신뢰도", x=0.02, ha="left",
                 fontsize=15, color=INK, fontweight="bold")
    # pad를 넉넉히 — 부제와 그래프 안 "확정 임계값" 주석이 세로로 붙지 않게
    ax.set_title(f"{panel_subtitle(labels, bucket['mean'], tag, bucket)}  ({source})",
                 loc="left", fontsize=10, color=TEXT2, pad=22)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / f"sign_{tag}.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def save_grid(out_dir: Path, labels: list[str], buckets: OrderedDict,
              threshold: float, dpi: int, cols: int, source: str) -> Path:
    '''전체 수신호를 한 장에 모은 비교용 그래프 (슬라이드 한 장짜리).'''
    n = len(buckets)
    cols = min(cols, n)
    rows = -(-n // cols)  # 올림 나눗셈
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.4 * rows), dpi=dpi,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (tag, bucket) in zip(axes.flat, buckets.items()):
        ax.set_facecolor(SURFACE)
        draw_panel(ax, labels, bucket["mean"], tag, threshold, compact=True)
        # 제목은 두 줄 — 좁은 패널에서 한 줄로 쓰면 옆 칸 제목과 붙어 읽히지 않는다
        ax.set_title(f"'{tag}' 수행\n{panel_subtitle(labels, bucket['mean'], tag, bucket, compact=True)}",
                     loc="left", fontsize=9.5, color=TEXT2, pad=8)
    for ax in axes.flat[n:]:
        ax.set_visible(False)  # 남는 칸은 빈 축을 그리지 않고 숨긴다
    fig.suptitle(f"수신호별 라벨 평균 신뢰도  ({source}, 세로선 = 확정 임계값 {threshold:g})",
                 x=0.01, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "sign_all.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def save_table(out_dir: Path, labels: list[str], buckets: OrderedDict) -> Path:
    '''그래프와 같은 수치를 표로도 남긴다 — 색을 못 읽어도, 인쇄해도 값은 남는다.'''
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "files", "frames", "duration_s"] + labels)
        for tag, bucket in buckets.items():
            writer.writerow([tag, bucket["files"], bucket["frames"], f"{bucket['duration']:.2f}"]
                            + [f"{v:.4f}" for v in bucket["mean"]])
    return path


def print_table(labels: list[str], buckets: OrderedDict) -> None:
    '''터미널에서도 바로 확인할 수 있게 같은 표를 출력한다.'''
    width = max(len(label) for label in labels + ["수행\\예측"]) + 2
    print("\n" + "수행\\예측".ljust(width) + "".join(label.rjust(width) for label in labels))
    for tag, bucket in buckets.items():
        cells = "".join(f"{v * 100:.0f}%".rjust(width) for v in bucket["mean"])
        print(tag.ljust(width) + cells)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="신뢰도 로그 → 수신호별 평균 신뢰도 막대그래프")
    parser.add_argument("--input", type=Path, default=DEFAULT_LOG_DIR,
                        help="CSV가 있는 디렉토리 또는 CSV 파일 경로 (기본: logs/)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="PNG·summary.csv를 저장할 디렉토리 (기본: logs/charts/)")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="그릴 수신호만 골라서 (기본: 찾은 것 전부)")
    parser.add_argument("--raw", action="store_true",
                        help="EMA 평활 전 원본 확률(raw_*)로 그린다 (기본: 평활 후 p_* — 실제 판정에 쓰이는 값)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="그래프에 표시할 확정 임계값 (predict.py --threshold와 맞출 것)")
    parser.add_argument("--cols", type=int, default=4, help="전체 비교 그래프의 열 수")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true", help="저장 후 화면에도 띄운다")
    args = parser.parse_args()

    if not args.input.exists():  # 아직 로그를 한 번도 남기지 않은 경우 (logs/ 자체가 없음)
        print(f"[plot] {args.input} 가 없습니다 — 먼저 `predict.py --log-probs`로 로그를 남기세요")
        return
    paths = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    paths = [p for p in paths if p.name != "summary.csv"]  # 이 도구가 만든 표를 다시 읽지 않도록
    if not paths:
        print(f"[plot] {args.input} 에서 CSV를 찾지 못했습니다 — "
              "먼저 `predict.py --log-probs`로 로그를 남기세요")
        return

    labels, buckets = collect(paths, args.raw)
    if args.tags:
        buckets = OrderedDict((t, b) for t, b in buckets.items() if t in args.tags)
    if not buckets:
        print("[plot] 그릴 데이터가 없습니다")
        return

    if not args.show:
        plt.switch_backend("Agg")  # 창을 띄우지 않는 저장 전용 백엔드 (원격/터미널 환경 대비)
    apply_korean_font()
    args.out.mkdir(parents=True, exist_ok=True)
    source = "EMA 평활 전 원본 확률" if args.raw else "EMA 평활 확률"

    for tag, bucket in buckets.items():
        print(f"[plot] 저장 → {save_single(args.out, labels, tag, bucket, args.threshold, args.dpi, source)}")
    if len(buckets) > 1:
        print(f"[plot] 저장 → {save_grid(args.out, labels, buckets, args.threshold, args.dpi, args.cols, source)}")
    print(f"[plot] 저장 → {save_table(args.out, labels, buckets)}")
    print_table(labels, buckets)

    if args.show:
        for tag, bucket in buckets.items():
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            fig.patch.set_facecolor(SURFACE)
            ax.set_facecolor(SURFACE)
            draw_panel(ax, labels, bucket["mean"], tag, args.threshold)
            ax.set_title(f"'{tag}' 수신호 — 라벨별 평균 신뢰도", loc="left",
                         fontsize=13, color=INK, fontweight="bold", pad=12)
            fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
