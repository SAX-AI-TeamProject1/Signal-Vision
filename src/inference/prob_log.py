# Last updated: 2026-08-03

'''
프레임별 라벨 확률 로깅 — 발표 자료용 시각화 데이터 생성.

실시간 추론(predict.py) 중 매 프레임 "각 수신호 라벨의 신뢰도"를 CSV 한 줄로 남긴다.
시각화는 이 저장소가 아니라 별도 디렉토리에서 하므로, 여기서는 그림을 그리지 않고
표 형태(CSV)로만 내보낸다 — pandas/matplotlib 어느 쪽에서 읽어도 되게.

CSV 열 구성:
    tag            어떤 수신호를 수행하며 기록했는지 (--log-tag, 없으면 빈 값)
    frame          기록 시작 이후 프레임 번호 (0부터)
    elapsed_s      기록 시작 이후 경과 시간(초) — x축으로 쓰면 된다 (FPS가 흔들려도 정확)
    wall_time      실제 시각(Unix epoch 초) — 다른 로그와 시간 정렬할 때 사용
    confirmed      그 프레임의 확정 신호 ("인식 불가" 포함)
    status         안정화 필터 상태 문자열 (연속 일치 n/N, 유예 대기 등)
    top            1위 라벨 (확률 없으면 빈 값)
    top_conf       1위 확률
    p_<라벨>...    EMA 평활된 확률 — HUD 막대·확정 판정이 쓰는 값
    raw_<라벨>...  모델이 그 프레임에 낸 원본 softmax 확률 (평활 전)

평활 전/후를 둘 다 남기는 이유: 발표에서 "EMA가 스파이크를 어떻게 눌러주는가"를
같은 그래프에 겹쳐 보여줄 수 있고, 둘 중 뭘 쓸지는 시각화 쪽에서 고르면 되기 때문.

3초 기록 워크플로:
    라벨 하나당 한 번씩 실행 — --log-tag stop --log-delay 3 --log-seconds 3
    → 3초 준비 후 3초 동안만 기록하고 자동 종료. 라벨을 바꿔가며 반복하면
      수신호별 CSV가 하나씩 쌓인다.
'''

import csv  # 시각화 쪽에서 pandas.read_csv 한 줄로 읽을 수 있게 표준 CSV로 저장
import time
from pathlib import Path

import numpy as np

DEFAULT_LOG_DIR = "logs"  # --log-probs를 값 없이 준 경우 CSV가 쌓이는 디렉토리 (ROOT 기준)


def default_log_path(root: Path, tag: str) -> Path:
    '''--log-probs를 경로 없이 켰을 때 쓸 파일 경로를 만든다 (덮어쓰기 방지용 타임스탬프 포함).'''
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"probs_{tag}_{stamp}.csv" if tag else f"probs_{stamp}.csv"
    return root / DEFAULT_LOG_DIR / name


class ProbabilityLogger:
    '''매 프레임 확률을 CSV로 append 하는 로거.

    상태는 세 단계다:
        대기(--log-delay 동안 준비 시간) → 기록 중 → 완료(--log-seconds 경과)
    시작 기준 시각(_start)은 "첫 확률이 나온 프레임"으로 잡는다. 윈도우(30프레임)가
    채워지기 전에는 확률 자체가 없어서, 카메라를 켠 시점부터 재면 3초 중 앞부분이
    빈 구간으로 날아가기 때문이다.
    '''

    def __init__(self, path: Path, labels: list[str], *, tag: str = "",
                 duration: float | None = None, delay: float = 0.0,
                 console: bool = False) -> None:
        self.path = path
        self.labels = labels
        self.tag = tag
        self.duration = duration  # None이면 종료(q/ESC)할 때까지 계속 기록
        self.delay = delay
        self.console = console  # True면 CSV와 별개로 콘솔에도 프레임마다 한 줄씩 출력

        self.finished = False  # duration을 채워 기록이 끝난 상태 (main 루프가 종료 판단에 사용)
        self._start: float | None = None  # 기록 시작 시각 (첫 확률 프레임에서 설정)
        self._first_seen: float | None = None  # 확률이 처음 나온 시각 (delay 카운트다운 기준)
        self._frame = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)  # logs/ 등 상위 폴더가 없으면 생성
        # newline="": csv 모듈 규약 — 안 주면 Windows에서 줄 사이에 빈 줄이 하나씩 낀다
        # utf-8-sig: 한글 라벨/확정값이 엑셀에서 깨지지 않도록 BOM 포함
        self._file = self.path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["tag", "frame", "elapsed_s", "wall_time", "confirmed", "status", "top", "top_conf"]
            + [f"p_{label}" for label in labels]
            + [f"raw_{label}" for label in labels]
        )
        span = f"{duration:.1f}초" if duration else "종료할 때까지"
        print(f"[log] 확률 로그 → {self.path}  (태그: {tag or '없음'}, {span} 기록)")

    def status_text(self) -> str:
        '''HUD 상태줄에 덧붙일 짧은 문구 — 지금 준비 중인지 기록 중인지 화면으로 확인용.'''
        if self.finished:
            return " | 기록 완료"
        if self._start is None:
            if self._first_seen is None:
                return " | 기록 대기"
            return f" | 기록 준비 {max(0.0, self.delay - (time.monotonic() - self._first_seen)):.1f}s"
        elapsed = time.monotonic() - self._start
        if self.duration is None:
            return f" | 기록 중 {elapsed:.1f}s"
        return f" | 기록 중 {elapsed:.1f}/{self.duration:.1f}s"

    def log(self, probs: np.ndarray | None, raw_probs: np.ndarray | None,
            confirmed: str, status: str) -> None:
        '''한 프레임 분을 기록한다. probs가 None(버퍼 채우는 중·사람 미감지)이면 확률 칸은 빈 값.

        predict.py 메인 루프가 프레임마다 무조건 한 번 호출하면 되고, 대기/기록/완료
        판단은 전부 이 안에서 처리한다.
        '''
        if self.finished:
            return

        now = time.monotonic()
        if self._start is None:
            # 아직 기록 시작 전 — 확률이 나오기 시작한 뒤 delay초가 지나야 시작한다
            if probs is None:
                return
            if self._first_seen is None:
                self._first_seen = now
                if self.delay > 0:
                    print(f"[log] {self.delay:.1f}초 뒤 기록 시작 — 수신호 준비")
            if now - self._first_seen < self.delay:
                return
            self._start = now
            print(f"[log] 기록 시작 (태그: {self.tag or '없음'})")

        elapsed = now - self._start
        top = int(probs.argmax()) if probs is not None else None
        blank = [""] * len(self.labels)  # 확률이 없는 프레임(사람 미감지 등)은 빈 칸으로 남긴다
        smoothed = [f"{float(p):.6f}" for p in probs] if probs is not None else blank
        raw = [f"{float(p):.6f}" for p in raw_probs] if raw_probs is not None else blank
        self._writer.writerow(
            [self.tag, self._frame, f"{elapsed:.4f}", f"{time.time():.4f}",
             confirmed, status,
             self.labels[top] if top is not None else "",
             f"{float(probs[top]):.6f}" if top is not None else ""]
            + smoothed + raw
        )
        if self.console:
            shown = " ".join(f"{label} {float(probs[i]):.2f}" for i, label in enumerate(self.labels)) \
                if probs is not None else "확률 없음"
            print(f"[log] {elapsed:6.2f}s  {shown}  → {confirmed}")
        self._frame += 1

        if self.duration is not None and elapsed >= self.duration:
            self.finished = True
            print(f"[log] 기록 완료 — {self._frame}프레임 / {elapsed:.2f}초 → {self.path}")

    def close(self) -> None:
        self._file.close()
        if not self.finished and self._frame:
            print(f"[log] 기록 종료 — {self._frame}프레임 → {self.path}")
