# Last updated: 2026-08-06

'''
predict.py의 화면 표시(HUD) 전담 — 한글 오버레이 텍스트, 클래스별 확률 막대 패널, 창 표시/키 입력을 다룬다.
'''

import platform
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Sign Inference"

# 화면이 표시할 상태는 이 셋뿐이고, 서로 겹치지 않는다 (SignalStabilizer.display_state).
#
#   emergency — 렌즈 이상(lens_fault). 이미지 자체를 믿을 수 없으니 즉시정지 + 정비 알림.
#   confirmed — 라벨이 확정됐고 그게 명령이 있는 라벨이다. 그대로 주행.
#   waiting   — 그 밖의 전부: 버퍼 채우는 중 / 사람 미감지 / 미확정 / 확정된 게 idle.
#               로봇은 서 있지만 이건 고장이 아니라 "아직 신호가 없다"는 뜻이다.
#
# 왜 waiting 을 따로 두는가: 예전에는 이 상태가 빨간 "인식 불가"로 떠서, 신호를 안 주고
# 서 있는 정상 상태와 렌즈가 막힌 비상 상태가 화면에서 똑같아 보였다. 그래서 EMERGENCY 가
# "인식이 되냐 안 되냐"로 뜨는 것처럼 읽혔다. 색과 문구를 갈라 그 혼동을 없앤다.
NO_SIGNAL_TEXT = "인식 대기중"

_HEADER_BY_STATE = {
    "emergency": ("비상정지: 렌즈 이상", (0, 0, 255)),      # 빨강 — 유일한 고장 상태
    "waiting": (NO_SIGNAL_TEXT, (0, 200, 255)),            # 주황 — 정상이지만 명령 없음
}


def header_for(state: str, confirmed: str) -> tuple[str, tuple[int, int, int]]:
    '''표시 상태 → (헤더 문구, BGR 색). 호출부 세 곳이 같은 규칙을 쓰게 한 곳에 모은다.

    predict.py(CLI)·function.py(임베드)·camera_node/inference.py(ROS) 가 각자
    3분기를 복사해 두면 한쪽만 고쳐져 화면 뜻이 갈라진다.
    '''
    if state == "confirmed":
        return f"확정: {confirmed}", (0, 220, 0)
    return _HEADER_BY_STATE[state]

# 한글 오버레이용 폰트 (OS별) — cv2.putText는 한글을 못 그리므로 PIL 사용
FONT_CANDIDATES = {
    "Darwin": ["/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               "/System/Library/Fonts/AppleSDGothicNeo.ttc"],
    "Windows": ["C:/Windows/Fonts/malgun.ttf"],
    "Linux": ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
}

def find_font() -> str | None:
    for path in FONT_CANDIDATES.get(platform.system(), []):
        if Path(path).exists():
            return path
    return None


def make_text_drawer():
    '''한글 텍스트 렌더러를 반환한다. 여러 텍스트를 한 번의 변환으로 그린다.

    draw(frame, items) — items: [(text, (x, y), BGR색, 크기), ...]
    폰트가 없으면 cv2 기본(영문 대체)으로 폴백.
    '''
    font_path = find_font()
    if font_path is None:
        def draw(frame, items):
            for text, xy, color, size in items:
                cv2.putText(frame, text.encode("ascii", "replace").decode(), xy,
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 2)
            return frame
        return draw

    from PIL import Image, ImageDraw, ImageFont
    fonts: dict[int, "ImageFont.FreeTypeFont"] = {}

    def draw(frame, items):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for text, xy, color, size in items:
            font = fonts.setdefault(size, ImageFont.truetype(font_path, size))
            d.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return draw


def draw_probability_panel(frame, labels, probs, top: int | None,
                           confirmed_idx: int | None, threshold: float, bar_top: int):
    '''클래스별 확률 막대 패널을 그린다. 텍스트 항목 목록을 반환한다 (draw_text용).

    화면 전체를 가리지 않도록 좌상단에 작게 그린다 — 패널 전체가 프레임 면적의 1/10 안팎
    (가로 폭 기준 약 1/3)만 차지하도록 비율을 맞췄다. 퍼센트 숫자는 막대 폭을 따로 안 늘리려고
    막대 오른쪽 끝 안쪽에 겹쳐 그린다. bar_top(패널 시작 y)은 헤더/상태 텍스트 높이를 감안해
    Hud.render()가 미리 계산해서 넘긴다 — 서로 겹치지 않게 순서대로 배치하기 위함.
    '''
    h, w = frame.shape[:2]
    bar_x = int(w * 0.16)
    bar_w = int(w * 0.20)
    bar_h = max(14, int(h * 0.025))
    bar_gap = max(4, int(h * 0.008))
    font_size = max(12, int(h * 0.028))

    panel_h = len(labels) * (bar_h + bar_gap) + bar_gap
    x0, y0 = 5, bar_top - bar_gap
    x1, y1 = bar_x + bar_w + 8, bar_top + panel_h - bar_gap
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    texts = []
    for i, label in enumerate(labels):
        y = bar_top + i * (bar_h + bar_gap)
        p = float(probs[i]) if probs is not None else 0.0
        if confirmed_idx is not None and i == confirmed_idx:
            color = (0, 220, 0)      # 확정 = 초록
        elif top is not None and i == top:
            color = (0, 180, 255)    # 현재 1위 = 주황
        else:
            color = (180, 130, 60)   # 나머지 = 파랑
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_w * p), y + bar_h), color, -1)
        tx = bar_x + int(bar_w * threshold)
        cv2.line(frame, (tx, y - 2), (tx, y + bar_h + 2), (255, 255, 255), 1)  # 임계값선
        texts.append((label, (6, y), color, font_size))
        texts.append((f"{p * 100:3.0f}%", (bar_x + bar_w - 34, y), (240, 240, 240), font_size))
    return texts

def draw_emergency_toast(frame):
    # 화면 상단에 빨간 EMERGENCY 배너를 그린다. 텍스트 항목을 반환.
    h,w = frame.shape[:2]
    banner_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,banner_h), (0,0,220),-1)
    cv2.addWeighted(overlay,0.75,frame,0.25,0.,frame)
    return [("EMERGENCY",(w//2-100,12),(255,255,255),36)]

class Hud:
    '''predict.py 메인 루프가 매 프레임 호출하는 렌더러.

    폰트 로딩처럼 한 번만 준비하면 되는 상태를 인스턴스에 들고 있다가 render()에서 재사용한다.
    '''

    def __init__(self) -> None:
        self._draw_text = make_text_drawer()

    def render(self, frame, labels, probs, top: int | None, confirmed_idx: int | None,
              threshold: float, header: str, header_color, status: str,
              status_color=(200, 200, 200), emergency : bool = False) -> bool:
        """프레임에 확률 패널 + 헤더/상태 텍스트를 그려 창에 표시한다. 반환: 종료 요청 여부(q/ESC)."""
        h = frame.shape[0]
        # 헤더 → 상태 → 패널 순서로, 각 줄의 실제 높이(폰트 크기 * 1.3, 한글 하강부 포함 여유)를
        # 감안해 다음 줄 y를 계산한다 — 해상도가 달라져도 서로 겹치지 않는다.
        margin = max(6, int(h * 0.015))
        header_size = max(22, int(h * 0.075))
        status_size = max(14, int(h * 0.035))
        header_y = margin
        status_y = header_y + int(header_size * 1.3) + margin
        panel_top = status_y + int(status_size * 1.3) + margin

        panel_texts = draw_probability_panel(frame, labels, probs, top, confirmed_idx, threshold, panel_top)
        texts = [(header, (10, header_y), header_color, header_size),
                (status, (10, status_y), status_color, status_size),
                *panel_texts]
        if emergency:
            texts += draw_emergency_toast(frame)
        frame = self._draw_text(frame,texts)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)  # 27 = ESC

    def close(self) -> None:
        """HUD 창을 닫는다 (종료 시 반드시 호출)."""
        cv2.destroyAllWindows()
