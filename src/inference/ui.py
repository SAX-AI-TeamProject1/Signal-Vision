# Last updated: 2026-07-27

'''
predict.py의 화면 표시(HUD) 전담 — 한글 오버레이 텍스트, 클래스별 확률 막대 패널, 창 표시/키 입력을 다룬다.
'''

import platform
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Sign Inference"

# 한글 오버레이용 폰트 (OS별) — cv2.putText는 한글을 못 그리므로 PIL 사용
FONT_CANDIDATES = {
    "Darwin": ["/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               "/System/Library/Fonts/AppleSDGothicNeo.ttc"],
    "Windows": ["C:/Windows/Fonts/malgun.ttf"],
    "Linux": ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
}

# 확률 패널 레이아웃
BAR_X, BAR_TOP, BAR_W, BAR_H, BAR_GAP = 210, 120, 320, 30, 12

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
                           confirmed_idx: int | None, threshold: float):
    '''클래스별 확률 막대 패널을 그린다. 텍스트 항목 목록을 반환한다 (draw_text용).'''
    panel_h = len(labels) * (BAR_H + BAR_GAP) + BAR_GAP
    x0, y0 = 10, BAR_TOP - BAR_GAP
    x1, y1 = BAR_X + BAR_W + 70, BAR_TOP + panel_h - BAR_GAP
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    texts = []
    for i, label in enumerate(labels):
        y = BAR_TOP + i * (BAR_H + BAR_GAP)
        p = float(probs[i]) if probs is not None else 0.0
        if confirmed_idx is not None and i == confirmed_idx:
            color = (0, 220, 0)      # 확정 = 초록
        elif top is not None and i == top:
            color = (0, 180, 255)    # 현재 1위 = 주황
        else:
            color = (180, 130, 60)   # 나머지 = 파랑
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (70, 70, 70), -1)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + int(BAR_W * p), y + BAR_H), color, -1)
        tx = BAR_X + int(BAR_W * threshold)
        cv2.line(frame, (tx, y - 3), (tx, y + BAR_H + 3), (255, 255, 255), 1)  # 임계값선
        texts.append((label, (20, y + 2), color, 24))
        texts.append((f"{p * 100:4.0f}%", (BAR_X + BAR_W + 10, y + 2), (230, 230, 230), 24))
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
        panel_texts = draw_probability_panel(frame, labels, probs, top, confirmed_idx, threshold)
        texts = [(header,(10,10),header_color,40),(status,(10,62),status_color,24),*panel_texts]
        if emergency:
            texts += draw_emergency_toast(frame)
        frame = self._draw_text(frame,texts)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)  # 27 = ESC

    def close(self) -> None:
        """HUD 창을 닫는다 (종료 시 반드시 호출)."""
        cv2.destroyAllWindows()
