# Last updated: 2026-07-27

'''
    collect.py의 화면 표시(UI) 전담 — 버튼 클릭 처리, 랜드마크/상태 텍스트/진행바 오버레이를 다룬다.
'''

import cv2

from src.capture.extractor import HandWarning, draw_detections

WINDOW = "Collect"
BTN_W, BTN_H, BTN_MARGIN = 150, 50, 10  # 버튼 하나의 가로/세로 크기와 버튼 사이 여백(픽셀)


class ButtonBar:
    ''' 창 우상단에 클릭 가능한 버튼을 그리고 클릭 이벤트를 받는다.'''

    def __init__(self) -> None:
        self._clicked: str | None = None  # 마지막으로 클릭된 버튼 이름 (pop()으로 소비됨)
        self._rects: dict[str, tuple[int, int, int, int]] = {}  # 버튼 이름 → (x1, y1, x2, y2) 픽셀 사각형

    def attach(self, window: str) -> None:
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, *_args) -> None:
        # cv2 콜백 시그니처가 고정이라 flags/param은 안 쓰지만 자리만 *_args로 받아둔다.
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (x1, y1, x2, y2) in self._rects.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._clicked = name

    def draw(self, frame, buttons: list[tuple[str, str, tuple[int, int, int]]]) -> None:
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
        clicked, self._clicked = self._clicked, None
        return clicked


def show(frame, hand_result, pose_result, text: str, color, bar: ButtonBar, buttons,
         progress: float | None = None,
         warning: HandWarning | None = None) -> str | None:
    '''
    랜드마크·상태 텍스트·버튼을 그려 표시하고, 발생한 동작을 반환한다.

    progress가 주어지면(0~1) 화면 하단에 빨간 진행 바를 그린다 (녹화 진행 표시용).
    warning이 주어지면 손 미감지 상태를 갱신하고 이탈 경고를 그린다.
    '''
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
