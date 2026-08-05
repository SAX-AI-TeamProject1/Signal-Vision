# Last updated: 2026-08-06
'''
외부(ROS 노드 등)에서 이 파일 하나만 import해서 추론/GUI를 호출할 수 있게 묶은 진입점.

predict.py의 main()은 무한루프+argparse CLI라 다른 프로세스에 끼워 넣기 어렵다.
여기서는 같은 로직을 "프레임 한 장 처리"만 하는 두 함수(infer, show_gui)로 쪼갰다 —
모델/카메라/HUD/안정화 필터 상태는 모듈 안에 숨겨두고, 첫 infer() 호출 때 한 번만 초기화한다.

사용 예:
    from src.inference.function import infer, show_gui
    while True:
        signal, confidence = infer()      # 웹캠 한 프레임 처리 + 안정화된 신호
        if show_gui():                    # 관절/확률 패널 표시, True면 종료 요청(q/ESC)
            break
'''

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..capture.extractor import (
    HAND_DIM,
    FeatureExtractor,
    FrameLivenessMonitor,
    HandWarning,
    LensHealthMonitor,
    NO_POSE_FRAMES,
    draw_detections,
    quality_status_text,
    open_camera,
    process_frame,
)
from .predict import DEFAULT_MODEL_PATH, SignalStabilizer, load_model
from .ui import Hud, header_for


class _Runtime:
    '''infer()/show_gui()가 공유하는 상태. 모듈 전역에 하나만 만든다(_get_runtime).'''

    def __init__(self, model_path: Path, threshold: float, consecutive: int,
                ema: float, release_grace: float) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.labels, self.num_frames = load_model(model_path, self.device)
        self.extractor = FeatureExtractor()
        self.cap = open_camera()
        self.hud = Hud()
        self.warning = HandWarning()
        self.lens_monitor = LensHealthMonitor()
        self.liveness = FrameLivenessMonitor()
        self.frozen = False
        self.no_pose = 0
        self.stabilizer = SignalStabilizer(threshold, consecutive, ema, release_grace)
        self.window: deque[np.ndarray] = deque(maxlen=self.num_frames)
        self.t0 = time.monotonic()
        self._last_frame = None

    def infer(self) -> tuple[str, float]:
        frame, hand_result, pose_result = process_frame(self.cap, self.extractor, self.t0)
        if frame is None:
            return "unknown", 0.0

        # 선명도 측정은 반드시 그리기 **전에** — 스켈레톤 선·경고 오버레이는 인공적인
        # 고대비 엣지라 그 뒤에 재면 선명도가 실측보다 크게 부풀려지고(predict.py 실측
        # 평균 +157%), 사람이 잡히는 동안에는 렌즈가 흐려져도 판정이 안 뜬다.
        poor_quality = self.lens_monitor.update(frame)
        self.frozen = self.liveness.update(frame)
        draw_detections(frame, hand_result, pose_result)
        self.warning.update(hand_result)
        self.warning.draw(frame)
        self.window.append(FeatureExtractor.vector(hand_result, pose_result))
        self.no_pose = 0 if pose_result.pose_landmarks else self.no_pose + 1

        if len(self.window) == self.num_frames:
            arr = np.stack(self.window)
            # 안전 가드: 윈도우 대부분에서 사람(포즈)이 안 잡히면 모델 판단을 신뢰하지 않는다
            pose_ratio = float((arr[:, HAND_DIM * 2:] != 0).any(axis=1).mean())
            if pose_ratio < 0.3 or (poor_quality and self.no_pose >= NO_POSE_FRAMES):
                # 사람 미검출 + 화질 이상 = 비상. 화질이 멀쩡하면 그냥 대기다.
                self.stabilizer.mark_person_absent(self.frozen or poor_quality)
            else:
                x = torch.from_numpy(arr[None]).to(self.device)
                with torch.no_grad():
                    raw_probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
                # 관절이 잡힌다 = 대체로 인식이 되고 있으므로 화질 저하는 넘기지 않는다.
                # 화면 정지·치명적 선명도 저하만 예외 — 관절이 잡혀도 못 믿는 경우다.
                self.stabilizer.update(raw_probs, self.labels,
                                       self.frozen or self.lens_monitor.critical)

        self._last_frame = frame
        idx = self.stabilizer.confirmed_idx
        signal = self.labels[idx] if idx is not None else "unknown"
        confidence = float(self.stabilizer.probs[idx]) \
            if (self.stabilizer.probs is not None and idx is not None) else 0.0
        return signal, confidence

    def show_gui(self) -> bool:
        if self._last_frame is None:
            return False
        idx = self.stabilizer.confirmed_idx
        header, header_color = header_for(self.stabilizer.display_state, self.stabilizer.confirmed)
        return self.hud.render(
            self._last_frame, self.labels, self.stabilizer.probs, self.stabilizer.top, idx,
            self.stabilizer.threshold, header=header, header_color=header_color,
            status=f"{self.stabilizer.status}  |  "
                   f"{quality_status_text(self.lens_monitor, self.frozen)}",
            emergency=self.stabilizer.emergency,
        )

    def close(self) -> None:
        self.cap.release()
        self.hud.close()
        self.extractor.close()


_runtime: _Runtime | None = None


def _get_runtime() -> _Runtime:
    global _runtime
    if _runtime is None:
        # predict.py의 argparse 기본값과 동일 (--threshold 0.8 --consecutive 5 --ema 0.4 --release-grace 1.0)
        _runtime = _Runtime(DEFAULT_MODEL_PATH, threshold=0.8, consecutive=5, ema=0.4, release_grace=1.0)
    return _runtime


def infer() -> tuple[str, float]:
    '''웹캠 한 프레임을 읽어 특징추출 → LSTM 추론 → 안정화 필터까지 진행하고 (signal, confidence)를 반환한다.

    사람 미검출/저신뢰도/카메라 프레임 없음은 전부 signal="unknown" (안전 기본값 = 정지).
    '''
    return _get_runtime().infer()


def show_gui() -> bool:
    '''가장 최근 infer() 결과를 관절/확률 패널 GUI로 띄운다. True를 반환하면 종료 요청(q/ESC).'''
    return _get_runtime().show_gui()


def close() -> None:
    '''카메라/HUD/추출기 리소스를 정리한다. 종료 시 반드시 호출.'''
    global _runtime
    if _runtime is not None:
        _runtime.close()
        _runtime = None
