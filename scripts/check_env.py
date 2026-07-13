# Last updated: 2026-07-13
"""프로젝트 셋업 검증: 핵심 패키지 import 및 버전 확인."""

import cv2
import mediapipe
import numpy

print(
    "OK | cv2",
    cv2.__version__,
    "| mediapipe",
    mediapipe.__version__,
    "| numpy",
    numpy.__version__,
)
