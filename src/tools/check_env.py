# Last updated: 2026-07-13
"""프로젝트 셋업 검증: 핵심 패키지 import 및 버전 확인. 성공/실패를 명확한 배너로 출력."""

import sys

from task_output import banner, step


def main() -> int:
    step("cv2 / mediapipe / numpy import 확인 중 (처음 실행 시 몇 초 걸릴 수 있음)")
    try:
        import cv2
        import mediapipe
        import numpy
    except ImportError as exc:
        banner(
            False,
            f"패키지 import 실패: {exc}\n"
            f"  인터프리터: {sys.executable}\n"
            "  '2. 패키지 설치' 태스크를 다시 실행해 보세요 (.venv를 다른 프로세스가 동시에 "
            "건드리는 중이었다면 재시도로 해결되는 경우가 많습니다)",
        )
        return 1

    banner(
        True,
        f"cv2 {cv2.__version__} | mediapipe {mediapipe.__version__} | numpy {numpy.__version__}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
