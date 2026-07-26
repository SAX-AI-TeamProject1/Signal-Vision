# Last updated: 2026-07-15
"""패키지 설치(pip install -e .) 실행 후 성공/실패를 명확한 배너로 출력."""

import subprocess
import sys
from pathlib import Path

from task_output import banner, step

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    step("pip 사용 가능 여부 확인 중")
    pip_check = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
    if pip_check.returncode != 0:
        banner(
            False,
            f"이 인터프리터에 pip이 없습니다 (.venv가 손상됐을 수 있음): {sys.executable}\n"
            "  '1. venv 생성' 태스크를 다시 실행해 .venv를 재생성해 보세요.",
        )
        return 1

    step("pip install -e . 실행 중 (처음이면 수 분 걸릴 수 있음)")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT)
    if result.returncode != 0:
        banner(False, "패키지 설치 실패")
        return 1

    # pip이 성공을 보고해도 실제 import가 되는지는 별개 문제이므로 설치 직후 바로 검증한다
    # (동시에 다른 프로세스가 같은 .venv를 건드리는 경우 등, pip 종료 코드만으로는 못 잡는 문제 대비).
    step("설치된 패키지 import 확인 중")
    check = subprocess.run(
        [sys.executable, "-c", "import cv2, mediapipe, numpy"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        banner(False, f"설치는 됐지만 import 실패 - .venv가 깨졌을 수 있음:\n{check.stderr}")
        return 1

    banner(True, "패키지 설치 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
