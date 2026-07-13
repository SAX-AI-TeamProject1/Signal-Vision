# Last updated: 2026-07-13
"""패키지 설치(pip install -e .) 실행 후 성공/실패를 명확한 배너로 출력."""

import subprocess
import sys
from pathlib import Path

from task_output import banner, step

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    step("pip install -e . 실행 중 (처음이면 수 분 걸릴 수 있음)")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT)
    if result.returncode != 0:
        banner(False, "패키지 설치 실패")
        return 1

    banner(True, "패키지 설치 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
