# Last updated: 2026-07-13
"""venv 생성: Python 3.12 확인/설치 후 .venv 생성. 성공/실패를 명확한 배너로 출력."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from task_output import banner, step

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"


def find_python312() -> list[str] | None:
    """3.12 인터프리터를 실행할 커맨드(argv)를 찾는다. 없으면 None."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["py", "-3.12", "-c", "print(1)"],
                capture_output=True,
                timeout=10,
            )
            return ["py", "-3.12"] if result.returncode == 0 else None
        except FileNotFoundError:
            return None
    exe = shutil.which("python3.12")
    return [exe] if exe else None


def install_python312() -> bool:
    if sys.platform == "win32":
        if shutil.which("winget") is None:
            print("winget이 없어 자동 설치 불가")
            return False
        step("Python 3.12 미설치 - winget으로 설치 시도")
        result = subprocess.run(
            [
                "winget", "install", "-e", "--id", "Python.Python.3.12",
                "--accept-package-agreements", "--accept-source-agreements",
            ]
        )
        return result.returncode == 0
    if sys.platform == "darwin":
        if shutil.which("brew") is None:
            print("Homebrew가 없어 자동 설치 불가")
            return False
        step("Python 3.12 미설치 - Homebrew로 설치 시도")
        return subprocess.run(["brew", "install", "python@3.12"]).returncode == 0
    print("이 플랫폼은 자동 설치를 지원하지 않습니다 (배포판 패키지 관리자로 python3.12 설치 필요)")
    return False


def main() -> int:
    if VENV_DIR.exists():
        banner(True, ".venv 이미 존재 - 건너뜀")
        return 0

    step("Python 3.12 탐색 중")
    python_cmd = find_python312()
    if python_cmd is None:
        if not install_python312() or (python_cmd := find_python312()) is None:
            banner(False, "Python 3.12 설치 실패 - https://www.python.org/downloads/ 에서 수동 설치 필요")
            return 1

    step(f".venv 생성 중 ({' '.join(python_cmd)})")
    result = subprocess.run([*python_cmd, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0:
        banner(False, "venv 생성 실패")
        return 1

    banner(True, f".venv 생성 완료 ({' '.join(python_cmd)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
