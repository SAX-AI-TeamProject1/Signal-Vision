# Last updated: 2026-07-15
"""venv 생성: Python 3.12 확인/설치 후 .venv 생성. 성공/실패를 명확한 배너로 출력."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from task_output import banner, step

ROOT = Path(__file__).resolve().parent.parent.parent
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


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_has_pip(venv_dir: Path) -> bool:
    exe = venv_python(venv_dir)
    if not exe.exists():
        return False
    return subprocess.run([str(exe), "-m", "pip", "--version"], capture_output=True).returncode == 0


def try_fix_missing_ensurepip_linux(python_cmd: list[str]) -> bool:
    """Debian/Ubuntu는 ensurepip이 core python3.X 패키지가 아니라 별도 python3.X-venv 패키지에
    들어있어, 그 패키지가 없으면 venv는 만들어져도 pip이 빠진 채로 생성된다. apt가 있으면 자동
    설치를 시도하고, apt가 없거나(다른 배포판) 실패하면 False를 반환해 수동 안내로 넘어간다."""
    if shutil.which("apt-get") is None:
        return False
    version_probe = subprocess.run(
        [*python_cmd, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
    )
    if version_probe.returncode != 0:
        return False
    pkg = f"python{version_probe.stdout.strip()}-venv"
    install_cmd = ["apt-get", "install", "-y", pkg]
    if os.geteuid() != 0:
        if shutil.which("sudo") is None:
            return False
        install_cmd = ["sudo", *install_cmd]
    step(f"ensurepip 모듈 없음 - {pkg} 설치 시도 (sudo 비밀번호 입력이 필요할 수 있습니다)")
    return subprocess.run(install_cmd).returncode == 0


def main() -> int:
    if VENV_DIR.exists():
        if venv_has_pip(VENV_DIR):
            banner(True, ".venv 이미 존재 - 건너뜀")
            return 0
        step(".venv는 있지만 pip이 없음 (ensurepip 누락 등) - 삭제 후 재생성")
        shutil.rmtree(VENV_DIR)

    step("Python 3.12 탐색 중")
    python_cmd = find_python312()
    if python_cmd is None:
        if not install_python312() or (python_cmd := find_python312()) is None:
            banner(False, "Python 3.12 설치 실패 - https://www.python.org/downloads/ 에서 수동 설치 필요")
            return 1

    step(f".venv 생성 중 ({' '.join(python_cmd)})")
    result = subprocess.run([*python_cmd, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0 or not venv_has_pip(VENV_DIR):
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        if sys.platform.startswith("linux") and try_fix_missing_ensurepip_linux(python_cmd):
            step(f".venv 재생성 중 ({' '.join(python_cmd)})")
            result = subprocess.run([*python_cmd, "-m", "venv", str(VENV_DIR)])
        if result.returncode != 0 or not venv_has_pip(VENV_DIR):
            if VENV_DIR.exists():
                shutil.rmtree(VENV_DIR)
            banner(
                False,
                "venv에 pip이 없습니다 (ensurepip 누락). 아래 명령 실행 후 태스크를 다시 실행하세요:\n"
                "  Debian/Ubuntu: sudo apt-get install -y python3.12-venv\n"
                "  Fedora/RHEL:   sudo dnf reinstall -y python3.12\n"
                "  Arch:          sudo pacman -S python\n"
                "  macOS:         brew reinstall python@3.12\n"
                "  Windows:       python.org 설치 프로그램을 다시 실행해 'pip' 옵션이 "
                "체크되어 있는지 확인 후 재설치",
            )
            return 1

    banner(True, f".venv 생성 완료 ({' '.join(python_cmd)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
