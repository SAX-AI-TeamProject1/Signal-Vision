# Last updated: 2026-07-27
"""VSCode 태스크 스크립트(setup_venv/install_deps/check_env) 공용 출력 포맷터.

진행 상황이 없으면 터미널이 멈춘 것처럼 보이므로, 오래 걸리는 작업 전에는
반드시 step()으로 진행 중임을 즉시 출력한다.
"""

import sys

_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_CYAN = "\033[36m"


def _enable_windows_ansi() -> None:
    """구형 cmd.exe(conhost)는 ANSI 색상 코드를 기본적으로 해석하지 않으므로 명시적으로 켠다.
    VSCode 통합 터미널은 이미 지원하므로 실패해도 조용히 무시한다."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        kernel32.SetConsoleMode(handle, 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING 포함
    except Exception:
        pass


_enable_windows_ansi()


def _color(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def step(message: str) -> None:
    """오래 걸리는 작업을 시작하기 직전에 호출 - 진행 중임을 즉시 보여준다."""
    print(_color(_CYAN, f">> {message}..."), flush=True)


def banner(ok: bool, message: str) -> None:
    """작업의 최종 성공/실패를 명확한 배너로 출력한다."""
    mark = "[OK]" if ok else "[FAIL]"
    line = "=" * 50
    text = f"\n{line}\n{mark} {message}\n{line}\n"
    print(_color(_GREEN if ok else _RED, text), flush=True)
