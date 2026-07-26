#!/usr/bin/env python3
"""디렉토리를 폴링해 .py 저장을 감지하면 update_date_header.stamp()로 날짜 헤더를 찍는다.

fswatch 같은 OS별 셸 도구 대신 순수 Python 폴링으로 구현해 Windows/macOS/Linux에서
동일하게 동작한다 (AGENT.md 규약: 셸 스크립트 대신 Python으로 작성해 OS 의존 제거).

사용법: watch_py_saves.py [directory]   (기본값: 이 저장소 루트)
"""

import os
import sys
import time
from pathlib import Path

from update_date_header import stamp

EXCLUDE_DIRS = {".venv", ".git", "node_modules"}
POLL_INTERVAL_SEC = 1.0


def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def watch(root: Path) -> None:
    print(f"Watching {root} for .py saves (Ctrl+C to stop)...", flush=True)
    mtimes = {p: p.stat().st_mtime for p in iter_py_files(root)}
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        current = set(iter_py_files(root))
        for path in current:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtimes.get(path) != mtime:
                mtimes[path] = mtime
                stamp(str(path))
        for path in list(mtimes):
            if path not in current:
                del mtimes[path]


if __name__ == "__main__":
    root_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent
    watch(root_arg)
