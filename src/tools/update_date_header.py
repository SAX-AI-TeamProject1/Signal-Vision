#!/usr/bin/env python3
# Last updated: 2026-07-13
import sys, json, re, datetime, os

MARKER = "# Last updated:"


def resolve_file_path():
    # CLI usage: update_date_header.py <file_path>  (used by tools/watch_py_saves.py)
    if len(sys.argv) > 1:
        return sys.argv[1]
    # Stdin usage: JSON with {"tool_input": {"file_path": ...}} on stdin
    try:
        data = json.load(sys.stdin)
    except Exception:
        return None
    return data.get("tool_input", {}).get("file_path")


def stamp(file_path):
    if not file_path or not file_path.endswith(".py") or not os.path.isfile(file_path):
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    today = datetime.date.today().isoformat()
    new_line = f"{MARKER} {today}\n"

    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if len(lines) > insert_at and re.match(r"#.*coding[:=]", lines[insert_at]):
        insert_at += 1

    changed = False
    for i, line in enumerate(lines[:insert_at + 2]):
        if line.startswith(MARKER):
            if line != new_line:
                lines[i] = new_line
                changed = True
            break
    else:
        lines.insert(insert_at, new_line)
        changed = True

    # Only write when content actually changed, so the file watcher calling
    # this script doesn't re-trigger itself in an infinite save loop.
    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


if __name__ == "__main__":
    stamp(resolve_file_path())
