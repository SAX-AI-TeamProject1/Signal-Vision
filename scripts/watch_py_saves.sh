#!/bin/bash
# Watches a directory for .py file saves (from any editor) and stamps a
# "# Last updated: YYYY-MM-DD" header via update_date_header.py.
#
# Usage: watch_py_saves.sh [directory]   (defaults to this repo's root)
#
# Requires fswatch (macOS: brew install fswatch).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${1:-$SCRIPT_DIR/..}"
STAMP_SCRIPT="$SCRIPT_DIR/update_date_header.py"

echo "Watching $DIR for .py saves (Ctrl+C to stop)..."

# fswatch's -i (include) filter takes priority over -e (exclude) regardless
# of order, so vendored .py files under .venv would slip through an -e
# exclude. Instead, only pre-filter by extension in fswatch, then reject
# noisy/vendored paths in the loop below.
fswatch -0 -i "\\.py$" -e ".*" "$DIR" | while IFS= read -r -d "" file; do
  case "$file" in
    *"/.venv/"*|*"/.git/"*|*"/node_modules/"*) continue ;;
  esac
  python3 "$STAMP_SCRIPT" "$file"
done
