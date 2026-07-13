# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Detailed design docs live in `doc/` — read them, don't duplicate them here:
- `doc/AGENT.md` — original agent-facing project brief (Korean)
- `doc/design.md` — architecture, feature format, safety rules, ODD (operational design domain) fault table
- `doc/ml-flow.md` — data collection → training → inference flow, dataset decisions, checklist
- `doc/TODO.md` — progress log and open tasks (referenced by AGENT.md; may not exist yet — create it if asked to track progress there)

Repository docs and code comments are written in Korean; match that convention when editing them.

## Project overview

Industrial hand-signal recognition: captures webcam video (OpenCV), extracts hand + pose landmarks (MediaPipe), classifies dynamic hand signals with a PyTorch LSTM, and forwards the classification result over a rosbridge websocket. **This repo's scope ends at emitting the classification result** — a separate project (`../Mocap-Sim`) consumes it via ROS 2 to drive a Gazebo simulated machine (stop/slow/approach/reverse). Do not add ROS 2 / Gazebo code here.

Pipeline: camera → MediaPipe HandLandmarker + PoseLandmarker → 150-dim feature sequence (30 frames) → LSTM classifier → result publisher (rosbridge/UDP).

## Current state

This is an early-stage skeleton: `pyproject.toml`, `.vscode/`, docs, and `models/` (pretrained MediaPipe `.task` files + `sign_classifier.pt`) are in place. Under `src/`, only `src/capture/` has real code so far: `extractor.py` (the 150-dim feature extractor — also holds the shared model download/landmarker-creation/drawing helpers) and `landmark_viewer.py` (the visualization/verification script, which imports those shared helpers from `extractor.py`). Keep the dependency direction one-way — `landmark_viewer.py` (debug tool) depends on `extractor.py` (production path), never the reverse. The other modules described in the docs (`src/dataset/collect.py`, `src/training/train.py`, `src/inference/predict.py`, `src/inference/publisher.py`, `src/training/augment.py`, `src/training/stress_test.py`) are planned but not yet implemented. Expect to build these from the design in `doc/design.md` / `doc/ml-flow.md` rather than finding existing implementations.

## Setup and commands

Python **3.12 required** (pinned in `pyproject.toml`); dependencies are pinned with `==` — never loosen or bump versions on your own.

```bash
# macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .

# Windows
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .
```

In VSCode, `Cmd/Ctrl+Shift+B` runs the default build task ("프로젝트 셋업"), which creates the venv, installs the package, and verifies imports — it picks the right shell per OS from `.vscode/tasks.json`. After setup, select `.venv` as the interpreter before using F5.

No test suite or linter is configured yet.

Other VSCode tasks (`.vscode/tasks.json`) worth knowing about, all run through `.venv`'s Python as `python -m <module>`:
- `src.capture.landmark_viewer` — landmark extraction/visualization check
- `src.dataset.collect --label <label>` — collect labeled training samples (prompts for a label)
- `src.training.train` — train the LSTM classifier
- `src.inference.predict` — real-time inference
- a background task that watches `.py` saves and stamps a `# Last updated: YYYY-MM-DD` header via `scripts/update_date_header.py` / `scripts/watch_py_saves.sh` (requires `fswatch`)

Run any module the same way manually, e.g. `.venv/bin/python -m src.training.train`.

## Architecture

`src/` is organized as pipeline stages, each stage a subpackage:

- `src/capture/` — camera capture + MediaPipe landmark extraction: `extractor.py` (the shared feature extractor, plus shared model download/landmarker-creation/drawing helpers other capture-stage scripts import), `landmark_viewer.py` (visualization/verification tool, imports those helpers from `extractor.py`)
- `src/dataset/` — data collection scripts (`collect.py` planned) and external-dataset conversion (`import_jester.py` planned; kept as a tool even though Jester is not used for training — see below)
- `src/training/` — LSTM training (`train.py`), augmentation (`augment.py`), robustness stress testing (`stress_test.py`) — all planned
- `src/inference/` — real-time prediction (`predict.py`) and result publishing (`publisher.py`, rosbridge/UDP) — both planned
- `asset/` — collected landmark samples, one folder per label (gitignored; not present in this checkout)
- `models/` — MediaPipe `.task` models and the trained `sign_classifier.pt`

### Key design constraints (see `doc/design.md` for full detail)

- **Feature format is load-bearing**: every stage (collect / convert / train / infer) must go through `src/capture/extractor.py` so features stay consistent between training and inference. Feature vector is 150-dim per frame: left hand (63) + right hand (63) + upper-body pose (8 joints × xyz = 24: shoulders, elbows, wrists, hips). Missing hand/pose data is zero-filled. Samples are `asset/<label>/<timestamp>.npy` with shape `(30, 150)` (30-frame sliding window).
- **Never hardcode the class list.** Labels come from scanning `asset/` subfolder names at train time and are saved alongside the model weights; inference loads labels from the model file, not from code.
- **Safety default**: below-confidence or undetected-person input must resolve to `"unknown"`, which downstream treats as "stop" — never let a low-confidence guess pass through as a real signal.
- **Cross-platform (Windows + macOS) is a hard requirement**:
  - Always use `pathlib.Path`; never hardcode path separators or absolute paths.
  - Camera backend selection (`cv2.VideoCapture`) should branch on `platform` only when necessary (Windows: `CAP_DSHOW`/`CAP_MSMF`, macOS: `CAP_AVFOUNDATION`).
  - Write tooling in Python, not shell scripts, to avoid OS-specific scripts (the one exception, `scripts/watch_py_saves.sh`, is dev tooling, not part of the pipeline).
  - Do not put webcam-using code in Docker (no camera access from containers on either OS).
- Result delivery is rosbridge websocket (`src/inference/publisher.py`, topic `/hand_signal`, JSON payload `{"signal", "confidence", "timestamp"}`), with a UDP backend for local debugging. Publishing must keep running even if no rosbridge server is connected (auto-reconnect), and should emit both on state change and on a periodic heartbeat.
- Training data is self-collected only (`asset/`) — an earlier experiment with the Jester dataset showed a large train/deploy domain gap (motion style differences, unmirrored selfie-view causing left/right confusion) and was abandoned; don't reintroduce external datasets as primary training data without checking `doc/ml-flow.md` §1-2 first.
