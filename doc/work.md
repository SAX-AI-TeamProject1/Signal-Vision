# 작업 이력 (Work Log)

`git log` 기준으로 정리한 날짜별 구현 내역. 최신 내용은 `CLAUDE.md`와 `doc/` 하위 문서를 우선 참고.

## 2026-07-10 — 프로젝트 초기화
- 레포 생성, `.gitignore` / `LICENSE` / `README.md` 초기 커밋.

## 2026-07-13 — 파이프라인 뼈대 + 캡처/데이터셋 단계
- VSCode 프로젝트 설정(`tasks.json`, `launch.json`), 설계 문서(`doc/AGENT.md`, `doc/design.md`, `doc/ml-flow.md`) 작성, `pyproject.toml` 추가.
- `src/capture/extractor.py`(공용 피처 추출기) + `src/capture/landmark_viewer.py`(랜드마크 시각화 도구) 구현, 사전학습 `models/sign_classifier.pt` 반입.
- `src/dataset/collect.py`(라벨링 데이터 수집 UI) 구현. 초기에 `import_jester.py`(Jester 외부 데이터셋 매핑)를 추가했다가 같은 날 삭제(자체 수집 데이터로 방향 확정).
- venv/윈도우 경로 이슈, 인라인 문자열 문법 오류 등 다수 수정.
- `scripts/task_output.py`, `scripts/check_env.py`, `scripts/install_deps.py`, `scripts/setup_venv.py`, `src/console.py`(터미널 상태 포맷터) 추가 — VSCode 빌드 태스크용 부트스트랩 도구.

## 2026-07-14 — 학습/추론/발행 단계 구현
- `src/inference/predict.py`(실시간 추론), `src/inference/bridge.py`(rosbridge/UDP 결과 발행), `src/inference/hud.py`(확률 패널 오버레이) 구현.
- `src/training/train.py`(LSTM `SignLSTM` 학습), `src/training/augment.py`(노이즈/드롭아웃/타임스트레치 등 증강), `src/training/stress_test.py`(ODD 조건별 강건성 측정) 구현.
- `.gitignore`에 수집 데이터(`asset/`) 등 제외 규칙 정리.

## 2026-07-15 — 휴머노이드 포팅 기능 추가 (이후 제거됨)
- `--publish-pose` 상반신 스트리밍 기능("휴머노이드 포팅") 을 `extractor.py` / `bridge.py` / `predict.py` / `train.py` / `augment.py`에 걸쳐 추가.
- 셋업 스크립트(`install_deps.py`, `setup_venv.py`) 보강.

## 2026-07-16 — 설계 롤백: 휴머노이드 포팅 제거
- 휴머노이드 포팅 관련 코드를 `extractor.py` / `bridge.py` / `predict.py`에서 되돌리고 `CLAUDE.md`, `doc/design.md`에서 관련 설명 삭제 — 스코프를 "분류 결과 발행까지"로 재확정.

## 2026-07-18 — 버그 조사
- 버그 재현/기록(`bug.txt`) 외 특이 변경 없음.

## 2026-07-20 — 문서 정리 + 배포 방식 확정
- 레포/시뮬레이터 명칭 변경 반영(`Mocap-Sample`→`Signal-Vision`, `Mocap-Sim`→`Signal-Simulation`) 문서 경로 정리 (`README.md`, `doc/AGENT.md`, `doc/design.md`).
- 배포 방식 결정: Vision·Simulation을 같은 Linux 머신에서 구동하되 전송 방식은 rosbridge websocket 유지 (rclpy 네이티브 발행 전환은 검토 후 기각).

## 2026-07-21 — 사소한 정리
- `empty.txt` 관련 소규모 수정.

## 2026-07-24 — 증강 파이프라인 긴급 수정
- `src/capture/extractor.py`, `src/inference/predict.py`, `src/inference/hud.py` 수정 및 `models/sign_classifier.pt` 재학습본 반영 (`fix : agument emergency`).
- `doc/AGENT.md` 갱신, 브랜치 병합.

## 2026-07-26 - 코드 리펙토링
- 지저분한 주석 처리 (docustring)와 코드 리펙토링 수정