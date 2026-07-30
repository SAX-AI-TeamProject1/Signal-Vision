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

## 2026-07-29 — 좌표 정규화 버그 수정 (한인혁)

- `left_go`/`right_go` 수신호가 실제 동작과 무관하게 화면 시작 위치에 따라 분류되는 문제 발견 → `src/capture/extractor.py`의 `FeatureExtractor.vector()`에 어깨 중심 원점 + 어깨너비 스케일 정규화(`_reference_frame()`) 추가.
- 좌표계 변경으로 `src/training/augment.py`의 스케일/렌즈왜곡 증강 기준점(화면 중앙 0.5 가정)이 어긋나던 문제도 함께 수정 (원점 0 기준으로 통일).
- 정규화 이전에 모은 `asset/left_go`, `asset/right_go` 삭제 후 라벨당 120개씩 재수집·재학습, 실시간 추론 정상 동작 확인.
- 문제 경위·대응·다음 수집 체크리스트를 기록하는 `doc/data_collection_log.md` 신설.
- 커밋 `2b539cb`(`fix: normalize landmarks to shoulder-relative coordinates`)로 반영, `Signal_Collector` 브랜치에 푸시.

## 2026-07-29 — idle 라벨 추가 시도 + 실시간 left/right 회귀 (미해결, 데이터 재수집으로 결론)

- `idle` 라벨 120개 수집(라벨이 `idle`이면 손 감지 품질 필터가 자동으로 꺼지도록 `src/dataset/collect.py` 수정) 후 3클래스(idle/left_go/right_go)로 재학습.
- idle이 offline test(94~96%)는 잘 나오는데 실시간에서 신뢰도 80%(확정 임계값)를 못 넘는 문제 확인. 손 소실 증강(`augment.py`의 `P_HAND_DROPOUT`), 손 검출 신뢰도 임계값 불일치(collect.py 0.5 vs predict.py 기본 0.4) 등을 의심해 검증했으나 둘 다 원인이 아님으로 확인.
- idle 개선을 위해 `SignLSTM.forward()`의 분류head 입력을 "마지막 타임스텝만" → "[마지막 | 전체 평균] concat"으로 변경, 총 3차례 재학습(순수 concat / concat + head dropout 0.5 + weight_decay 1e-4) 시도. offline test는 매번 양호했지만, 실시간에서는 idle은 되어도 `left_go`/`right_go`가 반대로 나오는 현상이 반복 발생.
- `git checkout`으로 `train.py`를 원래 구조(마지막 타임스텝만 사용)로 되돌려 재학습했으나 실시간 left/right 반전이 사라지지 않음 — 이 과정에서 `torch.manual_seed()`가 아예 없어 재학습할 때마다 모델 가중치 초기화가 랜덤이었던 것을 발견해 `42`로 고정.
- 시드 고정 후 재학습해도 동일 증상 지속. 손 좌우 슬롯(handedness) 의존 여부, 카메라 미러 모드(수집 시점 vs 실시간) 일치 여부, `asset/left_go`·`asset/right_go` 라벨 스왑 가능성을 차례로 점검했으나 결정적 원인을 특정하지 못함.
- **결론(미해결)**: 코드(정규화·증강·풀링 구조·시드) 쪽 문제라기보다, 이 세션 안에서 반복 재수집하는 과정에서 데이터 자체의 일관성이 깨졌을 가능성이 높다고 보고 원인 규명을 중단. `idle`/`left_go`/`right_go` 데이터를 전부 삭제하고 한 세션에서 처음부터 다시 수집하기로 결정 (다음 작업).
- `train.py`에 `torch.manual_seed(42)` 추가 — 데이터 셔플/증강(`np.random.default_rng(42)`)은 이미 시드 고정돼 있었지만 모델 가중치 초기화(PyTorch 자체 RNG)는 고정된 적이 없어 재학습마다 결과가 달라졌던 것을 발견해 고정 (재현 가능한 학습을 위한 선행 조건).

## 2026-07-29 — 데이터 재수집으로 left/right 문제 해결 + 전체 7클래스 확장

- `idle`/`left_go`/`right_go` 삭제 후 한 세션에서 `left_go`/`right_go`부터 재수집 → 방향 반전 현상 사라짐(offline test 100%, 방향도 정상). 다만 실시간 신뢰도가 58~71%로 확정 임계값(80%)에 못 미침 — 120개로는 실제 카메라 노이즈까지 커버하기엔 부족하다고 판단, 라벨당 30개씩 추가 수집(180개)하기로 함.
- 이어서 나머지 라벨(`back`, `come`, `idle`, `slow`, `stop`)도 각 120개씩 수집 완료 — 전체 7클래스(`asset/`) 데이터셋 완성.
- 전체 재학습 결과: test 정확도 93.8%. `left_go`/`right_go`는 offline 100%이고 실시간 추론에서도 정상 확정 확인. `back`/`come`/`idle` 세 라벨 사이에 일부 혼동 존재(예: come↔back, come/idle↔right_go 소수 오분류) — 다음 개선 대상.
- 모델 재학습 결과 `models/sign_classifier.pt`에 반영.

## 2026-07-29 — EMERGENCY 오작동 원인 규명 및 수정

- 정상적으로 수신호를 전환할 때(느린 전환/빠른 전환/복잡도 변화 없는 전환 모두 포함)마다 `predict.py`가 "렌즈 블러 + 인식 저하 [즉시]"로 EMERGENCY를 띄우는 문제 발견.
- `LensHealthMonitor`의 선명도/기준치 비율을 실시간으로 찍어보는 디버그 로그를 추가해 확인한 결과: idle 상태(선명도 기준치 대비 103~109%)와 실제 손동작 중(27~41%) 사이에 구조적인 선명도 차이가 있고, 이 낮은 수준이 손동작을 하는 내내 지속됨(순간적인 튐이 아님) — 손이 카메라에 가깝게/크게 들어오며 생기는 정상적인 현상으로, 렌즈 이상과 무관.
- 세션 시작 캘리브레이션이 idle 상태 기준으로 잡히다 보니, 이후 손동작을 하는 모든 순간이 "블러"로 오판되어 신호 전환 때마다 EMERGENCY가 걸렸던 것.
- 1차 수정: `SignalStabilizer.update()`(신호 인식 중 신뢰도 저하 처리)에서 블러 체크를 제거 — 신뢰도 저하는 항상 유예(`release_grace`)를 두고 해제, 블러 기반 즉시 해제는 `mark_person_absent()`(사람 미감지)에만 남김.
- 이후 팀 회의에서 "Signal-Vision이 사람 미감지를 이유로 EMERGENCY를 띄우는 것 자체가 책임 범위 밖"이라는 결론 → **2차 수정(최종)**: `emergency`를 "카메라 인지 능력 저하(렌즈 블러 등)"만 의미하도록 완전히 분리. `SignalStabilizer`에 `set_emergency(is_blurred)`를 신설해 사람 감지 여부·신호 확정 상태와 무관하게 매 프레임 독립적으로 갱신하고, `mark_person_absent()`는 "인식 불가" 보고만 할 뿐 emergency는 건드리지 않음. 사람 미감지 시 실제로 정지할지는 twistmux 등 상위 시스템이 라이다·YOLO까지 종합해 판단할 몫 — Signal-Vision은 "신호를 못 읽는다"는 사실만 보고.
- 디버그용으로 추가했던 `LensHealthMonitor.baseline` 프로퍼티와 로그 출력 블록은 원인 확인 후 제거.
- 3차 시도: "캘리브레이션 때 손을 보여준다"는 절차로 임시 우회를 시도했으나, 그러면 기준치 자체가 낮게(손동작 수준으로) 잡혀서 이번엔 진짜 렌즈 초점을 흐리게 해도(1분 이상 유지) EMERGENCY가 안 뜨는 반대 문제가 실측으로 확인됨 — 기준치를 얼마로 잡든 "손동작 오탐"과 "진짜 블러 감지"를 하나의 고정값으로는 동시에 만족 못 시킨다는 게 증명됨.
- 4차 시도: 손 감지 여부에 따라 비율을 이원화(`BLUR_RATIO_DROP_HAND_PRESENT=0.15` vs `BLUR_RATIO_DROP=0.5`)했으나, 실측 결과 "손 없음+블러"는 안 뜨고 "손 있음+블러"만 뜨는 등 두 비율이 상황에 따라 들쭉날쭉하게 걸려 신뢰할 수 없었음. 또한 화면 일부 고정 영역(예: 위쪽)만 보는 대안도, 손이 없는 다른 위치(예: 아래쪽)에 빗방울이 묻으면 못 잡는다는 반례로 기각.
- **최종 수정**: 화면 전체 대신, **그 순간 손이 차지하는 픽셀 영역(바운딩 박스, 여유 패딩 포함)만 선명도 계산에서 제외**하도록 변경 (`_hand_exclude_mask()` 신설, `_sharpness_score()`가 `exclude_mask` 인자를 받도록 확장). 손이 없으면 화면 전체, 있으면 그 위치만 빼고 나머지 전체 화면을 항상 동일한 기준(`BLUR_RATIO_DROP=0.5`)으로 감시 — 손 위치와 무관하게 화면 어디에 실제 렌즈 이상이 생겨도 잡아내면서, 손 자체로 인한 정상적인 선명도 저하는 제외된다. `LensHealthMonitor.update()`는 이제 `hand_result`를 받아 내부에서 마스크를 계산한다.

