# Signal-Vision — 수신호 인식 (산업 수신호 → 공정 기계 제어용 분류 결과)

웹캠 영상에서 작업자의 **수신호(hand signal)** 를 실시간 인식해 분류 결과를 내보내는 프로젝트.
**이 저장소의 범위는 분류 결과를 내보내는 데까지**이고, 그 결과를 ROS 2로 받아 Gazebo
시뮬레이션의 공정 기계를 제어하는 파트는 별도 프로젝트 [`../Mocap-Sim`](../Mocap-Sim)이 담당한다.

```
웹캠 → MediaPipe 랜드마크(손+상체 포즈) → LSTM 분류 → 결과 발행(rosbridge/UDP)
```

## 셋업

Python **3.12 필수** (`pyproject.toml`에 `>=3.12,<3.13`로 고정, 의존성도 `==`로 고정 — 임의로
버전을 올리지 않는다).

- **VSCode**: `Cmd/Ctrl+Shift+B` (기본 빌드 태스크 "프로젝트 셋업") 한 번으로
  venv 생성 → 패키지 설치 → import 검증까지 자동. 이후 인터프리터로 `.venv` 선택
  (`⌘⇧P → Python: Select Interpreter`) 후 F5.
- **수동**:
  ```bash
  # macOS
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -e .

  # Windows
  py -3.12 -m venv .venv
  .venv\Scripts\python -m pip install -e .
  ```

## 현재 상태

초기 단계 프로젝트. `src/capture/`, `src/dataset/`은 실제 동작하는 코드가 있고,
`src/training/`, `src/inference/`는 아직 구현 전이다 (`doc/design.md`, `doc/ml-flow.md` 기반으로
앞으로 작성). 테스트/린터는 아직 없다.

## 사용법 (VSCode 태스크: Terminal → Run Task)

| 태스크 | 용도 | 상태 |
|---|---|---|
| 실행: 랜드마크 시각화 (capture) | 웹캠 + 손/포즈 랜드마크 확인 (프레이밍 점검, 헤드리스/Docker 불가) | 동작함 |
| 실행: 데이터 수집 (dataset) | 라벨 입력 → 자동 연속 수집 (클릭 버튼 UI) | 동작함 |
| 실행: 모델 학습 (training) | `asset/` 스캔 → LSTM 학습 → 모델 저장 | 예정 (`train.py` 없음) |
| 실행: 실시간 추론 (inference) | 웹캠 실시간 인식 | 예정 (`predict.py` 없음) |
| 시뮬 데모: Gazebo + 인식기 (Mac) | `../Mocap-Sim` 연동 데모 | 예정 (`publisher.py` 없음, Mocap-Sim 쪽 의존) |

같은 모듈을 터미널에서 직접 실행할 수도 있다: `.venv/bin/python -m <module>`
(예: idle 라벨 수집 — `.venv/bin/python -m src.dataset.collect --label idle --idle`)

## 수신호 클래스

`asset/`의 라벨 폴더가 곧 클래스 — **코드에 클래스 이름을 하드코딩하지 않는다.** 학습이
폴더를 스캔해 라벨 목록을 구성하고 모델 파일에 함께 저장, 추론은 그것만 읽는다.

현재 설계된 목록 (`doc/design.md` 기준, 라벨명은 기계 명령어로 쓰이는 영어 snake_case):

| 라벨 | 동작 | 기계 행동 |
|---|---|---|
| `stop` | 손바닥 정면으로 내밀기 | 즉시 정지 |
| `slow` | 손바닥 아래로 눌러 내리기 | 감속 |
| `come` | 손짓해 부르기 | 접근 |
| `back` (미수집) | 손등으로 밀어내기 | 후진 |
| `left_go` / `right_go` | 좌/우로 팔 젓기 | 선회 |
| `idle` | 평상시 동작 (`--idle`로 수집, 필수 — 오작동 방지) | 직전 상태 유지 |

새 신호는 그 이름으로 수집하고 재학습하면 추가된다.

## 데이터 수집 (`src/dataset/`)

학습 자체는 아직 없는 `src/training/`의 몫이고, 이 폴더는 학습에 쓸 `(30, 150)` 랜드마크
시퀀스를 `asset/<라벨>/*.npy` 형식으로 모으는 역할만 한다. `src/capture/extractor.py`의
특징 추출기를 그대로 가져다 써서 수집·학습·추론이 같은 특징 형식을 유지한다.

### `collect.py` — 웹캠 수집 (학습 데이터의 유일한 소스)

카메라 앞에서 수신호를 반복해 보여주면 자동으로 "준비(READY) → 녹화(REC) → 저장"을
목표 개수만큼 반복 수집한다.

```bash
python -m src.dataset.collect --label stop [--samples 30] [--frames 30] [--prep 1.5] [--idle]
```

- `--label`: 라벨명 — 위 "수신호 클래스" 표 참고
- `--idle`: 손 감지 품질 필터를 끈다 (idle처럼 손이 안 보이는 게 정상인 라벨용)
- 조작: START/PAUSE 버튼 또는 SPACE, QUIT 버튼 또는 q/ESC
- 품질 보호: 손 감지 프레임이 절반 미만이면 저장하지 않고 SKIP
- 저장 형식: `asset/<label>/<타임스탬프>.npy`, shape `(frames, 150)`

## 문서

- [doc/design.md](doc/design.md) — 아키텍처·설계 결정 (특징 150차원, 안전 규칙, 전달 방식, ODD 고장 유형 대응표)
- [doc/ml-flow.md](doc/ml-flow.md) — 데이터 수집 → 학습 → 추론 플로우, 데이터셋 조사, 체크리스트
- [doc/AGENT.md](doc/AGENT.md) — 코딩 에이전트용 원본 프로젝트 브리프
