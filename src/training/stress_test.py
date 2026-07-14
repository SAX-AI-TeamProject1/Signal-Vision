# Last updated: 2026-07-14
"""악조건 스트레스 테스트: 조건별로 test 셋을 오염시켜 모델 강건성을 측정한다.

학습과 동일한 분할(seed 42)의 test 셋에 augment.py의 변형을 조건별로 강제
적용(p=1)하고 정확도 하락을 비교한다. 무작위 변형이므로 여러 번 반복한 평균을 본다.

    python -m src.training.stress_test [--repeats 20]

주의: 여기서 측정하는 것은 "좌표가 오염됐을 때 분류기의 강건성"이다.
저조도·연기가 MediaPipe 추출 자체를 실패시키는 효과는 실측(불 끄기, 가림 등)으로
확인해야 한다 — 그 경우는 안전 가드(인식 불가 → 정지)가 담당한다.
"""

import argparse # 커맨드라인 인자(--model, --repeats 등) 파싱

import numpy as np # 조건별 반복 정확도 평균 계산
import torch # 저장된 모델 로드·평가

import src.training.augment as aug # 모듈 자체를 가져와 아래에서 P_*/강도 상수를 일시적으로 덮어쓰기 위함
# train.py와 동일한 데이터 로드·분할·모델·평가 로직을 재사용 (중복 구현 금지) —
# split_dataset이 train.py와 같은 시드(42)를 쓰므로 학습 때와 동일한 test 셋을 다시 얻는다.
from src.training.train import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, SignLSTM, evaluate, load_dataset, split_dataset

# 조건 이름 → augment 모듈 확률/강도 오버라이드.
# 값을 1.0으로 주면 "이 변형이 무조건 적용된다"는 뜻이라, 그 조건만 순수하게 측정할 수 있다.
CONDITIONS = {
    "깨끗함 (기준)": None,
    "좌표 노이즈 (센서·저조도)": {"P_NOISE": 1.0, "NOISE_STD": 0.02},
    "손 소실 (연기·가림)": {"P_HAND_DROPOUT": 1.0, "DROPOUT_FRAMES": (3, 10)},
    "시간 신축 (동작 속도)": {"P_TIME_WARP": 1.0},
    "이동·스케일 (위치·거리)": {"P_TRANSLATE": 1.0, "P_SCALE": 1.0},
    "렌즈 왜곡": {"P_DISTORT": 1.0},
    "종합 (전부 동시)": {"P_NOISE": 1.0, "P_HAND_DROPOUT": 1.0, "P_TIME_WARP": 1.0,
                    "P_TRANSLATE": 1.0, "P_SCALE": 1.0, "P_DISTORT": 1.0},
}

# CONDITIONS의 오버라이드들이 건드릴 수 있는 augment 모듈 상수 전체 목록.
# 아래에서 "이번 조건에 안 쓰는 나머지는 0으로 꺼야" 그 조건 하나만 순수하게 측정되므로,
# 매 조건마다 이 목록 전체를 기준으로 초기화(defaults/zeros)한다.
ALL_KEYS = ["P_NOISE", "P_HAND_DROPOUT", "P_TIME_WARP", "P_TRANSLATE", "P_SCALE",
            "P_DISTORT", "NOISE_STD", "DROPOUT_FRAMES"]
 

def main() -> None:
    """저장된 모델을 로드해 조건별(노이즈/손 소실/시간 신축/이동·스케일/렌즈 왜곡)로
    test 셋을 오염시키고, 깨끗한 test 셋 대비 정확도가 얼마나 떨어지는지 출력한다.

    핵심 트릭: augment.py의 P_*/강도 상수를 모듈 속성(setattr/getattr)으로 직접
    덮어써서 "이 조건만 100% 확률로 강제 적용"하게 만든 뒤, 측정이 끝나면 원래 값으로
    되돌린다 — train.py를 건드리지 않고도 augment_batch()의 동작을 조건별로 바꿔볼 수 있다.
    """
    parser = argparse.ArgumentParser(description="악조건 스트레스 테스트")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--repeats", type=int, default=20, help="무작위 변형 반복 횟수")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SignLSTM(cfg["input_dim"], cfg["hidden_dim"], len(ckpt["labels"]),
                     num_layers=cfg["num_layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    X, y, labels = load_dataset(args.data_dir)
    # 데이터 폴더가 모델 학습 당시와 달라졌으면(라벨 추가/삭제) 클래스 인덱스가 어긋나
    # 엉뚱한 라벨로 채점하게 되므로, 애초에 그런 조합으로는 측정을 진행하지 않는다.
    assert labels == ckpt["labels"], f"데이터 라벨 {labels} ≠ 모델 라벨 {ckpt['labels']}"
    # train.py의 split_dataset과 동일한 random_state=42라, 학습 때 모델이 한 번도
    # 보지 못한 바로 그 test 셋을 다시 얻는다 (train/val 데이터가 섞여 들어가지 않게).
    *_, X_test, y_test = split_dataset(X, y)
    print(f"\ntest 셋 {len(X_test)}개, 조건별 {args.repeats}회 반복 평균\n")

    defaults = {k: getattr(aug, k) for k in ALL_KEYS}  # 나중에 복원할 augment.py의 원래 값
    zeros = {k: 0.0 for k in ALL_KEYS if k.startswith("P_")}  # 이번 조건 외 나머지 변형은 확률 0으로 꺼둠

    print(f"{'조건':<24} 정확도")
    print("-" * 34)
    baseline = None
    for name, overrides in CONDITIONS.items():
        if overrides is None:
            # "깨끗함(기준)" — 증강 없이 원본 test 셋 그대로 평가해 비교 기준선을 만든다
            acc, _ = evaluate(model, X_test, y_test, device)
            baseline = acc
        else:
            # defaults(원래 값) → zeros(전부 끄기) → overrides(이번 조건만 강제 켜기) 순으로
            # 덮어써서, 최종적으로 "이번 조건만 100%, 나머지는 0%"인 상태를 만든다
            for k, v in {**defaults, **zeros, **overrides}.items():
                setattr(aug, k, v)
            rng = np.random.default_rng(7)  # 조건마다 같은 시드로 시작 — 조건 간 비교가 공정하도록
            accs = [evaluate(model, aug.augment_batch(X_test, rng), y_test, device)[0]
                    for _ in range(args.repeats)]  # 무작위 변형이라 여러 번 반복해 평균을 낸다
            for k, v in defaults.items():  # 다음 조건에 영향 없도록 원상 복구
                setattr(aug, k, v)
            acc = float(np.mean(accs))
        drop = "" if baseline is None or overrides is None else f"  ({(acc - baseline) * 100:+.1f}%p)"
        print(f"{name:<24} {acc * 100:5.1f}%{drop}")

    print("\n해석: 하락 폭이 큰 조건이 모델의 약점 — 해당 조건 증강 강도를 올리거나"
          "\n      그 조건의 실측 데이터를 보강한다. (MediaPipe 층 실패는 실측으로 확인)")


if __name__ == "__main__":
    main()
