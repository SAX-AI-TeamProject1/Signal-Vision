# Last updated: 2026-07-27
'''
train.py

수신호 분류 모델 학습 스크립트

asset/의 라벨 폴더를 스캔해 클래스를 동적으로 구성하고 (하드코딩 금지),
LSTM 시퀀스 분류 모델을 학습한 뒤 라벨 목록과 함께 저장한다.

사용법:
    python -m src.training.train [--data-dir asset] [--epochs 300] [--hidden 64]

플로우:
    데이터 로드 → 층화 분할(70/15/15) → LSTM 학습 (early stopping)
    → test 평가 (혼동 행렬 포함) → models/sign_classifier.pt 저장 (라벨 목록 포함)
'''

import argparse # 커맨드라인 인자(--epochs, --hidden 등) 파싱
from pathlib import Path # OS 무관 경로 처리 (Windows/macOS 공용, design.md 규약)

import numpy as np # 데이터셋 로드·셔플·정확도 계산
import torch # 모델 정의·학습·저장
from torch import nn # LSTM·Linear·Dropout 레이어
from sklearn.metrics import classification_report, confusion_matrix # test 평가 리포트
from sklearn.model_selection import train_test_split # 층화(stratified) 분할
from tqdm import trange # epoch 진행 게이지바 (터미널에서 진행률·ETA 표시)

from src.training.augment import augment_batch # 학습 배치에 실시간으로 악조건 변형 주입 (doc/ml-flow.md 2절)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "asset"  # 라벨별 수집 데이터가 쌓이는 위치 (dataset/collect.py 출력)
DEFAULT_MODEL_PATH = ROOT / "models" / "sign_classifier.pt"  # 학습 결과 저장 위치 (predict.py가 읽음)


class SignLSTM(nn.Module):
    '''랜드마크 시퀀스 (batch, T, 150) → 수신호 클래스 로짓 (batch, num_classes).

    구조: LSTM(다층)으로 시퀀스를 처음부터 끝까지 훑어 매 타임스텝의 은닉 상태를 뽑고,
    그중 마지막 타임스텝의 은닉 상태 하나만 Dropout → Linear 분류head에 넣어 클래스를 정한다.
    수신호는 손이 멈춘 순간의 자세가 아니라 30프레임에 걸친 궤적(동적 제스처)이 의미를
    만들기 때문에, 시퀀스 전체를 순서대로 읽어 "지금까지 본 동작을 요약한 상태"를 유지할 수
    있는 LSTM을 쓴다 — 마지막 시점엔 그 요약(은닉 상태)에 동작 전체의 흐름이 반영되어
    있다는 전제다 (doc/ml-flow.md 3절, doc/design.md 근거).

    기본 하이퍼파라미터(hidden=64, layers=2)는 라벨당 30~100개 수준의 소량 데이터에서
    "가장 단순한 모델로 시작해 부족할 때만 복잡도를 올린다"는 원칙을 따른 값이다.

    Args:
        input_dim: 프레임당 특징 차원 (150 — extractor.FEATURE_DIM과 일치해야 함)
        hidden_dim: LSTM 은닉 상태 크기
        num_classes: 분류할 수신호 클래스 개수 (asset/ 라벨 폴더 수)
        num_layers: LSTM 층 수 (쌓을수록 표현력↑, 과적합 위험↑ — 소량 데이터라 2층에서 시작)
        dropout: 과적합 방지용 드롭아웃 확률 (레이어 내부 + 분류head 앞 양쪽에 적용)
    '''

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,  # 입력 텐서 모양을 (batch, time, feature)로 — 아래 forward()의 x와 일치
            dropout=dropout if num_layers > 1 else 0.0,  # 레이어 1개면 층 사이 드롭아웃 자체가 의미 없어 0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),  # 과적합 방지 — 소량 데이터(라벨당 30~100개)라 특히 중요
            nn.Linear(hidden_dim, num_classes),  # 은닉 상태 → 클래스별 로짓(정규화 안 된 점수)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, input_dim) 시퀀스 배치 → (batch, num_classes) 로짓."""
        out, _ = self.lstm(x)  # out: (batch, T, hidden_dim) — 매 타임스텝의 은닉 상태
        return self.head(out[:, -1])  # 마지막 타임스텝의 은닉 상태로 분류


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    '''asset/<라벨>/*.npy 를 스캔해 (X, y, 라벨목록)을 반환한다.

    라벨은 코드에 하드코딩하지 않고 폴더 이름을 정렬(sorted)해서 그대로 클래스 목록으로
    쓴다 — 이 순서가 y의 클래스 인덱스이자, 나중에 체크포인트에 저장되어 predict.py가
    그대로 읽어 쓰는 순서다 (라벨 목록이 어긋나면 클래스 인덱스가 뒤바뀌어버린다).
    '''
    labels = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and list(d.glob("*.npy")))
    if len(labels) < 2:
        raise SystemExit(f"클래스가 2개 이상 필요합니다. 현재: {labels} ({data_dir})")
    xs, ys = [], []
    for idx, label in enumerate(labels):
        for f in sorted((data_dir / label).glob("*.npy")):
            seq = np.load(f)
            xs.append(seq.astype(np.float32))
            ys.append(idx)  # 이 파일이 속한 라벨의 인덱스 (labels 리스트 순서 기준)
    X = np.stack(xs)  # (N, 30, 150) — 전체 샘플
    y = np.array(ys)  # (N,) — 각 샘플의 클래스 인덱스
    for idx, label in enumerate(labels):
        n = int((y == idx).sum())
        print(f"  {label}: {n}개" + ("  ⚠ 10개 미만 — 추가 수집 권장" if n < 10 else ""))
    return X, y, labels


def split_dataset(X, y):
    '''train 70 / val 15 / test 15 층화 분할.

    stratify=y로 각 분할에 클래스 비율이 원본과 동일하게 유지되도록 한다 — 그렇지 않으면
    운 나쁘게 val/test에 특정 라벨이 거의 없거나 몰려서 평가가 왜곡될 수 있다.
    random_state=42로 고정해, 같은 데이터로 다시 실행해도 항상 같은 분할이 나오게 한다
    (stress_test.py가 같은 test 셋을 다시 얻어 비교하려면 이 시드가 반드시 같아야 한다).
    '''
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

    

def evaluate(model, X, y, device) -> tuple[float, np.ndarray]:
    '''정확도와 예측값을 반환한다 (val/test 평가, stress_test.py도 재사용).'''
    model.eval()  # 드롭아웃 등 학습 전용 동작을 끄고 항상 같은 출력이 나오게 고정
    with torch.no_grad():  # 평가는 역전파가 필요 없으므로 그래디언트 추적을 꺼서 메모리·속도 절약
        logits = model(torch.from_numpy(X).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()  # 클래스별 점수 중 가장 큰 인덱스 = 예측 클래스
    return float((pred == y).mean()), pred


def main() -> None:
    '''데이터 로드 → 층화 분할 → LSTM 학습(early stopping) → test 평가 → 모델 저장.

    val 정확도가 --patience epoch 동안 개선되지 않으면 조기 종료하고, 학습 중 가장
    좋았던 시점의 가중치(best_state)를 최종 모델로 채택한다 — 마지막 epoch이 아니라
    "가장 잘한 시점"을 저장해야 과적합된 뒷부분 epoch을 피할 수 있다.
    '''
    parser = argparse.ArgumentParser(description="수신호 분류 모델 학습")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30, help="early stopping 대기 epoch")
    parser.add_argument("--no-augment", action="store_true",
                        help="악조건 증강(도메인 랜덤화) 끄기 — 기본은 켜짐")
    args = parser.parse_args()

    # 데이터 셔플/증강(np.random.default_rng(42))은 이미 시드 고정돼 있었지만, 모델 가중치
    # 초기화는 PyTorch 자체의 별도 난수 생성기를 써서 그동안 고정된 적이 없었다 — 매 실행마다
    # 결과가 달라져 재현·디버깅이 불가능했으므로 여기서 고정한다.
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"데이터 로드: {args.data_dir}")
    X, y, labels = load_dataset(args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    print(f"분할: train {len(X_train)} / val {len(X_val)} / test {len(X_test)}  (device: {device})")

    model = SignLSTM(X.shape[2], args.hidden, len(labels), num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류 표준 손실 (로짓 + 정답 인덱스를 직접 받음)

    train_y = torch.from_numpy(y_train).long().to(device)
    use_augment = not args.no_augment
    # 시드 고정 — 같은 --data-dir로 재실행해도 증강·셔플 결과가 재현 가능하도록.
    # np.random.permutation 같은 전역 함수 대신 이 rng 하나만 계속 써서, 실행 한 번의
    # 무작위성이 전부 이 시드 하나로 결정되게 한다.
    rng = np.random.default_rng(42)
    print(f"악조건 증강(도메인 랜덤화): {'켜짐 — 노이즈/손 소실/시간 신축/이동·스케일/렌즈 왜곡' if use_augment else '꺼짐'}")

    best_val, best_state, wait = 0.0, None, 0  # 지금까지 최고 val 정확도 / 그때의 가중치 / 그 뒤 개선 없던 epoch 수
    # trange(range용 tqdm 단축형)로 감싸면 진행률·경과/예상 시간이 담긴 게이지바가 자동으로
    # 표시된다. pbar.write()로 출력하는 줄은 게이지바를 깨지 않고 그 위에 이력으로 쌓인다.
    pbar = trange(1, args.epochs + 1, desc="학습", unit="epoch")
    for epoch in pbar:
        model.train()  # 드롭아웃 등을 학습 모드로 켬 (evaluate()의 model.eval()과 대비)
        perm = rng.permutation(len(X_train))  # 매 epoch 학습 순서를 섞어 배치 구성이 매번 달라지게
        for i in range(0, len(X_train), args.batch):
            idx = perm[i : i + args.batch]
            xb = X_train[idx]
            if use_augment:
                xb = augment_batch(xb, rng)  # 원본(X_train)은 보존, 이번 배치 사본에만 변형
            optimizer.zero_grad()  # 이전 스텝의 그래디언트가 누적되지 않도록 초기화
            loss = criterion(model(torch.from_numpy(xb).to(device)), train_y[idx])
            loss.backward()   # 역전파로 각 파라미터에 대한 그래디언트 계산
            optimizer.step()  # 계산된 그래디언트로 파라미터 갱신 (Adam)

        val_acc, _ = evaluate(model, X_val, y_val, device)
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            # detach().cpu().clone(): 그래디언트 연결을 끊고 별도 메모리로 복사 — 이후 model이
            # 계속 학습되며 값이 바뀌어도 이 시점의 가중치 스냅샷은 영향받지 않게 보존
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1  # 개선 없는 epoch이 --patience번 연속되면 아래에서 조기 종료

        # 게이지바 끝에 현재 수치를 붙여서, 막대만 보고도 loss/정확도 추이를 바로 보게 한다
        pbar.set_postfix(loss=f"{loss.item():.4f}", val_acc=f"{val_acc:.3f}", best=f"{best_val:.3f}")
        if epoch % 10 == 0 or wait == 0:
            # 10 epoch마다 또는 개선이 있었을 때만 이력을 한 줄 남긴다 (매 epoch 남기면 로그가 너무 길어짐)
            pbar.write(f"epoch {epoch:3d}  loss {loss.item():.4f}  val_acc {val_acc:.3f}  best {best_val:.3f}")
        if wait >= args.patience:
            pbar.write(f"early stopping (epoch {epoch}, {args.patience} epoch 동안 개선 없음)")
            break

    # 학습이 끝난 시점(마지막 epoch)의 가중치가 아니라, val 정확도가 가장 좋았던 시점의
    # 가중치(best_state)로 되돌려서 최종 평가·저장한다 (과적합된 뒷부분 epoch 회피).
    model.load_state_dict(best_state)
    test_acc, pred = evaluate(model, X_test, y_test, device)
    print(f"\n=== 최종 평가 (test) ===\n정확도: {test_acc:.3f}")
    print("혼동 행렬 (행=실제, 열=예측):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=labels, zero_division=0))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    # 가중치만 저장하지 않고 config(모델 구조 재현용)와 labels(클래스 순서)를 함께 묶어
    # 저장한다 — predict.py가 코드에 아무 것도 하드코딩하지 않고 이 파일 하나로 모델
    # 구조를 복원하고 라벨을 얻을 수 있게 하기 위함 (design.md의 "클래스 하드코딩 금지" 규칙).
    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "config": {
                "input_dim": X.shape[2],
                "hidden_dim": args.hidden,
                "num_layers": args.layers,
                "frames": X.shape[1],
            },
            "test_accuracy": test_acc,
        },
        args.model_out,
    )
    print(f"모델 저장: {args.model_out}  (라벨: {labels})")


if __name__ == "__main__":
    main()
