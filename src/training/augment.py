# Last updated: 2026-07-27
'''
augment.py

학습용 특징 수준 도메인 랜덤화.

악조건(저조도, 센서 노이즈, 연기·가림, 렌즈 왜곡, 위치·속도 변화)이 랜드마크
좌표에 남기는 효과를 학습 배치에 무작위 주입한다. 픽셀이 아니라 좌표를
변형하므로 계산이 싸고, 원본 데이터는 건드리지 않는다 (실시간 적용).

주의: 감지 안 된 지점(0 벡터)은 "없음"이라는 의미이므로 어떤 변형도
0을 깨뜨리지 않는다 — 존재하는 지점만 변형한다.

시퀀스 형식: (T, 150) = (T, 50포인트 × xyz). 앞 42포인트 = 양손, 뒤 8 = 상체 포즈
(extractor.py의 150차원 벡터를 "관절 하나당 xyz 3개"의 50개 포인트로 다시 묶어서 본 것).

'''

import numpy as np # 배치/시퀀스 배열 연산, 난수 생성(rng)에 사용

HAND_POINTS = 42  # 21관절 × 2손 — 시퀀스에서 앞쪽 42포인트(126차원)가 손
NUM_POINTS = 50   # 손 42 + 상체 포즈 8 — reshape(T, 50, 3)의 기준

# 변형별로 "이번 시퀀스에 적용할지 말지"를 결정하는 확률 (0.5 = 절반의 시퀀스에만 적용).
# 매번 다 적용하면 원본 패턴을 못 배우고 증강 자체가 노이즈가 되므로, 일부러 확률을 둬서
# "가끔 악조건, 대부분은 정상"인 실전 분포에 가깝게 섞는다 (경험적 기본값 — 과하면 학습이 흔들린다).
P_TIME_WARP = 0.5
P_TRANSLATE = 0.5
P_SCALE = 0.5
P_DISTORT = 0.3
P_HAND_DROPOUT = 0.3
P_NOISE = 0.8

# 적용될 때 그 변형이 실제로 얼마나 강하게 들어가는지(강도) 결정하는 값들
NOISE_STD = 0.01          # 좌표 노이즈 표준편차 (센서 노이즈·저조도 떨림)
TRANSLATE_MAX = 0.06      # 화면 대비 이동량 최대치 (작업자 위치 변화)
SCALE_RANGE = (0.9, 1.1)  # 확대/축소 배율 범위 (카메라 거리 변화)
WARP_RANGE = (0.8, 1.25)  # 시간 신축 배율 범위 — 1보다 작으면 빠르게, 크면 느리게 재생한 효과 (동작 속도 변화)
DISTORT_K = 0.15          # 배럴/핀쿠션 왜곡 계수 최대치 (렌즈 왜곡)
DROPOUT_FRAMES = (1, 6)   # 손을 지울 프레임 개수 범위 [최소, 최대) (연기·가림·감지 실패)


def _time_warp(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''재생 속도를 바꿨다가 원래 길이(T)로 다시 리샘플링한다.
    
    예를 들어 factor=1.25면 "1.25배 빠르게 재생한 동작"의 프레임 위치를 원래 T개
    타임스텝에 맞춰 다시 뽑아내는 것과 같다. 실제로 프레임을 늘리거나 줄이지 않고,
    "몇 번째 원본 프레임에서 값을 가져올지"를 소수점 위치(src)로 계산한 뒤, 그 사이의
    두 정수 프레임(lo, hi)을 선형 보간(가중치 w)으로 섞어서 매끄러운 값을 만든다.
    '''
    t = len(seq)
    factor = rng.uniform(*WARP_RANGE)
    # src: 0 ~ (t-1)*factor 구간을 t개로 균등 분할한 뒤, 시퀀스 길이를 벗어나지 않게 clip
    src = np.clip(np.linspace(0, (t - 1) * factor, t), 0, t - 1)
    lo = np.floor(src).astype(int)          # src 바로 아래 정수 프레임 인덱스
    hi = np.minimum(lo + 1, t - 1)           # 그 다음 프레임 (마지막 프레임을 넘지 않게 clamp)
    w = (src - lo)[:, None]                  # lo와 hi 사이에서 hi 쪽으로 얼마나 치우쳤는지(0~1)
    return (1 - w) * seq[lo] + w * seq[hi]   # lo, hi 프레임을 w 비율로 섞은 선형 보간 결과


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''시퀀스 1개 (T, 150)에 무작위 악조건 변형을 순서대로 적용한다.

    각 변형은 독립적으로 "적용할지(P_*)"를 굴리고, 적용되면 존재하는(present) 지점만
    바꾼다 — 미검출(0)인 지점은 여전히 "감지 안 됨"으로 남아야 하기 때문이다.
    '''
    seq = seq.copy()  # 원본 배치(X_train)는 절대 건드리지 않도록 사본에만 변형

    # 시간 신축은 프레임 배치 자체를 다시 뽑는 연산이라, 아래의 포인트 단위 변형들보다
    # 먼저 적용해야 한다 (나중에 하면 방금 만든 노이즈/이동 등을 다시 보간해 흐려짐).
    if rng.random() < P_TIME_WARP:
        seq = _time_warp(seq, rng)

    pts = seq.reshape(len(seq), NUM_POINTS, 3)  # (T, 150) → (T, 50관절, xyz) — 관절 단위로 다루기 위한 재구성
    present = (pts != 0).any(axis=2)  # (T, 50) 불리언 — 이 관절이 이 프레임에 검출됐는지 (xyz 중 하나라도 0이 아니면 True)

    # 이동 (작업자 위치 변화): 모든 프레임·관절에 동일한 x/y 오프셋을 더함 (통째로 화면 안에서 옮겨진 것처럼)
    if rng.random() < P_TRANSLATE:
        offset = rng.uniform(-TRANSLATE_MAX, TRANSLATE_MAX, size=2)
        pts[..., 0][present] += offset[0]
        pts[..., 1][present] += offset[1]

    # 스케일 (카메라 거리 변화) — 원점(어깨 중심, 0)을 기준점으로 삼아 확대/축소.
    # extractor.py가 좌표를 화면 절대좌표가 아니라 어깨 중심 기준 상대 좌표로 정규화하므로
    # (doc/data_collection_log.md 2026-07-29 #1), 기준점도 화면 중앙(0.5)이 아니라 0이어야
    # 한다. x/y/z 전부 이미 같은 원점 기준이라 축 구분 없이 배율만 곱하면 된다.
    if rng.random() < P_SCALE:
        s = rng.uniform(*SCALE_RANGE)
        pts[present] *= s

    # 렌즈 왜곡 (배럴/핀쿠션): r' = r(1 + k·r²), 어깨 중심 기준 극좌표 반경(r)을 왜곡시키는 공식.
    # k>0이면 바깥으로 갈수록 더 밀려나는 배럴 왜곡, k<0이면 반대(핀쿠션)를 흉내낸다.
    # 스케일 증강과 같은 이유로 기준점은 화면 중앙(0.5)이 아니라 원점(어깨 중심, 0)이다.
    # r² 상한(0.5)은 예전 화면 절대좌표(0~1) 기준으로 정한 값이라 새 좌표 스케일(어깨너비 단위)
    # 에서도 적절한지는 미검증 — 필요하면 나중에 재튜닝 (doc/data_collection_log.md 참고).
    if rng.random() < P_DISTORT:
        k = rng.uniform(-DISTORT_K, DISTORT_K)
        dx = pts[..., 0]
        dy = pts[..., 1]
        factor = 1.0 + k * np.minimum(dx * dx + dy * dy, 0.5)
        pts[..., 0][present] = (dx * factor)[present]
        pts[..., 1][present] = (dy * factor)[present]

    # 손 소실 (연기·가림·순간 감지 실패): 무작위로 고른 몇 개 프레임에서 손 블록(앞 42포인트)만
    # 통째로 0으로 지운다 — 실제로 MediaPipe가 그 프레임에서 손을 놓쳤을 때와 같은 모양이 되게.
    if rng.random() < P_HAND_DROPOUT:
        n = rng.integers(*DROPOUT_FRAMES)  # 지울 프레임 개수
        frames = rng.choice(len(pts), size=min(n, len(pts)), replace=False)  # 중복 없이 무작위 프레임 선택
        pts[frames, :HAND_POINTS, :] = 0.0
        present[frames, :HAND_POINTS] = False  # 지운 지점은 이후 노이즈 등 다른 변형 대상에서도 제외

    # 좌표 노이즈 (센서 노이즈·저조도 떨림): 존재하는 지점에만 아주 작은 가우시안 노이즈를 더함
    if rng.random() < P_NOISE:
        noise = rng.normal(0.0, NOISE_STD, size=pts.shape).astype(np.float32)
        pts[present] += noise[present]

    return pts.reshape(len(seq), -1).astype(np.float32)  # (T, 50, 3) → 다시 (T, 150)으로 펼쳐서 반환


def augment_batch(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''배치 (B, T, 150) 전체에 시퀀스별로 각각 독립적인 무작위 변형을 적용한다.

    시퀀스마다 augment_sequence를 따로 호출하므로, 같은 배치 안에서도 어떤 시퀀스는
    노이즈만 타고 어떤 시퀀스는 손이 사라지는 등 서로 다른 조합의 악조건을 겪는다.
    '''
    return np.stack([augment_sequence(s, rng) for s in batch])
