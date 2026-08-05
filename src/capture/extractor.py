# Last updated: 2026-08-06

'''
    공용 특징 추출기: 양손 랜드마크 + 상체 포즈 → 고정 크기 벡터.
    수집(collect)·추론(predict)이 모두 이 모듈을 사용해
    학습-추론 특징 형식을 일치시킨다. 손 모델 다운로드/랜드마커 생성/그리기/카메라 열기 같은
    공용 헬퍼도 여기 함께 두어, landmark_viewer.py(검증용 뷰어)와 dataset/collect.py(수집 UI)가
    이 모듈에서 가져다 쓴다 (다른 스크립트들이 core를 참조하는 방향 — 반대로 core가 그것들에
    의존하지 않는다).

    특징 벡터 (FEATURE_DIM = 150):
        [왼손 21관절 × xyz = 63] [오른손 63] [상체 8관절 × xyz = 24]
        상체 관절: 어깨(11,12), 팔꿈치(13,14), 손목(15,16), 골반(23,24)
        — 골반 포함으로 어깨-골반 상대 관계(상반신 회전)를 표현할 수 있다.
        감지 안 된 손/포즈는 0으로 채움.

        좌표는 화면(카메라 프레임) 기준 절대 좌표가 아니라, 어깨 중심(랜드마크 11·12 중점)을
        원점으로 한 상대 좌표를 어깨너비로 나눠 스케일까지 정규화한 값이다 — 그렇지 않으면
        데이터가 적을 때 모델이 "동작의 모양"이 아니라 "화면 어디서 시작했는지"라는 훨씬 쉬운
        지름길을 학습해버린다 (예: 항상 화면 오른쪽에서 시작하면 동작과 무관하게 매번 같은
        라벨로 분류됨). 포즈가 감지되지 않아 원점을 잡을 수 없으면 정규화를 건너뛰고 원본
        좌표를 그대로 쓴다 (드문 경우이며, 없는 것보다는 낫다).

    목장갑 등으로 손 랜드마크가 불안정한 환경에서도 포즈(팔 궤적)는 몸 전체
    스케일로 추정되어 상대적으로 강인하다 (doc/design.md 참고).

    학습/예측 알고리즘: 여기서 만든 (30, 150) 시퀀스는 sklearn의 KNN 같은 단순 분류기가 아니라
    PyTorch로 구현한 LSTM(2층, hidden 64) 시퀀스 분류 모델로 학습·추론한다 — 수신호는 정지 자세가
    아니라 시간에 따른 궤적(동적 제스처)이라 시계열을 다루는 RNN 계열을 선택했다
'''

import platform # OS(Windows/macOS/Linux) 판별 — 카메라 백엔드를 OS별로 분기하기 위함
import time # 손 이탈 경고 깜빡임 주기 계산
import urllib.request # 최초 실행 시 모델(.task) 파일을 구글 저장소에서 내려받기 위함
from pathlib import Path # OS 무관 경로 처리 (Windows/macOS 공용, design.md 규약)

import cv2 # OpenCV : 카메라 입력, 이미지 전처리/표시
import numpy as np # 랜드마크 좌표를 배열(특징 벡터)로 다루기 위한 수치 연산
import mediapipe as mp # 손/포즈 랜드마크 추출 프레임워크
from mediapipe.tasks import python as mp_tasks # Tasks API 공통 옵션(BaseOptions 등)
from mediapipe.tasks.python import vision # HandLandmarker, PoseLandmarker 등 비전 태스크

ROOT = Path(__file__).resolve().parents[2]

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker.task"

# mediapipe가 주는 연결 객체(.start/.end 속성)를 (start, end) 튜플로 미리 변환해 둔다.
# 손 연결선(HAND_CONNECTIONS)과 포즈 연결선(ARM_CONNECTIONS)이 같은 형태(튜플 리스트)가 되어
# draw_skeleton() 하나로 둘 다 그릴 수 있다.
HAND_CONNECTIONS = [
    (conn.start, conn.end) for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS
]

# 팔·몸통 그리기용 연결 (포즈 원본 관절 인덱스 기준 — 어깨11/12, 팔꿈치13/14, 손목15/16, 골반23/24)
ARM_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24)]

LANDMARKS_PER_HAND = 21
HAND_DIM = LANDMARKS_PER_HAND * 3  # 21관절 × xyz = 63
UPPER_BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]  # 어깨/팔꿈치/손목/골반
POSE_DIM = len(UPPER_BODY_JOINTS) * 3  # 8관절 × xyz = 24
FEATURE_DIM = HAND_DIM * 2 + POSE_DIM  # 왼손 63 + 오른손 63 + 포즈 24 = 150



def _ensure_model(path: Path, url: str, label: str) -> Path:
    '''모델 파일이 없으면 다운로드한다 (손/포즈 모델이 공유하는 로직).'''
    if not path.exists():  # 최초 1회만 다운로드, 이후 실행부터는 캐시된 파일을 그대로 사용
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{label} 모델 다운로드 중... → {path}")
        urllib.request.urlretrieve(url, path)
        print(f"{label} 모델 다운로드 완료")
    return path

def ensure_hand_model() -> Path:
    return _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "손")

def ensure_pose_model() -> Path:
    return _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL, "포즈")

def create_hand_landmarker(running_mode=vision.RunningMode.VIDEO, min_confidence: float = 0.5):
    '''손 랜드마커를 생성한다 (뷰어·특징 추출기가 동일한 옵션을 공유).

    min_confidence: 손 검출/존재/추적 신뢰도 임계값 (세 옵션에 동일하게 적용).
    낮출수록 장갑 낀 손처럼 윤곽이 흐릿한 경우도 더 잘 잡지만, 오검출·지터도 늘어난다.
    '''
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=running_mode,  # 연속 프레임 입력을 전제로 하는 모드 (타임스탬프 필요)
        num_hands=2,  # 양손 모두 추적
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)

def create_pose_landmarker(running_mode=vision.RunningMode.VIDEO):
    '''포즈 랜드마커를 생성한다.'''
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=running_mode,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_mp_image(bgr_frame):
    '''OpenCV BGR 프레임 → MediaPipe 입력 이미지로 변환한다.
    OpenCV는 색상 채널을 BGR 순서로 다루지만 MediaPipe는 RGB(SRGB)를 기대하므로,
    채널 순서를 뒤집은(BGR→RGB) 픽셀 배열을 mp.Image로 감싸서 넘겨야 한다.
    '''
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_skeleton(
    frame,
    points,
    connections,
    line_color,
    point_color,
    line_thickness: int = 2,
    point_radius: int = 4,
) -> None:

    '''이미 픽셀 좌표로 변환된 관절점(dict 또는 list)을 받아 뼈대 선 + 관절 점을 그린다.
    points: {관절 인덱스: (x_px, y_px)} 또는 인덱스 순서와 동일한 리스트.
    connections: (시작 인덱스, 끝 인덱스) 튜플 목록 — points를 참조해 선을 잇는다.
    '''
  
    for a, b in connections:
        cv2.line(frame, points[a], points[b], line_color, line_thickness)
    for pt in (points.values() if isinstance(points, dict) else points):
        cv2.circle(frame, pt, point_radius, point_color, -1)

class FeatureExtractor:
    '''수집,추론이 공통으로 쓰는 진입점.
    HandLandmarker + PoseLandmarker를 한 번에 들고 있다가, 프레임을 넣으면
    두 모델 검출 결과를 위 FEATURE_DIM 포맷의  150차원 특징 벡터로 변환해준다.
    VIDEO 모드
    '''
    def __init__(self, hand_confidence: float = 0.5) -> None:
        self.hands = create_hand_landmarker(min_confidence=hand_confidence)
        self.pose = create_pose_landmarker()

    def detect(self, bgr_frame, timestamp_ms: int):
        image = to_mp_image(bgr_frame)  #BGR 프레임에서 (손 결과, 포즈 결과)를 반환.
        return (
            self.hands.detect_for_video(image, timestamp_ms),
            self.pose.detect_for_video(image, timestamp_ms),
        )

    @staticmethod
    def _reference_frame(pose_lms) -> tuple[np.ndarray, float] | None:
       
        if pose_lms is None:
            return None
        l_sh = np.array([pose_lms[11].x, pose_lms[11].y, pose_lms[11].z], dtype=np.float32)  # 왼쪽 어깨
        r_sh = np.array([pose_lms[12].x, pose_lms[12].y, pose_lms[12].z], dtype=np.float32)  # 오른쪽 어깨
        origin = (l_sh + r_sh) / 2  # 두 어깨의 중점 = 몸통 중심선 위의 기준점
        # z(깊이)는 원근에 따라 안정성이 떨어지므로 스케일 계산에는 화면 평면(x, y)만 사용
        scale = float(np.linalg.norm(r_sh[:2] - l_sh[:2]))
        if scale < 1e-6:
            return None
        return origin, scale

    @staticmethod
    def vector(hand_result, pose_result) -> np.ndarray:
        # 검출 결과 → (150,) 벡터. 왼손 | 오른손 | 상체 포즈 순.
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        # 이번 프레임의 정규화 기준을 한 번만 계산해서 손·포즈 좌표 변환에 공통으로 사용한다
        # (손과 포즈가 서로 다른 기준으로 정규화되면 상대적 위치 관계가 깨지기 때문).
        pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None  # 인물 1명 기준, 첫 검출만 사용
        frame = FeatureExtractor._reference_frame(pose_lms)
        # 기준을 못 구했으면 origin=0, scale=1로 두어 사실상 "정규화 없음"(원본 좌표 그대로)과
        # 동일하게 동작시킨다 — 별도 분기 없이 아래 수식 하나로 정규화 여부를 모두 처리하기 위함.
        origin, scale = frame if frame is not None else (np.zeros(3, dtype=np.float32), 1.0)

        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            slot = 0 if handedness[0].category_name == "Left" else 1  # 왼손=0번 슬롯, 오른손=1번 슬롯
            # 관절 21개 × (x, y, z) → 어깨 중심 기준 상대 좌표로 변환(평행이동+스케일 정규화) 후
            # 63개짜리 1차원 배열로 펼침(flatten)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
            coords = (coords - origin) / scale
            vec[slot * HAND_DIM : (slot + 1) * HAND_DIM] = coords.flatten()
        if pose_lms is not None:
            # 상체 8관절만 골라 (x, y, z) 순으로 펼쳐 24개짜리 배열을 만듦.
            # 손과 동일한 origin/scale로 정규화해야 "어깨 대비 손목이 얼마나 벌어졌는지" 같은
            # 손-포즈 간 상대 관계가 유지된다.
            coords = np.array(
                [[pose_lms[j].x, pose_lms[j].y, pose_lms[j].z] for j in UPPER_BODY_JOINTS],
                dtype=np.float32,
            )
            coords = (coords - origin) / scale
            vec[HAND_DIM * 2 :] = coords.flatten()  # 벡터 뒤쪽 24차원 = 상체 포즈
        return vec

    def close(self) -> None:
        '''랜드마커가 들고 있는 네이티브 리소스 해제 (종료 시 반드시 호출).'''
        self.hands.close()
        self.pose.close()

# 포즈랜드마크
def draw_pose(frame, pose_result) -> None:
    '''학습/추론 로직과는 무관한 디버그·수집용 시각화/카메라 도우미.
    landmark_viewer.py(검증 뷰어)와 dataset/collect.py(수집 UI)가 공용으로 가져다 쓴다.
    MediaPipe 좌표는 0~1 정규화 값이므로, 가로/세로 픽셀 크기(w, h)를 곱해 화면 좌표로 환산한다.
    '''
    if not pose_result.pose_landmarks:  # 사람이 검출 안 되면 그릴 것이 없으므로 그대로 반환
        return
    lms = pose_result.pose_landmarks[0]
    h, w = frame.shape[:2]
    pts = {j: (int(lms[j].x * w), int(lms[j].y * h)) for j in UPPER_BODY_JOINTS}
    draw_skeleton(frame, pts, ARM_CONNECTIONS, (255, 200, 0), (255, 100, 0), point_radius=5)

# 21개 손 관절을 순서(0~20)대로 HAND_CONNECTIONS의 인덱스와 맞추어 그린다.
def draw_landmarks(frame, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    points = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_landmarks)}
    draw_skeleton(frame, points, HAND_CONNECTIONS, (0, 255, 0), (0, 0, 255))


# 검출된 손(최대 2개) + 상체 포즈
def draw_detections(frame, hand_result, pose_result) -> None:
    for hand in hand_result.hand_landmarks:
        draw_landmarks(frame, hand)
    draw_pose(frame, pose_result)



def process_frame(cap: cv2.VideoCapture, extractor: "FeatureExtractor", t0: float):
    '''프레임 1장을 읽어 검출까지 수행 (수집·추론 공용). 
    반환: (frame, 손결과, 포즈결과) 또는 (None,)*3.
    t0는 이 캡처 세션의 시작 시각(time.monotonic() 기준) — VIDEO 모드가 요구하는 단조 증가
    타임스탬프를 만들기 위한 기준점이다.
    '''
    ok, frame = cap.read()
    if not ok:
        return None, None, None
    frame = cv2.flip(frame, 1)  # 셀피 뷰
    timestamp_ms = int((time.monotonic() - t0) * 1000)
    hand_result, pose_result = extractor.detect(frame, timestamp_ms)
    return frame, hand_result, pose_result


def _camera_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


# OS별 백엔드 웹캠
def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, _camera_backend())
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라 {index}번을 열 수 없습니다. "
            "다른 앱이 카메라를 사용 중인지, OS 카메라 권한이 허용되어 있는지 확인하세요."
        )
    return cap


def list_cameras(max_index: int = 5) -> list[int]:
    '''0부터 max_index-1까지 순서대로 열어봐서 실제로 연결된 카메라 인덱스만 반환한다.'''
    backend = _camera_backend()
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def choose_camera_index(preset: int | None = None) -> int:
    '''연결된 카메라가 여러 대면 번호를 골라 선택하게 한다. 한 대뿐이면 그걸 바로 쓴다.'''
    if preset is not None:
        return preset
    found = list_cameras()
    if not found:
        return 0  # 못 찾아도 0번으로 시도 (open_camera가 실패 시 에러 메시지로 안내)
    if len(found) == 1:
        return found[0]
    print("연결된 카메라가 여러 개 감지됐습니다:")
    for i in found:
        print(f"  {i}. 카메라 {i}")
    while True:
        choice = input("사용할 카메라 번호 입력: ").strip()
        if choice.isdigit() and int(choice) in found:
            return int(choice)
        print("목록에 없는 번호입니다.")


# 손 이탈 경고: 연속 미감지가 이 프레임 수를 넘으면 화면을 빨갛게 깜빡인다
WARN_AFTER_MISSES = 5

# ── 기준치 보정 ─────────────────────────────────────────────────────────────────
# 이 판정기의 유일한 침묵 경로는 "기준치가 나쁜 화면에서 잡히는 것"이다. 그러면 그
# 상태가 100% 로 기록되어 이후 아무것도 안 걸린다. 실제로 그 일이 났다 — 실행하자마자
# 화면이 뿌연데 "화질 정상"으로 떴다.
#
# 원인은 웹캠 예열이다. UVC 웹캠은 열린 직후 노출·화이트밸런스·초점을 맞추는 데 1~3초가
# 걸리는데, 보정 30프레임은 추론 주기(약 80ms) 기준 2.4초라 그 구간과 정확히 겹친다.
# 그래서 두 가지를 둔다:
#   1. 예열 프레임은 아예 버리고 그 뒤부터 표본을 모은다.
#   2. 표본의 중앙값이 아니라 상위 백분위를 기준치로 쓴다 — 흐린 프레임이 몇 장 섞여도
#      "이 카메라가 낼 수 있는 화질" 쪽으로 기준이 잡힌다. 중앙값은 그걸 그대로 깎는다.
CALIB_WARMUP_FRAMES = 20        # 이만큼은 버린다 (노출·초점이 자리 잡기를 기다림)
BLUR_CALIBRATION_FRAMES = 30    # 그 뒤 이 프레임 수로 "정상 화질" 기준치를 잡는다
BASELINE_PERCENTILE = 75        # 표본의 이 백분위를 기준치로 (중앙값 50 보다 위)
# 보정이 끝난 뒤에도 이 배수 이상으로 좋은 화질이 계속 관측되면, 기준치가 나쁜 화면에서
# 잡혔다는 뜻이다 — 자동으로 고치지는 않고(장면이 좋아졌을 뿐일 수도 있으므로) 경고만 한다.
BASELINE_SUSPECT_RATIO = 1.5
BASELINE_SUSPECT_FRAMES = 30

BLUR_HIGH_FREO_FRAC = 0.5       # 화면 반경 중 바깥쪽 몇 %를 "고주파(잔 디테일)"로 볼지

# 프레임을 이 비율로 줄인 뒤 다시 선명도를 잰다 = "거친 스케일의 장면 구조".
# 축소(INTER_AREA)가 영역 평균이라 물방울·먼지·센서 노이즈처럼 작은 것들을 지워 버리므로,
# 그 밑에 장면이 남아 있는지가 드러난다.
STRUCTURE_SCALE = 0.25

# ══ 이 값들은 "비상"을 혼자 결정하지 않는다 ═══════════════════════════════════════
#
# 아래 임계값들은 "화질에 이상이 있다"까지만 판정한다. 비상으로 올릴지는 호출부가
# **인식이 실제로 되고 있는지와 함께** 결정한다:
#
#     비상  =  화면 정지·끊김                      (인식 여부와 무관. 아래 참고)
#           또는 (화질 이상  그리고  관절이 안 잡힘)
#
# 왜 이렇게 갈랐는가 — 화질 지표만으로는 "인식이 깨지는 지점"을 맞힐 수 없기 때문이다:
#     · 라플라시안 분산은 렌즈가 아니라 **장면**을 잰다. 깨끗한 화면끼리도 장면만
#       바뀌면 233 ~ 1282 로 5.5배 흔들렸다.
#     · 흐림에 대한 붕괴가 너무 빠르다 — 눈으로 알기 어려운 k=3 흐림에서 이미 17.7%.
#       "선명도 25% 에서도 수신호 인식은 멀쩡" 이라는 실사용 관측과 합치면, 임계값을
#       어디에 두든 한쪽에서는 틀린다.
#     · 실제 저조도에서는 두 축이 **거꾸로 올라간다**. 웹캠이 게인을 올려 노이즈를
#       키우는데 라플라시안은 노이즈와 디테일을 구분하지 못한다(실측: 선명도 2457%).
# 그래서 "인식이 깨졌나"는 지표로 추정하지 않고 관절 검출 결과로 직접 본다. 지표는
# "그럴 만한 광학적 원인이 있나"만 답한다 — 사람이 그냥 자리를 비운 것을 고장으로
# 오인하지 않기 위한 조건이다.
#
# 그래서 임계값은 "명백히 무너진 수준"으로 잡는다. 애매한 저하는 어차피 인식이 되면
# 통과하므로 굳이 예민하게 잡을 이유가 없고, 사람이 없는 동안 오탐만 늘린다.
#
# 화면 정지만 예외인 이유: 프레임이 얼어붙으면 그 정지 화면 속 사람이 계속 검출되므로
# 인식 게이트를 걸면 영영 안 잡힌다. 이건 광학이 아니라 장치 문제라 단독으로 올린다.
LOW_LIGHT_P95 = 60              # 상위 5% 밝기(0~255)가 이 값 미만 = 저조도. 절대 기준
GLARE_LEVEL = 250               # 이 밝기 이상을 포화(빛반사)로 본다
GLARE_FRACTION = 0.15           # 화면의 이 비율 이상이 포화 = 빛반사
NOISE_RATIO_DROP = 0.25         # 구조/선명도 비가 기준 대비 이 밑 = 노이즈 지배(저조도 게인)
STRUCTURE_RATIO_DROP = 0.35     # 구조가 기준 대비 이 밑 = 가려짐·심한 흐림

# 선명도만은 인식 게이트를 거치지 않고 **단독으로** 비상을 올린다(치명 축).
# 실사용 관측 두 개가 그 경계를 좁혀 준다:
#     선명도 25% — 수신호 인식이 멀쩡히 됐다.        → 여기서 뜨면 오탐
#     선명도 21% — 이 정도부터는 잡히면 안 된다.      → 여기서는 떠야 함
# 그 사이에 둔다. 게이트를 안 거치는 이유: 이 화질에서 MediaPipe 가 관절을 계속
# 잡아내더라도 그 좌표를 믿을 수 없기 때문이다. "관절이 잡힌다 = 인식이 된다"는
# 게이트의 전제가 여기서 깨지므로, 게이트에 맡기면 영영 안 뜬다.
BLUR_RATIO_DROP = 0.22

# 지표 평활 계수 — **나빠질 때는 빠르게, 좋아질 때는 천천히**.
# 대칭 EMA(0.2)로 두면 암전 같은 급격한 열화에도 임계값까지 내려오는 데 7프레임이
# 걸렸다(실측: 80%→64%→51%→41%→33%→26%→21%). 추론 주기 80ms 기준 0.56초다.
# 비상 판정은 늦으면 의미가 없으므로 하강만 빠르게 따라가고, 복구 판정은 그대로
# 느리게 둔다 — 순간적으로 화면이 돌아왔다고 비상을 성급히 푸는 쪽이 더 위험하다.
QUALITY_EMA_FALL = 0.6          # 값이 나빠지는 방향으로 움직일 때
QUALITY_EMA_RISE = 0.2          # 좋아지는 방향으로 움직일 때

# ── 부분 가림 판정 (화면 일부만 죽는 경우) ───────────────────────────────────────
# 위 지표들은 전부 **화면 전체 평균**이라, 절반만 가려지면 값도 절반만 떨어진다.
# 실측 — 화면을 왼쪽부터 가려 보면 전체 구조 지표는:
#       25% 가림 → 63%,  50% 가림 → 50%,  75% 가림 → 40%
# 임계값(35%)에 75% 를 가려도 안 닿는다. 손으로 렌즈 절반을 덮어도 "정상"이 되는 이유다.
#
# 그래서 화면을 격자로 쪼개 **타일별로** 구조를 재고, 자기 기준치 대비 죽어 버린 타일이
# 몇 개인지를 본다. 같은 실측에서 죽은 타일은 4/16 → 8/16 → 12/16 으로 가린 면적을
# 그대로 따라간다. 장면이 통째로 바뀐 경우(로봇이 다른 구역으로 이동)는 4/16 이었으므로
# 그보다 위에 선을 그으면 주행 오탐 없이 절반 가림을 잡는다.
TILE_GRID = 4                   # 4x4 = 16 타일
DEAD_TILE_RATIO = 0.2           # 타일 구조가 자기 기준치의 이 비율 밑이면 "죽은 타일"
OCCLUSION_FRACTION = 0.35       # 죽은 타일이 이 비율 이상이면 부분 가림 (6/16 이상)

# ── 기준치가 필요 없는 절대 축 ──────────────────────────────────────────────────
# 상대 축들은 전부 "기동 시 기준치보다 나빠졌나"를 잰다. 그래서 처음부터 나쁜 상태로
# 시작하면 그게 100% 로 박혀 아무것도 안 걸린다. 아래 둘은 프레임 한 장의 절대값이라
# 기준치 없이도 판정되고, 보정 전에도 돈다.
#
# 값은 **같은 웹캠에서 받은 원본 640x480 프레임 두 장**으로 정했다 (HUD 스크린샷이 아니라
# 원본 — 창이 프레임을 세로로 눌러 그리기 때문에 스크린샷으로 재면 흐림이 더해져 값이
# 달라진다. 그걸 모르고 스크린샷으로 맞추다 계속 어긋났다):
#
#     지표        뿌연 화면(비상이어야)   정상 화면(통과해야)   판별력
#     선명도            8.8                27.9            3.2배
#     구조              8.5               106.4           12.5배   ← 가장 잘 가름
#     잡음비            0.97                3.82            3.9배
#     대비             25.7                25.0            없음
#     밝기 p95           97                 101            없음
#
# 구조는 1/4 로 줄인 뒤의 라플라시안 분산이라 "거친 스케일에 장면이 남아 있는가"를 잰다.
# 뿌연 화면은 8.5 — 사실상 아무것도 없다. 30 은 그 위, 정상(106)의 1/3.5 지점이다.
#
# 노이즈에도 흔들리지 않는다는 것이 이 축의 핵심이다. 두 프레임에 합성 노이즈를 얹어 보면
# 구조는 정상 106→126, 뿌연 8.5→13.5 로 순위가 그대로다(노이즈는 축소하면 사라지므로).
# 30 은 그 사이에 넉넉히 들어간다 — 정상의 1/3.5, 뿌연 최악값의 2.2배.
#
# **절대 축으로 쓰지 못한 것 두 가지도 기록해 둔다.** 둘 다 실측으로 탈락했다:
#   Crete 무참조 블러 — 뿌연 0.422 vs 정상 0.510 으로 **역전**. 미세 노이즈를 디테일로
#       읽어 뿌연 화면을 더 선명하다고 판정한다. 라플라시안이 속는 것과 같은 함정.
#   잡음비(구조/선명도) — 뿌연 0.97 vs 정상 3.82 로 잘 갈리는 듯했지만, 정상 프레임에
#       노이즈 σ=2 를 얹으면 1.02 가 되어 뿌연 쪽과 구분이 사라진다. 이건 화질이 아니라
#       노이즈 양을 재는 값이라, 기준치 대비(상대) 로만 의미가 있다.
STRUCTURE_ABS_MIN = 30.0        # 구조가 이 절대값 밑이면 장면이 없는 것 = 뭉개짐

# 게이트 경로(가려짐·노이즈)를 앞당기기 위한 연속 미검출 프레임 수.
# pose_ratio 는 30프레임 창의 비율이라 사람이 사라진 뒤 임계값(0.3) 밑으로 내려가는 데
# 21프레임(추론 80ms 기준 약 1.7초)이 걸린다. 비상 판정에는 너무 느리다. "방금 연속으로
# 몇 장 못 잡았나"는 같은 사실을 훨씬 빨리 말해 주므로, 그쪽을 게이트에 함께 쓴다.
# 6프레임 ≈ 0.5초 — 신호수가 잠깐 몸을 튼 정도로는 안 걸리고, 렌즈가 막히면 바로 걸린다.
NO_POSE_FRAMES = 6

# 화면 정지/끊김 판정 — 렌즈가 아니라 장치·드라이버 문제다.
FREEZE_DIFF = 0.05              # 프레임 간 평균 절대차가 이 값 미만이면 "같은 그림"
                                # (원본 해상도 기준. 살아 있는 웹캠은 정적인 장면에서도
                                #  센서 노이즈로 이 값을 훨씬 넘는다)
FREEZE_FRAMES = 15              # 그 상태가 이만큼 이어지면 정지로 판정 (20fps 기준 약 0.75초)
class HandWarning:
    '''학습 데이터 수집 중 손이 프레임 밖으로 나가면 그 구간은 0으로 채워진 저품질 샘플이 되므로,
    수집자가 바로 알아챌 수 있도록 손 이탈, 렌즈 이상 시 화면에 빨간 깜빡임 경고 등 강한 시각 경고를 준다.
    '''
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.misses = 0  # 연속 미검출 프레임 수

    def update(self, hand_result) -> None:
        # 이번 프레임에 손이 하나라도 잡히면 카운트 리셋, 아니면 누적
        self.misses = 0 if hand_result.hand_landmarks else self.misses + 1

    def draw(self, frame) -> None:
        if not self.enabled or self.misses < WARN_AFTER_MISSES:  # 임계치 미만이면 그리지 않음
            return
        h, w = frame.shape[:2]
        blink_on = int(time.monotonic() * 4) % 2 == 0
        if blink_on:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 16)  

        cv2.putText(frame, "!! HAND OUT OF FRAME !!", (w // 2 - 260, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        
        cv2.putText(frame, "!! HAND OUT OF FRAME !!", (w // 2 - 260, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)
def _track_down_fast(prev: float | None, new: float) -> float:
    '''비대칭 EMA — 값이 떨어질 때는 빠르게, 오를 때는 천천히 따라간다.

    선명도·구조는 낮을수록 나쁜 지표라 "떨어지는 방향"이 곧 고장 방향이다.
    그쪽만 빠르게 반영해 비상 판정을 앞당기고, 회복 방향은 느리게 둬서 한두 프레임
    좋아진 것으로 비상이 풀리지 않게 한다.
    '''
    if prev is None:
        return new
    alpha = QUALITY_EMA_FALL if new < prev else QUALITY_EMA_RISE
    return alpha * new + (1 - alpha) * prev


def _sharpness_score(gray: np.ndarray) -> float:
    '''그레이스케일 프레임의 선명도를 라플라시안(2차 미분) 분산으로 측정한다.
    라플라시안은 엣지(밝기가 급변하는 지점)에서 값이 크고, 평탄한 영역에서는 0에 가깝다.
    블러(디포커스·김서림)는 엣지를 뭉개므로 이 분산이 함께 떨어진다.
    FFT 비율과 달리 화면 전체 밝기(저주파) 성분이 계산에 섞여 들어가지 않아 선명/블러 구분력이 훨씬 높다.
    '''
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

class FrameLivenessMonitor:
    '''화면이 살아 있는지만 본다 — 렌즈가 아니라 장치·드라이버 쪽 고장.

    왜 따로 두는가: 프레임이 얼어붙으면 그 정지 화면 속 사람이 계속 검출되므로,
    "인식이 되고 있으니 정상"이라는 게이트를 통과해 버린다. 화질 지표도 멀쩡하다 —
    마지막 정상 프레임이 그대로 반복되는 것이기 때문이다. 그래서 이 판정만은 인식
    여부와 무관하게 단독으로 비상을 올린다.

    카메라 케이블이 헐거워지거나 USB 대역이 모자랄 때 드라이버가 마지막 버퍼를 계속
    돌려주는 일이 실제로 있다. 그때 로봇은 몇 초 전 수신호를 현재 명령으로 믿는다.
    '''

    def __init__(self) -> None:
        self._prev = None
        self._same_count = 0

    def update(self, frame: np.ndarray) -> bool:
        '''프레임 한 장을 받아 "정지 상태인가"를 돌려준다.

        **원본 해상도로 비교한다.** 축소해서 비교하면 안 된다 — 살아 있는 화면임을
        증명해 주는 것이 바로 화소 단위 센서 노이즈인데, 축소(영역 평균)가 그걸 지운다.
        1/4로 줄여 재봤더니 깨끗한 실시간 화면조차 평균 절대차 0.45 로 내려앉아
        정지로 오판했다.

        완전 일치(==)가 아니라 아주 작은 임계값을 쓰는 이유: 드라이버가 같은 버퍼를
        재전송할 때 JPEG 디코딩 경로에 따라 1비트씩 다를 수 있다. 반대로 임계값을
        키우면 진짜 정적인 장면을 정지로 오인하므로 최소한만 둔다.
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev is None:
            self._prev = gray
            return False
        diff = float(np.abs(gray - self._prev).mean())
        self._prev = gray
        self._same_count = self._same_count + 1 if diff < FREEZE_DIFF else 0
        return self._same_count >= FREEZE_FRAMES


def quality_status_text(lens: "LensHealthMonitor", frozen: bool) -> str:
    """HUD 상태줄용 한 줄. 화질 이상이 있으면 그쪽을, 없으면 정지 여부를 말한다.

    화질 이상을 먼저 보는 이유: "화면 정지"의 본래 모습은 **멀쩡해 보이는 화면이
    안 바뀌는 것**이다. 화면이 새하얗게 날아가서(빛반사) 프레임 간 차이가 0이 된
    경우까지 정지라고 부르면, 사람은 카메라 케이블을 확인하러 가게 된다 — 정작
    해야 할 일은 조명을 가리는 것인데.
    """
    if lens.fault_reason is None and frozen:
        return "화면 정지 — 프레임이 갱신되지 않음"
    return lens.status_text


class LensHealthMonitor:
    '''렌즈로 세상이 제대로 보이고 있는지를 매 프레임 진단한다.

    **이 클래스는 비상을 결정하지 않는다.** "화질에 이상이 있다"까지만 답하고, 그것을
    비상으로 올릴지는 호출부가 관절 검출 결과와 함께 정한다(위 상수 주석 참고).
    이유는 하나 — 화질 수치로 "인식이 깨지는 지점"을 맞히는 것이 불가능했기 때문이다.

    측정하는 원시 값은 넷:
        선명도  = 원본 프레임의 라플라시안 분산. 초점 흐림·해상도 저하에서 떨어진다.
        구조    = 1/4로 줄인 뒤의 라플라시안 분산. 축소가 작은 것(물방울·먼지·노이즈)을
                  평균으로 지우므로, **그 밑에 장면이 남아 있는지**가 드러난다.
        밝기    = 상위 5% 화소의 밝기. 잘 노출된 화소가 있기는 한지.
        포화    = 상위 밝기에 눌어붙은 화소의 비율. 빛반사가 여기서 잡힌다.

    이걸로 다섯 가지를 각각 다른 물리적 근거로 본다:
        해상도 변경 — 프레임 크기 자체가 바뀜. 정확하고 공짜인 검사라 제일 먼저 본다.
        저조도     — 밝기가 절대 기준 미만. 장면과 무관하므로 절대값으로 본다.
        빛반사     — 포화 화소 비율이 절대 기준 초과.
        노이즈     — 구조/선명도 비 붕괴. 노이즈는 축소하면 사라지고 디테일은 남으므로,
                     이 비가 무너지면 고주파의 정체가 디테일이 아니라 노이즈라는 뜻이다.
                     저조도에서 웹캠이 게인을 올릴 때가 정확히 이 경우다.
        가려짐/흐림 — 구조·선명도가 기준치 대비 붕괴.

    선명도·구조·노이즈비는 시작 후 BLUR_CALIBRATION_FRAMES 동안의 중앙값을 그 카메라·
    조명의 정상 기준치로 삼는다. 밝기·포화만 절대값이다 — 어둡거나 눌어붙은 건
    기준치가 무엇이든 그런 것이기 때문이다.

    한계 — 기준치는 반드시 렌즈가 깨끗하고 조명이 정상일 때 잡혀야 한다. 시작 순간
    이미 나쁘면 그 상태가 "정상"으로 기록되어 이후 아무것도 감지하지 못한다. 그래서
    보정이 끝나는 순간 기준치를 로그로 찍고, status_text 로 화면에도 계속 내보낸다 —
    사람이 알아챌 수 있는 자리를 두 곳 만들어 둔 것이다.
    '''
    def __init__(self)->None:
        self._baseline:float|None=None                  # 선명도 기준치
        self._structure_baseline: float | None = None   # 구조 기준치
        self._noise_baseline: float | None = None       # 구조/선명도 비 기준치
        self._samples : list[float] = []
        self._structure_samples: list[float] = []
        self._shape: tuple[int, int] | None = None      # 첫 프레임의 해상도
        self.ratio_ema : float | None = None
        self.structure_ema: float | None = None
        self.brightness: float | None = None            # 상위 5% 밝기(절대, 0~255)
        self.glare: float | None = None                 # 포화 화소 비율(절대, 0~1)
        self._tile_baseline: list[float] | None = None   # 타일별 구조 기준치
        self._tile_samples: list[list[float]] = []
        self.occlusion: float = 0.0                     # 죽은 타일 비율(0~1)
        self._warmup = 0                                # 버린 예열 프레임 수
        self._suspect = 0                               # 기준치가 의심스러운 연속 프레임 수
        self.fault_reason: str | None = None            # 직전 update() 가 올린 이상의 종류
        # 인식 게이트를 거치지 않고 단독으로 비상을 올려야 하는 이상인가.
        # 관절이 잡히더라도 그 좌표를 믿을 수 없는 수준일 때만 True 다.
        self.critical = False

    @staticmethod
    def _structure_score(gray: np.ndarray) -> float:
        '''1/4로 줄인 뒤의 선명도 = 거친 스케일에 남아 있는 장면 구조의 양.

        INTER_AREA 로 줄이는 것이 핵심이다 — 영역 평균이라 물방울·먼지·센서 노이즈처럼
        작은 것이 주변에 섞여 사라진다. NEAREST 로 줄이면 그것들이 그대로 살아남아
        이 축이 선명도와 같은 것을 재게 되고, 노이즈 판정도 같이 죽는다.
        '''
        small = cv2.resize(gray, None, fx=STRUCTURE_SCALE, fy=STRUCTURE_SCALE,
                           interpolation=cv2.INTER_AREA)
        return _sharpness_score(small)

    @staticmethod
    def _tile_structures(gray: np.ndarray) -> list[float]:
        '''화면을 TILE_GRID x TILE_GRID 로 쪼개 타일마다 구조 점수를 낸다.

        전체 평균으로는 부분 가림이 안 잡히기 때문이다 — 절반을 가려도 평균은 절반만
        떨어져 임계값에 안 닿는다(상수 주석의 실측 참고). 타일별로 보면 가려진 쪽
        타일들이 통째로 죽으므로, 죽은 타일 개수가 곧 가려진 면적이 된다.
        '''
        h, w = gray.shape
        n = TILE_GRID
        return [LensHealthMonitor._structure_score(
            gray[j * h // n:(j + 1) * h // n, i * w // n:(i + 1) * w // n])
            for j in range(n) for i in range(n)]

    @staticmethod
    def _brightness_score(gray: np.ndarray) -> float:
        '''상위 5% 화소의 밝기 = "이 프레임에 잘 노출된 부분이 있기는 한가".

        평균이 아니라 상위 백분위인 이유: 어두운 배경에 사람만 조명을 받는 정상 상황을
        저조도로 오판하지 않기 위해서다. 그런 화면은 평균은 낮아도 상위 5%는 높다.
        '''
        return float(np.percentile(gray, 95))

    @property
    def status_text(self) -> str:
        '''HUD 상태줄에 붙일 한 줄 진단 — 네 축의 현재 값.

        숫자를 화면에 내보내는 이유: 이상이 안 잡힐 때 판정기가 안 도는 건지,
        기준치가 이상하게 잡힌 건지, 진짜로 아직 멀쩡한 건지를 로그 없이 가려야 한다.
        640px 폭 상태줄에 앞의 안정화 상태와 함께 들어가야 해서 문구는 짧게 유지한다.
        '''
        if self.ratio_ema is None:
            return "렌즈 --"
        if self._baseline is None:
            # 보정 전에도 절대 축은 돌고 있다. 이상이 잡혔으면 그걸 먼저 말한다 —
            # 이 상태에서는 보정이 진행되지 않으므로(표본을 안 받는다) 사람이 원인을
            # 알아야 조치할 수 있다.
            if self.fault_reason is not None:
                return (f"화질 {self.fault_reason} — 보정 대기 "
                        f"(실{self.structure_ema:.0f}/{self._noise_ratio():.1f} "
                        f"밝{self.brightness:.0f})")
            if self._warmup < CALIB_WARMUP_FRAMES:
                # 예열 중에도 현재 값을 보여 준다 — 초점이 자리 잡는 게 눈으로 보이면,
                # 기준치가 언제 잡히는지 사람이 판단할 수 있다.
                return (f"렌즈 예열중 {self._warmup}/{CALIB_WARMUP_FRAMES} "
                        f"(실{self.structure_ema:.0f}/{self._noise_ratio():.1f})")
            return (f"렌즈 보정중 {len(self._samples)}/{BLUR_CALIBRATION_FRAMES} "
                    f"(실{self.structure_ema:.0f})")
        # 라벨을 한 글자로 줄인 이유: 640px 폭 상태줄에 앞의 안정화 상태까지 함께
        # 들어가야 하는데, "선명NN 구조NN ..." 로는 실제 화면에서 오른쪽이 잘렸다.
        # 선=선명도 구=구조 잡=잡음비 밝=상위5%밝기 반=포화비율 가=죽은 타일 비율(%)
        # 실=기준치 없이 보는 절대값 '구조/잡음비' (판정선 30 / 2.0).
        detail = (f"선{self.ratio_ema / self._baseline * 100:.0f} "
                  f"구{self.structure_ema / self._structure_baseline * 100:.0f} "
                  f"잡{self._noise_ratio() / self._noise_baseline * 100:.0f} "
                  f"밝{self.brightness:.0f} 반{self.glare * 100:.0f} "
                  f"가{self.occlusion * 100:.0f} "
                  f"실{self.structure_ema:.0f}/{self._noise_ratio():.1f}")
        if self.fault_reason is not None:
            return f"화질 {self.fault_reason} — {detail}"
        return f"화질 정상 ({detail})"

    def _absolute_fault(self) -> str | None:
        '''기준치를 쓰지 않는 축들만으로 이상 여부를 낸다. 원인 이름 또는 None.

        이 셋은 프레임 한 장만으로 판정되므로 보정이 끝나기를 기다릴 필요가 없다.
        그래서 두 곳에서 쓰인다 — 보정 전(그 구간을 무방비로 두지 않기 위해)과
        보정 후(다른 축들과 함께). 기준치가 나쁜 화면에서 잡히는 사고도 이 판정으로
        막는다: 여기서 이상이면 표본 자체를 받지 않는다.
        '''
        # 순서가 곧 원인/증상 구분이다. 캄캄하거나 새하얗게 날아간 화면은 엣지가 없어
        # 흐림 지표도 1.0 으로 치솟는데, 그때 "화면 흐림"이라고 하면 사람이 렌즈를
        # 닦으러 간다 — 정작 해야 할 일은 조명을 손보는 것이다. 그래서 노출을 먼저 본다.
        if self.brightness < LOW_LIGHT_P95:
            return "저조도"
        if self.glare > GLARE_FRACTION:
            return "빛반사"
        if self.structure_ema < STRUCTURE_ABS_MIN:
            return "화면 뭉개짐"
        return None

    def _noise_ratio(self) -> float:
        '''구조/선명도. 고주파의 정체가 디테일이면 크고, 노이즈면 작다.

        노이즈는 화소끼리 상관이 없어 축소하면 평균으로 사라지지만, 진짜 디테일은
        축소해도 남는다. 그래서 이 비가 무너졌다는 건 "고주파는 많은데 그게 장면이
        아니다" = 센서 노이즈라는 뜻이다. 저조도에서 웹캠이 게인을 올릴 때 정확히
        이렇게 된다 — 선명도는 되레 폭증하므로 선명도만 보면 절대 못 잡는다.
        '''
        return self.structure_ema / max(self.ratio_ema, 1e-6)

    def update(self,frame:np.ndarray) -> bool:
        '''프레임 한 장을 보고 "화질에 이상이 있나"를 돌려준다. 비상 여부가 아니다.

        원인은 fault_reason 에 남는다. 호출부는 이 반환값을 관절 검출 결과와 묶어서
        비상 여부를 정한다 — 인식이 되고 있으면 화질이 좀 나빠도 통과시켜야 하기 때문.
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = _sharpness_score(gray)
        structure = self._structure_score(gray)
        self.brightness = self._brightness_score(gray)
        self.glare = float(np.mean(gray >= GLARE_LEVEL))
        tile_now = self._tile_structures(gray)
        self.ratio_ema = _track_down_fast(self.ratio_ema, ratio)
        self.structure_ema = _track_down_fast(self.structure_ema, structure)

        # 해상도 변경은 지표가 아니라 사실이라 기준치도 EMA도 필요 없다. 카메라가 대역
        # 부족 등으로 해상도를 재협상하면 여기서 바로 잡힌다.
        if self._shape is None:
            self._shape = gray.shape
        elif gray.shape != self._shape:
            self.fault_reason = f"해상도 변경 {self._shape[1]}x{self._shape[0]}→" \
                                f"{gray.shape[1]}x{gray.shape[0]}"
            self.critical = True    # 추정이 아니라 사실이라 게이트를 거칠 이유가 없다
            return True

        # 기준치가 없어도 되는 축들(절대 흐림·저조도·빛반사)은 **보정 전에도** 판정한다.
        # 예전에는 예열+보정 50프레임(약 4초) 동안 판정 자체를 쉬었는데, 그러면 처음부터
        # 화질이 나쁜 경우가 정확히 그 구간에 걸려 안 잡혔다. 기준치가 필요 없는 축을
        # 기준치를 기다리느라 꺼 둘 이유가 없다.
        absolute = self._absolute_fault()

        if self._baseline is None:
            # 예열 구간은 표본에 넣지 않는다 — 여기서 모으면 초점이 안 맞은 화면이
            # "정상"으로 박혀 이후 판정이 통째로 죽는다(상수 주석 참고).
            if self._warmup < CALIB_WARMUP_FRAMES:
                self._warmup += 1
                self.fault_reason = absolute
                self.critical = absolute is not None
                return absolute is not None
            # **화질이 나쁜 프레임은 기준치 표본으로 받지 않는다.** 이게 기준치 오염을
            # 원천 차단한다 — 흐리거나 어두운 채로 시작하면 보정이 진행되지 않고, 그동안
            # 계속 이상으로 보고된다. 화면이 정상으로 돌아와야 비로소 표본이 모인다.
            if absolute is not None:
                self.fault_reason = absolute
                self.critical = True
                return True
            self._samples.append(self.ratio_ema)
            self._structure_samples.append(self.structure_ema)
            self._tile_samples.append(tile_now)
            if len(self._samples) >= BLUR_CALIBRATION_FRAMES:
                pct = BASELINE_PERCENTILE
                self._baseline = float(np.percentile(self._samples, pct))
                self._structure_baseline = float(np.percentile(self._structure_samples, pct))
                self._noise_baseline = self._structure_baseline / max(self._baseline, 1e-6)
                # 타일 기준치도 같은 백분위로. 원래 아무것도 없는 타일(민무늬 벽 등)은
                # 기준치 자체가 낮게 잡히므로 나중에 "죽었다"고 오판되지 않는다.
                self._tile_baseline = list(np.percentile(self._tile_samples, pct, axis=0))
                # 이 숫자들이 이후 모든 판정의 기준이다. 렌즈가 더럽거나 조명이 나쁜 채로
                # 시작하면 여기 나쁜 값이 박히고 판정기는 영영 침묵하므로, 반드시 남긴다.
                print(f"화질 기준치 확정: 선명도 {self._baseline:.0f}, "
                      f"구조 {self._structure_baseline:.0f}, "
                      f"잡음비 {self._noise_baseline:.2f}, 밝기 {self.brightness:.0f} "
                      f"(예열 {CALIB_WARMUP_FRAMES} 제외, "
                      f"{BLUR_CALIBRATION_FRAMES}프레임 상위 {pct}%)")
            self.fault_reason = None
            self.critical = False
            return False

        # 보정이 끝난 뒤에도 화질이 기준치보다 한참 좋게 유지되면, 기준치가 나쁜 화면에서
        # 잡혔다는 뜻이다(예열이 더 길었던 카메라). 자동으로 고치지는 않는다 — 장면이
        # 좋아진 것뿐일 수도 있어서다. 대신 한 번 알려 주어 사람이 재시작을 판단하게 한다.
        if self.ratio_ema > self._baseline * BASELINE_SUSPECT_RATIO:
            self._suspect += 1
            if self._suspect == BASELINE_SUSPECT_FRAMES:
                print(f"[경고] 화질이 기준치의 {self.ratio_ema / self._baseline * 100:.0f}% 로 "
                      "계속 관측됩니다 — 기준치가 예열/불량 구간에서 잡혔을 수 있습니다. "
                      "화면이 선명한 상태에서 노드를 다시 시작하세요.")
        else:
            self._suspect = 0

        # 죽은 타일 비율 — 자기 기준치 대비 구조가 사라진 타일이 몇 개인가.
        dead = sum(1 for now, base in zip(tile_now, self._tile_baseline)
                   if now < base * DEAD_TILE_RATIO)
        self.occlusion = dead / len(tile_now)

        # 각 축을 "자기 임계값 대비" 얼마나 내려갔는지로 환산한다. 1.0 미만이면 그 축이
        # 판정을 올린 것이다. 빛반사만 방향이 반대라(높을수록 나쁨) 역수로 맞춘다.
        #
        # 두 무리로 나눠 앞의 것을 먼저 보는 이유 — 여러 축은 거의 항상 같이 무너지는데
        # 그중 하나가 **원인**이고 나머지는 **증상**이기 때문이다. 조명이 어두워지면
        # 대비가 죽어 구조·선명도가 같이 떨어지는데(실측: 밝기 x0.15 에서 구조는 임계값의
        # 6%까지 내려가 "가려짐"이 이겼다), 사람이 해야 할 일은 렌즈를 닦는 게 아니라
        # 불을 켜는 것이다. 그래서 조명·노출 쪽을 먼저 보고, 거기가 멀쩡할 때만
        # 광학 쪽(가려짐·흐림)을 원인으로 부른다.
        causes = {
            # 절대 축(구조·잡음비의 원값)은 기준치를 안 쓰므로 기준치가 나쁘게 잡힌
            # 경우에도 유일하게 살아 있다. 그래서 상대 축들보다 먼저 본다.
            "화면 뭉개짐": self.structure_ema / STRUCTURE_ABS_MIN,
            "저조도": self.brightness / LOW_LIGHT_P95,
            "빛반사": GLARE_FRACTION / max(self.glare, 1e-6),
            "노이즈": self._noise_ratio() / (self._noise_baseline * NOISE_RATIO_DROP),
        }
        # 아래 둘은 원인 판정이 모두 통과했을 때만 원인 이름으로 쓰인다(표시용 순서).
        # 둘 다 critical 에도 들어가므로 게이트 없이 단독으로 비상을 올린다.
        symptoms = {
            f"부분 가림 {self.occlusion * 100:.0f}%": (
                OCCLUSION_FRACTION / max(self.occlusion, 1e-6)),
            "가려짐": self.structure_ema / (self._structure_baseline * STRUCTURE_RATIO_DROP),
        }
        # 게이트를 거치지 않고 단독으로 비상을 올리는 축들.
        #   저조도·빛반사 — 절대 기준이라 프레임 하나로 확정된다. 평활도 게이트도
        #       필요 없다: 화면이 캄캄하거나 새하얗게 날아갔으면 관절이 잡히든 말든
        #       그 좌표를 믿을 수 없다. 게이트에 묶어 뒀더니 암전을 1프레임째에
        #       알아보고도 pose_ratio(30프레임 창)를 기다리느라 21프레임 늦었다.
        #   노이즈 — 조명만 끄면 웹캠이 게인을 올려 밝기를 유지하므로 위 저조도에
        #       걸리지 않는다. 그 화면에서도 MediaPipe 는 관절을 계속 잡아내는데,
        #       게이트에 묶어 뒀더니 비상이 안 뜨고 그 좌표로 stop 까지 확정했다.
        #       구조/선명도 비가 무너지는 정상 장면은 없다 — 그 서명은 센서 노이즈뿐이다.
        #   선명도 — 관절이 계속 잡혀도 못 믿는 수준(상수 주석 참고).
        #   구조   — 같은 이유. 예전엔 "민무늬 벽에 바짝 붙으면 구조가 정상적으로도
        #       낮아진다"는 오탐 우려로 게이트에 뒀는데, 실사용에서 화면이 뭉개져
        #       확률이 27/6/24/34 로 전부 흩어진 상태(구조 24%)인데도 MediaPipe 가
        #       관절을 잡아내 게이트가 안 열렸다. 구조 35% 미만은 실측상 k=15 수준의
        #       심한 뭉개짐이라 "적당한 저하"가 아니다 — 그 화질에서 잡힌 관절 좌표는
        #       잡혔다는 사실과 무관하게 못 믿는다.
        #   부분 가림 — 화면의 3분의 1 이상이 죽었으면 남은 쪽에서 관절이 잡히더라도
        #       신호수 전신이 보인다는 보장이 없다. 전체 평균 축들이 원리상 못 잡는
        #       구간이라(75% 를 가려도 전체 구조는 40%) 이 축이 유일한 탐지 수단이다.
        # 결국 모든 축이 단독이다 — 게이트는 "화질이 애매할 때 사람이 없으면 비상"이라는
        # 마지막 안전망으로만 남는다.
        self.critical = (absolute is not None
                         or self.brightness < LOW_LIGHT_P95
                         or self.glare > GLARE_FRACTION
                         or self._noise_ratio() < self._noise_baseline * NOISE_RATIO_DROP
                         or self.ratio_ema < self._baseline * BLUR_RATIO_DROP
                         or self.occlusion >= OCCLUSION_FRACTION
                         or self.structure_ema
                         < self._structure_baseline * STRUCTURE_RATIO_DROP)
        for group in (causes, symptoms):
            worst, margin = min(group.items(), key=lambda kv: kv[1])
            if margin < 1.0:
                self.fault_reason = worst
                return True
        if self.critical:
            self.fault_reason = "흐림·해상도 저하"
            return True
        self.fault_reason = None
        return False