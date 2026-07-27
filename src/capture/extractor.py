# Last updated: 2026-07-27

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

def create_hand_landmarker(running_mode=vision.RunningMode.VIDEO):
    '''손 랜드마커를 생성한다 (뷰어·특징 추출기가 동일한 옵션을 공유).'''
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=running_mode,  # 연속 프레임 입력을 전제로 하는 모드 (타임스탬프 필요)
        num_hands=2,  # 양손 모두 추적
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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
    def __init__(self) -> None:
        self.hands = create_hand_landmarker()
        self.pose = create_pose_landmarker()

    def detect(self, bgr_frame, timestamp_ms: int):
        image = to_mp_image(bgr_frame)  #BGR 프레임에서 (손 결과, 포즈 결과)를 반환.
        return (
            self.hands.detect_for_video(image, timestamp_ms),
            self.pose.detect_for_video(image, timestamp_ms),
        )

    @staticmethod
    def vector(hand_result, pose_result) -> np.ndarray:
        # 검출 결과 → (150,) 벡터. 왼손 | 오른손 | 상체 포즈 순.
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            slot = 0 if handedness[0].category_name == "Left" else 1  # 왼손=0번 슬롯, 오른손=1번 슬롯
            # 관절 21개 × (x, y, z) → 63개짜리 1차원 배열로 펼침 (flatten)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32).flatten()
            vec[slot * HAND_DIM : (slot + 1) * HAND_DIM] = coords
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]  # 인물 1명 기준, 첫 번째 검출만 사용
            # 상체 8관절만 골라 (x, y, z) 순으로 펼쳐 24개짜리 배열을 만듦
            coords = np.array(
                [[pose_lms[j].x, pose_lms[j].y, pose_lms[j].z] for j in UPPER_BODY_JOINTS],
                dtype=np.float32,
            ).flatten()
            vec[HAND_DIM * 2 :] = coords  # 벡터 뒤쪽 24차원 = 상체 포즈
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


# OS별 백엔드 웹캠
def open_camera(index: int = 0) -> cv2.VideoCapture:
    system = platform.system()
    if system == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Darwin":  # macOS
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라 {index}번을 열 수 없습니다. "
            "다른 앱이 카메라를 사용 중인지, OS 카메라 권한이 허용되어 있는지 확인하세요."
        )
    return cap


# 손 이탈 경고: 연속 미감지가 이 프레임 수를 넘으면 화면을 빨갛게 깜빡인다
WARN_AFTER_MISSES = 5

BLUR_CALIBRATION_FRAMES = 30    # 시작 후 이 프레임 수 동안 "정상 화질" 기준치를 캘리브레이션(보간)
BLUR_RATIO_DROP = 0.5           # 기준치 대비 이 비율 밑으로 떨어지면 블러(김서림 등)로 판정
BLUR_HIGH_FREO_FRAC = 0.5       # 화면 반경 중 바깥쪽 몇 %를 "고주파(잔 디테일)"로 볼지
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
def _sharpness_score(gray: np.ndarray) -> float:
    '''그레이스케일 프레임의 선명도를 라플라시안(2차 미분) 분산으로 측정한다.
    라플라시안은 엣지(밝기가 급변하는 지점)에서 값이 크고, 평탄한 영역에서는 0에 가깝다.
    블러(디포커스·김서림)는 엣지를 뭉개므로 이 분산이 함께 떨어진다.
    FFT 비율과 달리 화면 전체 밝기(저주파) 성분이 계산에 섞여 들어가지 않아 선명/블러 구분력이 훨씬 높다.
    '''
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

class LensHealthMonitor:
    '''카메라 프레임의 고주파 성분 추이 (렌즈 블러 현상 진단)
    시작 후 BLUR_CALIBRAION_FRAMES 동안 관측된 값의 중앙값을 사용되는 카메라,조명의 정상 기준치로 잡는다. 
    이후 EMA로 평활한 비율이 기준치 대비 크게 떨어지면 블러 상태로 판정한다.
    '''
    def __init__(self)->None:
        self._baseline:float|None=None
        self._samples : list[float] = []
        self.ratio_ema : float | None = None

    def update(self,frame:np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = _sharpness_score(gray)
        self.ratio_ema = ratio if self.ratio_ema is None else 0.2* ratio + 0.8 *self.ratio_ema

        if self._baseline is None:
            self._samples.append(self.ratio_ema)
            if len(self._samples) >= BLUR_CALIBRATION_FRAMES:
                self._baseline = float(np.median(self._samples))
            return False 

        return self.ratio_ema < self._baseline * BLUR_RATIO_DROP