# Last updated: 2026-08-06
'''
publish.py
분류 결과 외부 전달 인터페이스.

이 저장소의 책임은 확정된 수신호를 외부(기계 시뮬레이터 프로젝트)로 내보내는
것까지다. ROS 2 노드 자체는 별도 프로젝트가 담당한다.

백엔드 두 가지:
    rosbridge — rosbridge 웹소켓 서버로 ROS 2 토픽(std_msgs/String) 발행.
                기계 프로젝트(Docker/Ubuntu)에서 rosbridge_server를 띄워두면
                macOS의 추론 프로세스가 ROS 없이도 토픽을 발행할 수 있다.
    udp       — JSON을 UDP 데이터그램으로 송신. 의존성·서버 불필요, 디버깅용.

메시지 형식 (JSON, ROS에서는 std_msgs/String.data 안에 담김):
    {"signal": "stop", "confidence": 0.93, "timestamp": 1783300000.0}
    signal은 모델 라벨 그대로, 인식 대기중 상태는 "unknown" (기계 쪽 안전 기본값 = 정지).
'''

import json # 메시지(signal/confidence/timestamp)를 문자열로 직렬화
import socket # UdpPublisher가 쓰는 UDP 소켓
import time # 메시지 timestamp, rosbridge 재접속 쿨다운 계산


class ConsolePublisher:
    """전송 없이 콘솔 출력만 (기본값, --publish 미지정 시).

    predict.py가 확정/하트비트 시점에 이미 print()로 로그를 남기므로, 여기서는
    별도 전송 없이도 동작 확인이 가능하다 — 서버 없이 로컬에서 추론만 검증할 때 쓴다.
    """

    def publish(self, signal: str, confidence: float) -> None:
        pass  # predict.py가 이미 콘솔에 출력하므로 아무것도 하지 않는다

    def close(self) -> None:
        pass  # 정리할 리소스가 없음


class UdpPublisher:
    '''JSON을 UDP로 송신한다 (의존성·서버 불필요 — 로컬 디버깅용 백엔드).'''

    def __init__(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 연결 없는 소켓 — 매 전송이 독립적
        print(f"[publisher] UDP → {host}:{port}")

    def publish(self, signal: str, confidence: float) -> None:
        payload = json.dumps(
            {"signal": signal, "confidence": round(confidence, 3), "timestamp": time.time()},
            ensure_ascii=False,  # 한글 라벨이 유니코드 이스케이프(\uXXXX) 대신 그대로 나가도록
        )
        self.sock.sendto(payload.encode("utf-8"), self.addr)

    def close(self) -> None:
        self.sock.close()


class RosbridgePublisher:
    '''rosbridge 웹소켓으로 ROS 2 토픽(std_msgs/String)을 발행한다.

    서버가 없거나 끊겨도 추론을 막지 않는다 — 전송 실패는 경고만 남기고
    5초 간격으로 재접속을 시도한다.
    '''

    RECONNECT_COOLDOWN = 5.0  # 연결 끊김/실패 후 재시도까지 대기하는 최소 간격(초) — 매 프레임 재시도해 추론 루프를 막지 않기 위함

    def __init__(self, url: str = "ws://localhost:9090", topic: str = "/hand_signal") -> None:
        self.url = url
        self.topic = topic
        self.ws = None  # 연결 안 됐거나 끊긴 상태 = None (publish()가 이 값으로 재접속 필요 여부 판단)
        self._last_attempt = 0.0  # 마지막 연결 시도 시각 (RECONNECT_COOLDOWN 계산 기준)
        self._connect()  # 생성 시점에 1회 시도 — 서버가 없어도 예외를 던지지 않고 재시도 상태로 남는다

    def _connect(self) -> None:
        import websocket  # pip 패키지명은 websocket-client, import 이름은 websocket (pyproject.toml에 이미 고정됨)

        self._last_attempt = time.monotonic()
        try:
            self.ws = websocket.create_connection(self.url, timeout=2)
            # rosbridge 프로토콜: 메시지를 보내기 전에 토픽을 advertise해야 구독자가 타입을 안다
            self.ws.send(json.dumps({
                "op": "advertise",
                "topic": self.topic,
                "type": "std_msgs/String",
            }))
            print(f"[publisher] rosbridge 연결됨 → {self.url}  토픽 {self.topic}")
        except Exception as e:
            self.ws = None  # 연결 실패해도 프로세스를 죽이지 않고 다음 publish() 호출에서 재시도
            print(f"[publisher] rosbridge 연결 실패 ({e}) — {self.RECONNECT_COOLDOWN}초 후 재시도")

    def publish(self, signal: str, confidence: float) -> None:
        if self.ws is None:
            # 매 프레임 재접속을 시도하면 서버가 죽어있는 동안 추론 루프가 계속 지연되므로
            # 쿨다운이 지났을 때만 재시도한다
            if time.monotonic() - self._last_attempt >= self.RECONNECT_COOLDOWN:
                self._connect()
            if self.ws is None:  # 재시도했는데도 여전히 실패 → 이번 프레임은 그냥 건너뜀
                return
        data = json.dumps(
            {"signal": signal, "confidence": round(confidence, 3), "timestamp": time.time()},
            ensure_ascii=False,
        )
        try:
            # std_msgs/String 메시지는 data 필드 하나뿐이라, 위에서 만든 JSON 문자열을
            # 그 안에 한 번 더 감싸 넣는다 (그래서 json.dumps를 두 번 호출)
            self.ws.send(json.dumps({
                "op": "publish",
                "topic": self.topic,
                "msg": {"data": data},
            }))
        except Exception as e:
            print(f"[publisher] rosbridge 전송 실패 ({e}) — 재접속 예정")
            self.ws = None  # 다음 publish() 호출에서 쿨다운 이후 재연결 시도

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.send(json.dumps({"op": "unadvertise", "topic": self.topic}))
                self.ws.close()
            except Exception:
                pass  # 종료 중 전송 실패는 무시 — 이미 끊기는 중이므로 재시도할 필요 없음
            self.ws = None


def make_publisher(kind: str, *, udp_host: str, udp_port: int,
                   rosbridge_url: str, ros_topic: str):
    """predict.py의 --publish 옵션에 따라 백엔드를 골라 생성하는 팩토리.

    kind: "udp" | "rosbridge" | 그 외(기본값) → ConsolePublisher.
    세 클래스 모두 publish(signal, confidence) / close() 인터페이스가 동일해서,
    predict.py는 어떤 백엔드가 반환됐는지 신경 쓰지 않고 그대로 호출한다.
    """
    if kind == "udp":
        return UdpPublisher(udp_host, udp_port)
    if kind == "rosbridge":
        return RosbridgePublisher(rosbridge_url, ros_topic)
    return ConsolePublisher()
