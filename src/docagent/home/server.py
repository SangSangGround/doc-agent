"""루프백 전용 문서 보조 UI. 원본을 저장하지 않으며 OCR은 임시 파일을 쓸 수 있다."""
from __future__ import annotations

import base64
import binascii
import io
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from docagent.home.capture import CaptureCoach
from docagent.home.privacy import propose_redactions, render_redaction_preview
from docagent.home.session import HomeSession
from docagent.home.templates import HomeTemplate
from docagent.vision.checkbox_review import verify_checkbox_groups

MAX_BODY = 12 * 1024 * 1024
MAX_PIXELS = 16_000_000
ASSETS = Path(__file__).with_name("assets")


def decode_image(value: Any) -> np.ndarray:
    """압축 폭탄과 잘못된 입력을 디코딩 전에 제한한다."""
    if not isinstance(value, str) or len(value) > MAX_BODY:
        raise ValueError("이미지가 너무 크거나 올바르지 않습니다.")
    if value.startswith("data:image/"):
        value = value.split(",", 1)[-1]
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as candidate:
            if candidate.format not in ("JPEG", "PNG") or candidate.width*candidate.height > MAX_PIXELS:
                raise ValueError("PNG/JPEG 이미지와 1,600만 화소 이하를 사용하세요.")
            candidate.verify()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except (OSError, binascii.Error, Image.DecompressionBombError) as exc:
        raise ValueError("이미지를 읽을 수 없습니다.") from exc
    if image is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return image


def encode_image(image: np.ndarray) -> str:
    success, data = cv2.imencode(".png", image)
    if not success:
        raise ValueError("미리보기를 만들지 못했습니다.")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class HomeApplication:
    """단일 로컬 사용자용 세션. 저장은 사용자 지정 세션 경로에 한정한다."""
    def __init__(self, template: HomeTemplate, *, session_path: Path | None = None):
        self.template = template
        self.session_path = session_path
        self.session = (HomeSession.load(session_path, template)
                        if session_path and session_path.exists() else HomeSession(template))
        self.coach = CaptureCoach()
        self.capture: np.ndarray | None = None
        self.observation: dict | None = None
        self.document_ready = False
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()

    def state(self) -> dict:
        return {"session":self.session.view(), "document_ready":self.document_ready,
                "has_capture":self.capture is not None, "purpose":self.template.manifest["purpose"],
                "can_verify": self.observation is not None,
                "title":self.template.manifest["title"], "external_transfer_enabled":False,
                "session_saved":self.session_path is not None}

    def persist(self) -> None:
        if self.session_path:
            self.session.save(self.session_path)

    def handle(self, action: str, data: dict) -> dict:
        if action == "frame":
            return self.coach.update(decode_image(data.get("image")))
        if action == "capture":
            # 재촬영이 실패해도 이전 성공 증거가 남지 않는다.
            self.session.invalidate_verification()
            self.document_ready = False
            self.observation = None
            self.capture = None
            self.persist()
            image = decode_image(data.get("image"))
            mode = data.get("mode", "photo")
            if mode not in ("photo", "scan"):
                raise ValueError("촬영 형식이 올바르지 않습니다.")
            result = verify_checkbox_groups(self.template.reference, image, self.template.manifest["groups"],
                        page_size_mm=tuple(self.template.manifest["page_size_mm"]), capture_mode=mode)
            self.capture, self.observation = image, result
            # 문서 정렬·인쇄 내용 확인과 체크 흔적 판독은 독립적인 결과다.
            # 희미한 체크 때문에 검토된 원문 읽기까지 차단하지 않는다.
            self.document_ready = result.get("alignment", {}).get("reason") == "registered"
            self.persist()
            if not self.document_ready:
                instruction = "문서를 정확히 확인하기 어렵습니다. 전체 문서가 선명하게 나오도록 다시 촬영해주세요."
            elif result.get("status") == "uncertain":
                instruction = "지원 양식을 확인해 원문을 읽을 수 있습니다. 일부 체크 표시는 판독하기 어려워 완료 여부를 확정할 수 없습니다. 표시를 확인하거나 다시 촬영해주세요."
            else:
                instruction = "지원 양식과 사진의 정렬을 확인했습니다. 원문을 읽거나 체크 결과를 확인할 수 있습니다."
            return {**self.state(), "observation":result,
                    "instruction":instruction}
        if action == "privacy":
            if self.capture is None:
                raise ValueError("먼저 문서를 촬영하거나 불러오세요.")
            proposal = propose_redactions(self.capture)
            result = {"proposal":proposal, "instruction":"마스킹 후보입니다. 자동 검출이 모든 개인정보를 찾았다는 뜻은 아닙니다. 외부로 전송하지 않습니다."}
            if proposal["status"] == "draft":
                result["preview"] = encode_image(render_redaction_preview(self.capture, proposal))
            else:
                result["instruction"] = "로컬 OCR로 개인정보 후보를 확인하지 못했습니다. OCR 설치와 촬영 상태를 확인해주세요."
            return result
        if action == "reset_capture":
            self.coach.reset()
            self.capture = self.observation = None
            self.document_ready = False
            self.session.invalidate_verification()
            self.persist()
            return self.state()
        if action == "read":
            if not self.document_ready:
                raise ValueError("먼저 촬영한 문서가 지원 양식인지 확인해야 합니다.")
            scope = data.get("scope", "current")
            if scope not in ("page", "current"):
                raise ValueError("읽기 범위가 올바르지 않습니다.")
            text = self.session.read_page() if scope == "page" else self.session.read_current()
            return {"instruction":text, **self.state()}
        if action == "preferences":
            self.session.set_speech_rate(data.get("speech_rate"))
        elif action == "move":
            self.session.move(data.get("direction", "replay"))
        elif action == "select":
            if not self.document_ready:
                raise ValueError("먼저 촬영한 문서를 확인하세요.")
            self.session.propose_selection(data.get("group_id"), data.get("option_ids"))
            self.observation = None
        elif action == "confirm":
            self.session.confirm_selection(data.get("group_id"))
            self.observation = None
        elif action == "cancel":
            self.session.cancel_selection(data.get("group_id"))
            self.observation = None
        elif action == "verify":
            if self.observation is None:
                raise ValueError("작성 후 문서를 촬영하거나 불러오세요.")
            self.session.apply_verification(self.observation)
        else:
            raise ValueError("지원하지 않는 명령입니다.")
        self.persist()
        return self.state()


def make_server(app: HomeApplication, port: int = 8765) -> ThreadingHTTPServer:
    """원격 노출·교차 출처 요청 없이 루프백에서만 제공한다."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass  # 원문·선택·접속 쿼리를 로그로 남기지 않는다.

        def send_body(self, status: int, body: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(body)

        def reply(self, status: int, data: dict) -> None:
            self.send_body(status,json.dumps(data,ensure_ascii=False,allow_nan=False).encode(),"application/json; charset=utf-8")

        def same_host(self) -> bool:
            expected = {f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in expected

        def authorized(self) -> bool:
            origin = self.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{self.server.server_port}",f"http://localhost:{self.server.server_port}"}
            token = self.headers.get("X-Session-Token", "")
            return (self.same_host() and (origin is None or origin in allowed)
                    and isinstance(token, str) and token.isascii()
                    and secrets.compare_digest(token, app.token))

        def do_GET(self) -> None:
            if not self.same_host():
                self.reply(403,{"error":"로컬 주소로 접속하세요."}); return
            if self.path == "/":
                body=(ASSETS/"index.html").read_text(encoding="utf-8").replace("__SESSION_TOKEN__",app.token)
                self.send_body(200,body.encode(),"text/html; charset=utf-8")
            elif self.path in ("/app.js","/style.css"):
                mime = "application/javascript" if self.path.endswith("js") else "text/css"
                self.send_body(200,(ASSETS/self.path[1:]).read_bytes(),mime+"; charset=utf-8")
            elif self.path == "/api/state" and self.authorized():
                with app.lock:
                    self.reply(200,app.state())
            elif self.path == "/api/reference" and self.authorized():
                self.reply(200,{"image":encode_image(app.template.reference)})
            else:
                self.reply(404,{"error":"요청을 찾을 수 없습니다."})

        def do_POST(self) -> None:
            if not self.authorized():
                self.reply(403,{"error":"로컬 세션을 다시 여세요."}); return
            if self.headers.get("Content-Type","").split(";")[0] != "application/json":
                self.reply(415,{"error":"JSON 요청이 필요합니다."}); return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0 < length <= MAX_BODY:
                    self.reply(413,{"error":"요청 크기를 확인하세요."}); return
                self.connection.settimeout(15)
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict) or not self.path.startswith("/api/"):
                    raise ValueError
                with app.lock:
                    result=app.handle(self.path[5:],data)
                self.reply(200,result)
            except (ValueError,TypeError,KeyError,cv2.error):
                self.reply(400,{"error":"입력을 확인하세요. 촬영·선택·확인 순서대로 진행해주세요."})
            except Exception:
                self.reply(503,{"error":"처리하지 못했습니다. 로컬 설치 상태를 확인하고 다시 시도해주세요."})

    return ThreadingHTTPServer(("127.0.0.1",port),Handler)
