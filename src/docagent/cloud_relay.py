"""Same-Wi-Fi HTTPS relay for DocAgent on iPhone. Run with python -m docagent.cloud_relay."""
from __future__ import annotations

import argparse
import hmac
import os
import re
import secrets
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from docagent.cloud_providers import DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, Providers, RelayError, json_bytes
from docagent.home.mobile import connection_files, connection_payload, private_ipv4, write_pairing
from docagent.writing_fields import MAX_REQUEST, WritingFieldDetector


def wifi_address() -> str:
    """Resolve the Mac Wi-Fi interface rather than picking a VPN or the default route."""
    try:
        ports = subprocess.check_output(["networksetup", "-listallhardwareports"], text=True, timeout=4)
        match = re.search(r"Hardware Port: (?:Wi-Fi|AirPort)\s*\nDevice: (en\d+)\b", ports)
        if match:
            address = subprocess.check_output(["ipconfig", "getifaddr", match[1]], text=True, timeout=4)
            return private_ipv4(address.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    raise ValueError("맥 Wi-Fi IPv4 주소를 찾지 못했어요. --host 옵션에 맥의 Wi-Fi IPv4 주소를 지정해 주세요.")


def provider_settings(path: Path | None) -> Providers:
    names = ("OPENAI_API_KEY", "OPENAI_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL")
    values = {}
    if path:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, val = line.partition("=")
            if not sep:
                raise ValueError(f"키 설정 파일 {number}번째 줄에 KEY=value 형식을 사용하세요.")
            key = key.strip()
            if key in names:
                val = val.strip()
                if len(val) >= 2 and val[0] in ("'", '"') and val[-1] == val[0]:
                    val = val[1:-1]
                values[key] = val
    for name in names:
        if name in os.environ:
            values[name] = os.environ[name]
        val = values.get(name, "")
        if val and (not val.isascii() or not val.isprintable() or any(c.isspace() for c in val)):
            raise ValueError(f"{name} 값에 공백이나 잘못된 문자가 있어요.")
    return Providers(values.get("OPENAI_API_KEY", ""), values.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                     values.get("GEMINI_API_KEY", ""), values.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, tls: ssl.SSLContext, token: str, providers: Providers):
        self.token = token
        self.providers = providers
        self.writing = WritingFieldDetector(providers.gemini_key, providers.gemini_model)
        self.slots = threading.BoundedSemaphore(4)
        self.tls = tls
        super().__init__((private_ipv4(host), port), RelayHandler)

    def finish_request(self, request, client_address):
        # Handshake on the individual worker, so a stalled client cannot block the listener.
        request.settimeout(12)
        try:
            with self.tls.wrap_socket(request, server_side=True) as connection:
                super().finish_request(connection, client_address)
                self._close_after_response(connection)
        except (OSError, EOFError):
            pass

    @staticmethod
    def _close_after_response(connection: ssl.SSLSocket) -> None:
        # The handler has flushed the HTTP reply. Send TLS close_notify without waiting for the peer.
        connection.setblocking(False)
        try:
            connection.unwrap()
            return
        except ssl.SSLError:
            pass  # A client can still be uploading the body of an already rejected request.

        # Closing TCP with unread bytes can reset it before the client sees the reply.
        # Half-close first, then discard in-flight bytes without delaying the HTTP response.
        with socket.socket(fileno=connection.detach()) as transport:
            transport.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + 1.0
            while (remaining := deadline - time.monotonic()) > 0:
                transport.settimeout(remaining)
                if not transport.recv(64 * 1024):
                    break

    def handle_error(self, request, client_address):
        # Never print arbitrary exception data from a request.
        print("요청 처리 중 연결이 종료됐어요.", flush=True)


class RelayHandler(BaseHTTPRequestHandler):
    server: RelayServer
    # Send headers and small JSON errors together, including when rejecting before reading the body.
    wbufsize = 64 * 1024

    def log_message(self, format, *args):
        pass

    def reply(self, status: int, value: dict):
        data = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorize(self) -> bool:
        if self.headers.get("Origin") is not None:
            self.reply(403, {"error": "아이폰 앱에서 연결해 주세요."})
            return False
        host, port = self.server.server_address
        if self.headers.get_all("Host", []) != [f"{host}:{port}"]:
            self.reply(403, {"error": "중계 서버 주소를 확인해 주세요."})
            return False
        auth = self.headers.get_all("Authorization", [])
        expected = f"Bearer {self.server.token}".encode()
        if len(auth) != 1 or not hmac.compare_digest(auth[0].encode("utf-8"), expected):
            self.reply(401, {"error": "현재 맥에서 생성한 연결 정보를 다시 입력해 주세요."})
            return False
        return True

    def do_GET(self):
        if not self.authorize():
            return
        if self.path != "/v1/status":
            self.reply(404, {"error": "중계 서버 경로를 확인해 주세요."})
            return
        p = self.server.providers
        self.reply(200, {"answersReady": bool(p.openai_key), "fieldsReady": bool(p.gemini_key)})

    def do_POST(self):
        if not self.authorize():
            return
        handle = {"/v1/answer": self.server.providers.answer, "/v1/review": self.server.providers.review,
                  "/v1/fields": self.server.writing.fields}.get(self.path)
        if handle is None:
            self.reply(404, {"error": "중계 서버 경로를 확인해 주세요."})
            return
        if self.headers.get_content_type() != "application/json":
            self.reply(415, {"error": "지원하는 질문 형식을 사용해 주세요."})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            self.reply(400, {"error": "요청 길이를 확인해 주세요."})
            return
        length = int(lengths[0])
        limit = MAX_REQUEST if self.path == "/v1/fields" else 512 * 1024
        if not 0 < length <= limit:
            self.reply(413, {"error": "요청 크기를 확인해 주세요."})
            return
        if not self.server.slots.acquire(blocking=False):
            self.reply(429, {"error": "중계 서버가 사용 중이에요. 잠시 뒤 다시 시도해 주세요."})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise RelayError(400, "요청이 중간에 끊겼어요.")
            self.reply(200, handle(body))
        except RelayError as exc:
            self.reply(exc.status, {"error": str(exc)})
        except (TimeoutError, ConnectionError):
            return
        except Exception:
            self.reply(500, {"error": "중계 서버가 요청을 마치지 못했어요."})
        finally:
            self.server.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Mac Wi-Fi LAN IPv4 (자동 감지 가능)")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--directory", type=Path, default=Path("outputs/cloud-relay"))
    parser.add_argument("--env-file", type=Path, help="서비스 키 파일 (KEY=value)")
    args = parser.parse_args()
    try:
        providers = provider_settings(args.env_file)
        host = private_ipv4(args.host) if args.host else wifi_address()
        if not 0 <= args.port <= 65535:
            raise ValueError("포트 범위를 확인해 주세요.")
        token = secrets.token_urlsafe(32)
        args.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.directory, prefix=".tls-") as temp:
            cert, key = connection_files(Path(temp), host)
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.minimum_version = ssl.TLSVersion.TLSv1_2
            tls.load_cert_chain(cert, key)
            with RelayServer(host, args.port, tls, token, providers) as server:
                pairing = args.directory / "pairing.json"
                write_pairing(pairing, connection_payload(cert, host, server.server_port, token))
                print(f"중계 서버 주소 https://{host}:{server.server_port}\n아이폰 연결 파일 {pairing.resolve()}", flush=True)
                print(f"OpenAI 키 {'설정됨' if providers.openai_key else '미설정'}", flush=True)
                print(f"작성란 분석 Gemini 키 {'설정됨' if providers.gemini_key else '미설정'}", flush=True)
                server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        parser.exit(1, f"중계 서버를 시작할 수 없어요. {exc}\n")


if __name__ == "__main__":
    main()
