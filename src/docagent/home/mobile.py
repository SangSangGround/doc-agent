"""아이폰 개발 시연용 HTTPS 서버. 지정한 사설 IP, 인증서 지문, 실행별 토큰을 사용한다."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import ssl
import subprocess
import tempfile



def private_ipv4(host: str) -> str:
    ip = ipaddress.IPv4Address(host)
    ranges = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")
    if not any(ip in ipaddress.ip_network(network) for network in ranges):
        raise ValueError("같은 Wi-Fi의 PC 사설 IPv4 주소를 지정하세요.")
    return str(ip)


def connection_files(directory: Path, host: str) -> tuple[Path, Path]:
    """개발 인증서를 만든다. 개인키를 출력하거나 시스템 신뢰 저장소를 변경하지 않는다."""
    host = private_ipv4(host)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    cert, key = directory / "server-cert.pem", directory / "server-key.pem"
    if cert.exists() != key.exists():
        raise ValueError("인증서와 키가 모두 있는 폴더 또는 새 폴더를 지정하세요.")
    if not cert.exists():
        with tempfile.TemporaryDirectory(prefix=".tls-", dir=directory) as temporary:
            temp = Path(temporary)
            result = subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
                "-days", "30", "-subj", "/CN=DocAgent Development",
                "-addext", f"subjectAltName=IP:{host}",
                "-addext", "extendedKeyUsage=serverAuth",
                "-addext", "keyUsage=critical,digitalSignature,keyEncipherment",
                "-addext", "basicConstraints=critical,CA:FALSE",
                "-keyout", str(temp / "key.pem"), "-out", str(temp / "cert.pem"),
            ], capture_output=True, timeout=30, check=False)
            if result.returncode:
                raise ValueError("개발 인증서를 만들지 못했습니다. OpenSSL 설치를 확인하세요.")
            os.chmod(temp / "key.pem", 0o600)
            os.replace(temp / "key.pem", key)
            os.replace(temp / "cert.pem", cert)
    return cert, key


def connection_payload(cert: Path, host: str, port: int, token: str) -> dict:
    der = ssl.PEM_cert_to_DER_cert(cert.read_text(encoding="ascii"))
    return {"schema_version": 1, "server_url": f"https://{private_ipv4(host)}:{port}",
            "token": token, "certificate_sha256": hashlib.sha256(der).hexdigest()}


def write_pairing(path: Path, payload: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".pairing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def serve_mobile(template: Path, *, host: str, port: int, directory: Path,
                 demo: bool = False, session: Path | None = None) -> None:
    from docagent.home.server import HomeApplication, make_server
    from docagent.home.templates import load_template

    host = private_ipv4(host)
    if not 0 <= port <= 65535:
        raise ValueError("포트 범위를 확인하세요.")
    app = HomeApplication(load_template(template, demo=demo), session_path=session)
    cert, key = connection_files(directory, host)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    server = make_server(app, port, mobile_host=host, tls_context=context)
    try:
        pairing = directory / "pairing.json"
        write_pairing(pairing, connection_payload(cert, host, server.server_port, app.token))
        print(f"아이폰 연결: https://{host}:{server.server_port}\n연결 정보 파일: {pairing}\n"
              "앱에서 연결 정보를 붙여넣으세요. 실행마다 토큰이 바뀝니다. 종료: Ctrl+C", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
