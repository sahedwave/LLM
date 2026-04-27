"""Lightweight local chat UI for the locked nuclear LLM."""

from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from src.execution_graph import (
    activate_dag_execution,
    authorize_entrypoint,
    close_bootstrap_window,
    mark_state,
)


PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "chat_ui"
STATIC_DIR = WEB_DIR / "static"
VOICE_DIR = PROJECT_DIR / "casual" / "voice_interact"
HOST = os.environ.get("NUCLEAR_LLM_CHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("NUCLEAR_LLM_CHAT_PORT", "8008"))


authorize_entrypoint()
close_bootstrap_window()
activate_dag_execution()
mark_state("EVAL_ONLY")

from casual.casual_router import load_casual_runtime, route as route_message  # noqa: E402
from casual.voice_interact.styletts2_service import StyleTTS2Service  # noqa: E402
from generate import load_runtime  # noqa: E402


RUNTIME = load_runtime()
CASUAL_RUNTIME = load_casual_runtime()
VOICE_SERVICE = StyleTTS2Service()
VOICE_SERVICE.prewarm_async()


def _read_text(path: Path) -> bytes:
    return path.read_bytes()


def _json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        _json_response(handler, {"error": "not_found"}, status=404)
        return

    content = _read_text(path)
    content_type, _ = mimetypes.guess_type(str(path))
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    handler.wfile.write(content)


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "NuclearLLMChat/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            _serve_file(self, WEB_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            relative = parsed.path.removeprefix("/static/")
            _serve_file(self, STATIC_DIR / relative)
            return
        if parsed.path.startswith("/voice_interact/"):
            relative = parsed.path.removeprefix("/voice_interact/")
            _serve_file(self, VOICE_DIR / relative)
            return
        if parsed.path == "/api/health":
            _json_response(self, {"status": "ok", "host": HOST, "port": PORT})
            return
        if parsed.path == "/api/tts/status":
            _json_response(self, VOICE_SERVICE.describe_status())
            return
        _json_response(self, {"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, {"error": "invalid_json"}, status=400)
            return

        if parsed.path == "/api/tts":
            text = str(payload.get("text", "")).strip()
            if not text:
                _json_response(self, {"error": "empty_text"}, status=400)
                return

            result = VOICE_SERVICE.synthesize(
                text=text,
                emotion=str(payload.get("emotion", "") or "").strip() or None,
            )
            status = 200 if result.get("ok") else 503
            _json_response(self, result, status=status)
            return

        if parsed.path != "/api/chat":
            _json_response(self, {"error": "not_found"}, status=404)
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            _json_response(self, {"error": "empty_message"}, status=400)
            return

        try:
            payload = route_message(message, RUNTIME, CASUAL_RUNTIME)
        except Exception as exc:  # pragma: no cover - defensive server path
            _json_response(
                self,
                {"error": "generation_failed", "details": str(exc)},
                status=500,
            )
            return

        _json_response(
            self,
            {
                "answer": payload["answer"],
                "route": payload["route"],
                "model_used": payload.get("model_used"),
                "pcgs_v2": payload["pcgs_v2"],
                "sas_score": payload["sas_score"],
                "used_simulation": payload["used_simulation"],
                "was_repaired": payload["was_repaired"],
                "simulation_influenced_output": payload["simulation_influenced_output"],
                "simulation_summary": payload["simulation_summary"],
                "route_reason": payload["route_reason"],
            },
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def run_server(host: str = HOST, port: int = PORT) -> None:
    """Run the local threaded HTTP chat server."""
    server = ThreadingHTTPServer((host, port), ChatHandler)
    print(f"Nuclear chat UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        VOICE_SERVICE.close()
        server.server_close()


if __name__ == "__main__":
    run_server()
