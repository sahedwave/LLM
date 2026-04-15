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
HOST = os.environ.get("NUCLEAR_LLM_CHAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("NUCLEAR_LLM_CHAT_PORT", "8008"))


authorize_entrypoint()
close_bootstrap_window()
activate_dag_execution()
mark_state("EVAL_ONLY")

from generate import generate_text, load_runtime  # noqa: E402
from stage6_openmc.tool_router import route_query  # noqa: E402


RUNTIME = load_runtime()


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
        if parsed.path == "/api/health":
            _json_response(self, {"status": "ok", "host": HOST, "port": PORT})
            return
        _json_response(self, {"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            _json_response(self, {"error": "not_found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, {"error": "invalid_json"}, status=400)
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            _json_response(self, {"error": "empty_message"}, status=400)
            return

        try:
            payload = generate_text(query=message, runtime=RUNTIME, return_metadata=True)
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
        server.server_close()


if __name__ == "__main__":
    run_server()
