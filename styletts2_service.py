"""Local StyleTTS2 service wrapper for the chat UI."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict


VOICE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = VOICE_DIR / "styletts2_settings.json"
WORKER_PATH = VOICE_DIR / "styletts2_worker.py"
CACHE_DIR = VOICE_DIR / "audio_cache"
CACHE_ROOT = VOICE_DIR / ".cache"
HF_CACHE_DIR = CACHE_ROOT / "huggingface"
MPL_CACHE_DIR = CACHE_ROOT / "matplotlib"
NLTK_CACHE_DIR = CACHE_ROOT / "nltk_data"
CACHED_PATH_CACHE_DIR = CACHE_ROOT / "cached_path"


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


class StyleTTS2Service:
    """Generate and cache StyleTTS2 WAV replies via a separate Python worker."""

    def __init__(self, settings_path: Path = SETTINGS_PATH) -> None:
        self.settings_path = settings_path.resolve()
        self.voice_dir = self.settings_path.parent
        self.cache_dir = self.voice_dir / "audio_cache"
        self.log_path = self.voice_dir / "styletts2_daemon.log"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        NLTK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHED_PATH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_handle: Any = None
        self._prewarm_started = False
        atexit.register(self.close)

    def _load_settings(self) -> Dict[str, Any]:
        return json.loads(self.settings_path.read_text(encoding="utf-8"))

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["HF_HOME"] = str(HF_CACHE_DIR)
        env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
        env["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
        env["NLTK_DATA"] = str(NLTK_CACHE_DIR)
        env["CACHED_PATH_CACHE_ROOT"] = str(CACHED_PATH_CACHE_DIR)
        return env

    def _resolve_python(self, settings: Dict[str, Any]) -> str:
        override = os.environ.get("NUCLEAR_LLM_STYLETTS2_PYTHON")
        if override:
            return override
        raw = settings.get("python_executable", "")
        if not raw:
            return sys.executable
        path = Path(raw)
        if not path.is_absolute():
            path = self.voice_dir / path
        return str(path.absolute())

    def _cache_key(self, text: str, emotion: str, settings: Dict[str, Any]) -> str:
        payload = {
            "text": _normalize_text(text),
            "emotion": emotion,
            "reference_voice_by_emotion": settings.get("reference_voice_by_emotion"),
            "model_checkpoint_path": settings.get("model_checkpoint_path"),
            "config_path": settings.get("config_path"),
            "alpha": settings.get("alpha"),
            "beta": settings.get("beta"),
            "diffusion_steps": settings.get("diffusion_steps"),
            "embedding_scale": settings.get("embedding_scale"),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:24]

    def _basic_status_error(self) -> Dict[str, Any] | None:
        if not self.settings_path.exists():
            return {
                "available": False,
                "backend": "styletts2",
                "detail": "Settings file is missing.",
            }
        settings = self._load_settings()
        if not settings.get("enabled", True):
            return {
                "available": False,
                "backend": "styletts2",
                "detail": "StyleTTS2 backend is disabled in settings.",
            }
        python_exec = self._resolve_python(settings)
        if not Path(python_exec).exists():
            return {
                "available": False,
                "backend": "styletts2",
                "detail": f"Configured Python was not found: {python_exec}",
            }
        return None

    def _stop_process_locked(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            try:
                if process.stdin and process.stdout and process.poll() is None:
                    process.stdin.write(json.dumps({"action": "stop"}) + "\n")
                    process.stdin.flush()
                    ready, _, _ = select.select([process.stdout], [], [], 2.0)
                    if ready:
                        process.stdout.readline()
            except Exception:
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            finally:
                self._stderr_handle = None

    def _ensure_process_locked(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        self._stop_process_locked()
        settings = self._load_settings()
        python_exec = self._resolve_python(settings)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = self.log_path.open("a", encoding="utf-8")
        command = [
            python_exec,
            "-u",
            str(WORKER_PATH),
            "--settings",
            str(self.settings_path),
            "--daemon",
        ]
        process = subprocess.Popen(
            command,
            cwd=str(self.voice_dir),
            env=self._build_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            bufsize=1,
        )
        self._process = process
        ready, _, _ = select.select([process.stdout], [], [], 15.0)
        if not ready:
            self._stop_process_locked()
            raise RuntimeError("StyleTTS2 worker did not acknowledge startup.")
        started_line = process.stdout.readline()
        if not started_line:
            self._stop_process_locked()
            raise RuntimeError("StyleTTS2 worker exited before startup completed.")
        try:
            payload = json.loads(started_line)
        except json.JSONDecodeError as exc:
            self._stop_process_locked()
            raise RuntimeError(f"StyleTTS2 worker returned invalid startup payload: {exc}") from exc
        if not payload.get("ok"):
            self._stop_process_locked()
            raise RuntimeError(payload.get("error") or "StyleTTS2 worker failed to start.")
        return process

    def _request_locked(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        process = self._ensure_process_locked()
        if process.stdin is None or process.stdout is None:
            self._stop_process_locked()
            raise RuntimeError("StyleTTS2 worker streams are unavailable.")
        process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_process_locked()
                raise RuntimeError(f"StyleTTS2 worker timed out during {payload.get('action', 'request')}.")
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                self._stop_process_locked()
                raise RuntimeError(f"StyleTTS2 worker timed out during {payload.get('action', 'request')}.")
            line = process.stdout.readline()
            if not line:
                self._stop_process_locked()
                raise RuntimeError("StyleTTS2 worker closed its output unexpectedly.")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def close(self) -> None:
        with self._lock:
            self._stop_process_locked()

    def prewarm_async(self) -> None:
        with self._lock:
            if self._prewarm_started:
                return
            self._prewarm_started = True

        def _run() -> None:
            try:
                with self._lock:
                    self._request_locked({"action": "warmup"}, timeout=600.0)
            except Exception:
                return

        threading.Thread(target=_run, daemon=True, name="styletts2-prewarm").start()

    def describe_status(self) -> Dict[str, Any]:
        basic_error = self._basic_status_error()
        if basic_error is not None:
            return basic_error

        try:
            with self._lock:
                payload = self._request_locked({"action": "status"}, timeout=60.0)
        except Exception as exc:
            return {"available": False, "backend": "styletts2", "detail": str(exc)}

        if payload.get("ok"):
            detail = payload.get("detail", "Ready")
            if payload.get("model_loaded"):
                detail = "StyleTTS2 model warmed and ready."
            return {"available": True, "backend": "styletts2", "detail": detail}

        detail = payload.get("error") or "StyleTTS2 runtime check failed."
        return {"available": False, "backend": "styletts2", "detail": detail}

    def synthesize(self, text: str, emotion: str | None = None) -> Dict[str, Any]:
        clean_text = _normalize_text(text)
        if not clean_text:
            return {"ok": False, "backend": "styletts2", "error": "Text must not be empty."}

        basic_error = self._basic_status_error()
        if basic_error is not None:
            return {"ok": False, "backend": "styletts2", "error": basic_error["detail"]}

        settings = self._load_settings()
        chosen_emotion = emotion or settings.get("default_emotion", "neutral")
        cache_key = self._cache_key(clean_text, chosen_emotion, settings)
        output_path = self.cache_dir / f"{cache_key}.wav"
        if output_path.exists():
            return {
                "ok": True,
                "backend": "styletts2",
                "audio_url": f"/voice_interact/audio_cache/{output_path.name}",
                "cache_hit": True,
                "emotion": chosen_emotion,
            }

        try:
            with self._lock:
                payload = self._request_locked(
                    {
                        "action": "synthesize",
                        "text": clean_text,
                        "output": str(output_path),
                        "emotion": chosen_emotion,
                    },
                    timeout=600.0,
                )
        except Exception as exc:
            return {"ok": False, "backend": "styletts2", "error": str(exc)}
        if not payload.get("ok"):
            return {
                "ok": False,
                "backend": "styletts2",
                "error": payload.get("error") or "StyleTTS2 synthesis failed.",
            }

        return {
            "ok": True,
            "backend": "styletts2",
            "audio_url": f"/voice_interact/audio_cache/{output_path.name}",
            "cache_hit": False,
            "emotion": chosen_emotion,
        }
