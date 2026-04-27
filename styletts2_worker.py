"""Standalone StyleTTS2 worker invoked by the local chat server."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


STATE = {
    "model": None,
    "settings_path": None,
    "settings": None,
    "base_dir": None,
}


def _load_settings(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_optional_path(raw_value: str, base_dir: Path) -> str | None:
    if not raw_value:
        return None
    path = Path(raw_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _resolve_reference_voice(settings: dict, emotion: str, base_dir: Path) -> str | None:
    voices = settings.get("reference_voice_by_emotion") or {}
    selected = voices.get(emotion) or voices.get(settings.get("default_emotion", "neutral"), "")
    return _resolve_optional_path(selected, base_dir)


def _patch_torch_load_default() -> None:
    import torch

    original_load = torch.load
    if getattr(original_load, "_nuclear_llm_styletts2_patched", False):
        return

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    patched_load._nuclear_llm_styletts2_patched = True  # type: ignore[attr-defined]
    torch.load = patched_load


def run_healthcheck() -> dict:
    cache_root = Path(os.environ.get("NLTK_DATA", "")).resolve() if os.environ.get("NLTK_DATA") else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
    try:
        import nltk
        if cache_root:
            nltk.data.path.insert(0, str(cache_root))
        _patch_torch_load_default()
        from styletts2 import tts as _tts_module  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime dependency path
        return {
            "ok": False,
            "backend": "styletts2",
            "error": f"StyleTTS2 import failed: {exc}",
        }
    return {"ok": True, "backend": "styletts2", "detail": "StyleTTS2 runtime is importable."}


def _prepare_runtime(settings_path: Path):
    cache_root = Path(os.environ.get("NLTK_DATA", "")).resolve() if os.environ.get("NLTK_DATA") else None

    import nltk
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
        nltk.data.path.insert(0, str(cache_root))
    _patch_torch_load_default()
    from styletts2 import tts as styletts_tts

    settings = _load_settings(settings_path)
    base_dir = settings_path.parent
    return settings, base_dir, styletts_tts


def _get_model(settings_path: Path):
    resolved_settings_path = settings_path.resolve()
    if STATE["model"] is not None and STATE["settings_path"] == resolved_settings_path:
        return STATE["model"], STATE["settings"], STATE["base_dir"]

    settings, base_dir, styletts_tts = _prepare_runtime(resolved_settings_path)

    checkpoint_path = _resolve_optional_path(settings.get("model_checkpoint_path", ""), base_dir)
    config_path = _resolve_optional_path(settings.get("config_path", ""), base_dir)

    kwargs = {}
    if checkpoint_path:
        kwargs["model_checkpoint_path"] = checkpoint_path
    if config_path:
        kwargs["config_path"] = config_path

    with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
        model = styletts_tts.StyleTTS2(**kwargs)
    STATE["model"] = model
    STATE["settings_path"] = resolved_settings_path
    STATE["settings"] = settings
    STATE["base_dir"] = base_dir
    return model, settings, base_dir


def warmup(settings_path: Path) -> dict:
    try:
        _get_model(settings_path)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        return {"ok": False, "backend": "styletts2", "error": str(exc)}
    return {"ok": True, "backend": "styletts2", "detail": "StyleTTS2 model warmed and ready."}


def synthesize(text: str, output_path: Path, settings_path: Path, emotion: str) -> dict:
    try:
        model, settings, base_dir = _get_model(settings_path)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        return {
            "ok": False,
            "backend": "styletts2",
            "error": f"StyleTTS2 load failed: {exc}",
        }

    target_voice_path = _resolve_reference_voice(settings, emotion, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
        model.inference(
            text=text,
            target_voice_path=target_voice_path,
            output_wav_file=str(output_path),
            output_sample_rate=int(settings.get("output_sample_rate", 24000)),
            alpha=float(settings.get("alpha", 0.3)),
            beta=float(settings.get("beta", 0.7)),
            diffusion_steps=int(settings.get("diffusion_steps", 5)),
            embedding_scale=float(settings.get("embedding_scale", 1.0)),
        )

    return {
        "ok": True,
        "backend": "styletts2",
        "output_path": str(output_path),
        "emotion": emotion,
        "target_voice_path": target_voice_path,
    }


def _write_payload(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def run_daemon(settings_path: Path) -> int:
    _write_payload({"ok": True, "backend": "styletts2", "event": "started"})
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write_payload({"ok": False, "backend": "styletts2", "error": "Invalid JSON request."})
            continue

        action = request.get("action")
        if action == "stop":
            _write_payload({"ok": True, "backend": "styletts2", "event": "stopped"})
            return 0
        if action == "status":
            payload = run_healthcheck()
            if payload.get("ok"):
                payload["model_loaded"] = STATE["model"] is not None
            _write_payload(payload)
            continue
        if action == "warmup":
            _write_payload(warmup(settings_path))
            continue
        if action == "synthesize":
            text = str(request.get("text", "")).strip()
            output = str(request.get("output", "")).strip()
            emotion = str(request.get("emotion", "neutral")).strip() or "neutral"
            if not text:
                _write_payload({"ok": False, "backend": "styletts2", "error": "Text must not be empty."})
                continue
            if not output:
                _write_payload({"ok": False, "backend": "styletts2", "error": "Output path must not be empty."})
                continue
            _write_payload(
                synthesize(
                    text=text,
                    output_path=Path(output).resolve(),
                    settings_path=settings_path,
                    emotion=emotion,
                )
            )
            continue

        _write_payload({"ok": False, "backend": "styletts2", "error": f"Unknown action: {action}"})

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate StyleTTS2 speech for the local chat UI.")
    parser.add_argument("--settings", required=True, help="Path to the StyleTTS2 settings JSON.")
    parser.add_argument("--text", default="", help="Text to synthesize.")
    parser.add_argument("--output", default="", help="Target WAV file path.")
    parser.add_argument("--emotion", default="neutral", help="Emotion label used to choose a reference voice.")
    parser.add_argument("--healthcheck", action="store_true", help="Validate that the StyleTTS2 runtime is importable.")
    parser.add_argument("--daemon", action="store_true", help="Run as a long-lived JSON-line worker.")
    args = parser.parse_args()

    settings_path = Path(args.settings).resolve()

    if args.daemon:
        return run_daemon(settings_path)

    if args.healthcheck:
        payload = run_healthcheck()
    else:
        if not args.text.strip():
            payload = {"ok": False, "backend": "styletts2", "error": "Text must not be empty."}
        elif not args.output:
            payload = {"ok": False, "backend": "styletts2", "error": "Output path must not be empty."}
        else:
            payload = synthesize(
                text=args.text.strip(),
                output_path=Path(args.output).resolve(),
                settings_path=settings_path,
                emotion=args.emotion.strip() or "neutral",
            )

    _write_payload(payload)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
