(function () {
  const SETTINGS_URLS = ["/voice_interact/voice_settings.json", "/static/voice_settings.json"];
  const STORAGE_KEY = "nuclear_llm_voice_enabled";
  const state = {
    settings: {
      enabled_by_default: true,
      backend: "styletts2",
      backend_url: "/api/tts",
      status_url: "/api/tts/status",
      autoplay: true,
      default_emotion: "neutral",
    },
    enabled: true,
    available: false,
    backendDetail: "Checking StyleTTS2 backend...",
    audio: null,
  };

  function setStatus(message) {
    const status = document.getElementById("voice-status");
    if (status) {
      status.textContent = message;
    }
  }

  function loadStoredPreference() {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === null) {
      return !!state.settings.enabled_by_default;
    }
    return stored === "1";
  }

  function savePreference() {
    window.localStorage.setItem(STORAGE_KEY, state.enabled ? "1" : "0");
  }

  function stopAudio() {
    if (!state.audio) {
      return;
    }
    state.audio.pause();
    state.audio.currentTime = 0;
  }

  function refreshControlState() {
    const toggle = document.getElementById("voice-toggle-button");
    const stop = document.getElementById("voice-stop-button");
    if (!toggle || !stop) {
      return;
    }

    toggle.textContent = state.enabled ? "Voice On" : "Voice Off";
    toggle.classList.toggle("active", state.enabled);
    stop.disabled = !state.audio || state.audio.paused;

    if (!state.enabled) {
      setStatus("Voice off");
      return;
    }
    if (!state.available) {
      setStatus(state.backendDetail || "StyleTTS2 unavailable");
      return;
    }
    if (state.audio && !state.audio.paused) {
      setStatus("Speaking...");
      return;
    }
    setStatus("StyleTTS2 ready");
  }

  async function loadSettings() {
    for (const url of SETTINGS_URLS) {
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) {
          continue;
        }
        const payload = await response.json();
        state.settings = { ...state.settings, ...payload };
        return;
      } catch (_error) {
        continue;
      }
    }
  }

  async function loadBackendStatus() {
    try {
      const response = await fetch(state.settings.status_url, { cache: "no-store" });
      if (!response.ok) {
        state.available = false;
        state.backendDetail = "StyleTTS2 status check failed";
        return;
      }
      const payload = await response.json();
      state.available = !!payload.available;
      state.backendDetail = payload.detail || (state.available ? "StyleTTS2 ready" : "StyleTTS2 unavailable");
    } catch (_error) {
      state.available = false;
      state.backendDetail = "StyleTTS2 status check failed";
    }
  }

  async function requestSpeech(text) {
    if (!state.enabled || !state.available || !state.settings.autoplay) {
      refreshControlState();
      return;
    }

    setStatus("Generating speech...");

    try {
      const response = await fetch(state.settings.backend_url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          emotion: state.settings.default_emotion || "neutral",
        }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok || !payload.audio_url) {
        state.available = false;
        state.backendDetail = payload.error || "StyleTTS2 synthesis failed";
        refreshControlState();
        return;
      }

      state.available = true;
      state.backendDetail = payload.backend || "styletts2";
      stopAudio();
      state.audio.src = `${payload.audio_url}?v=${Date.now()}`;
      await state.audio.play();
      refreshControlState();
    } catch (_error) {
      state.backendDetail = "Audio playback blocked or StyleTTS2 failed";
      refreshControlState();
    }
  }

  function mountControls() {
    const toggle = document.getElementById("voice-toggle-button");
    const stop = document.getElementById("voice-stop-button");
    if (!toggle || !stop) {
      return;
    }

    toggle.addEventListener("click", () => {
      state.enabled = !state.enabled;
      savePreference();
      if (!state.enabled) {
        stopAudio();
      }
      refreshControlState();
    });

    stop.addEventListener("click", () => {
      stopAudio();
      refreshControlState();
    });
  }

  async function init() {
    state.audio = new Audio();
    state.audio.addEventListener("play", refreshControlState);
    state.audio.addEventListener("ended", refreshControlState);
    state.audio.addEventListener("pause", refreshControlState);
    state.audio.addEventListener("error", () => {
      state.backendDetail = "Audio playback failed";
      refreshControlState();
    });

    await loadSettings();
    state.enabled = loadStoredPreference();
    await loadBackendStatus();
    mountControls();
    refreshControlState();
  }

  window.addEventListener("assistant-reply", (event) => {
    window.setTimeout(() => {
      requestSpeech(event.detail && event.detail.text ? String(event.detail.text) : "");
    }, 0);
  });

  window.addEventListener("user-message", () => {
    stopAudio();
    refreshControlState();
  });

  window.addEventListener("chat-cleared", () => {
    stopAudio();
    refreshControlState();
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
