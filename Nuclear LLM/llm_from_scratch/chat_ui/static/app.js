const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const clearButton = document.getElementById("clear-button");
const template = document.getElementById("message-template");

function scoreTone(score) {
  if (score === null || score === undefined) return "neutral";
  if (score >= 0.7) return "good";
  if (score >= 0.4) return "warn";
  return "bad";
}

function addMessage(role, body, meta = "", details = null) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".role").textContent = role === "user" ? "You" : "Assistant";
  node.querySelector(".meta").textContent = meta;
  const bodyNode = node.querySelector(".message-body");
  bodyNode.textContent = body;

  if (details && role === "assistant") {
    const observability = document.createElement("div");
    observability.className = "observability";

    const rows = [
      ["Route", details.route || "unknown", ""],
      ["PCGS", details.pcgs_v2 ?? "n/a", scoreTone(details.pcgs_v2)],
      ["SAS", details.sas_score ?? "n/a", scoreTone(details.sas_score)],
      ["Used Simulation", details.used_simulation ? "yes" : "no", ""],
      ["Repaired", details.was_repaired ? "yes" : "no", ""],
      ["Simulation Influenced Output", details.simulation_influenced_output ? "yes" : "no", ""],
    ];

    if (details.simulation_summary) {
      rows.push(["k_eff", details.simulation_summary.k_eff ?? "n/a", ""]);
      rows.push(["Notes", details.simulation_summary.notes ?? "n/a", ""]);
    }

    for (const [label, value, tone] of rows) {
      const item = document.createElement("div");
      item.className = `obs-item ${tone}`;
      item.innerHTML = `<span class="obs-label">${label}</span><span class="obs-value">${value}</span>`;
      observability.appendChild(item);
    }

    node.appendChild(observability);
  }

  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Thinking..." : "Send";
  messageInput.disabled = isBusy;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage("user", message, new Date().toLocaleTimeString());
  messageInput.value = "";
  setBusy(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.details || payload.error || "Request failed");
    }
    addMessage("assistant", payload.answer, payload.route_reason || payload.route, payload);
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, "server");
  } finally {
    setBusy(false);
    messageInput.focus();
  }
});

clearButton.addEventListener("click", () => {
  chatLog.innerHTML = "";
  messageInput.focus();
});

addMessage(
  "assistant",
  "Ask a nuclear engineering question to start. Simulation-worthy prompts will use the Stage 6 verification path automatically.",
  "ready"
);
