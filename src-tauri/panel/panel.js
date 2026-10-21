"use strict";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const elements = {
  trayDot: document.getElementById("tray-dot"),
  trayState: document.getElementById("tray-state"),
  floorState: document.getElementById("floor-state"),
  transcript: document.getElementById("transcript"),
  caption: document.getElementById("caption"),
  taskList: document.getElementById("task-list"),
  noTasks: document.getElementById("no-tasks"),
  setup: document.getElementById("setup"),
  activity: document.getElementById("activity"),
  tasks: document.getElementById("tasks"),
  keyStatus: document.getElementById("key-status"),
  keyField: document.getElementById("provider-key"),
  keyError: document.getElementById("key-error"),
  saveKey: document.getElementById("save-key"),
  modelStatus: document.getElementById("model-status"),
  modelPath: document.getElementById("model-path"),
  modelError: document.getElementById("model-error"),
  downloadModel: document.getElementById("download-model"),
  downloadProgress: document.getElementById("download-progress"),
  downloadLine: document.getElementById("download-line"),
  autostartChoice: document.getElementById("autostart-choice"),
  finishSetup: document.getElementById("finish-setup"),
  setupError: document.getElementById("setup-error"),
  shortcutHint: document.getElementById("shortcut-hint"),
  copyCommand: document.getElementById("copy-command"),
  copyStatus: document.getElementById("copy-status"),
  setupShortcutHint: document.getElementById("setup-shortcut-hint"),
  setupCopyCommand: document.getElementById("setup-copy-command"),
  serviceError: document.getElementById("service-error"),
  serviceErrorText: document.getElementById("service-error-text"),
  retryService: document.getElementById("retry-service"),
  hidePanel: document.getElementById("hide-panel"),
};

function titleCase(value) {
  if (!value) {
    return "";
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function taskRow(task) {
  const row = document.createElement("li");
  row.className = "task";

  const header = document.createElement("div");
  header.className = "task-header";

  const id = document.createElement("span");
  id.className = "task-id";
  id.textContent = task.id;

  const state = document.createElement("span");
  state.className = `task-state task-${task.state}`;
  state.textContent = titleCase(task.state);

  header.append(id, state);

  const objective = document.createElement("p");
  objective.className = "task-objective";
  objective.textContent = task.objective;

  const result = document.createElement("p");
  result.className = "task-result";
  result.textContent = task.result;

  row.append(header, objective, result);
  return row;
}

function showMessage(element, message) {
  element.textContent = message || "";
  element.hidden = !message;
}

function renderSetup(setup) {
  const running = Boolean(setup.download && setup.download.running);

  elements.keyStatus.textContent = setup.key_ready
    ? "A provider key is ready."
    : "No provider key yet.";
  elements.keyField.disabled = !setup.key_editable;
  elements.saveKey.disabled = !setup.key_editable;
  showMessage(elements.keyError, "");

  elements.modelStatus.textContent = setup.model_ready
    ? "The speech model is ready."
    : "The speech model is not downloaded yet.";
  elements.modelPath.textContent = setup.model_path || "";
  elements.downloadModel.hidden = !setup.model_downloadable;
  elements.downloadModel.textContent = setup.model_ready
    ? "Re-download model"
    : "Download model";
  elements.downloadModel.disabled = running;
  elements.downloadProgress.hidden = !running;
  elements.downloadLine.textContent = setup.download ? setup.download.last_line : "";
  showMessage(elements.modelError, "");

  elements.autostartChoice.checked = Boolean(setup.autostart);
  elements.autostartChoice.disabled = !setup.autostart_supported;

  elements.finishSetup.disabled =
    !setup.key_ready || !setup.model_ready || running;
  showMessage(elements.setupError, setup.error);
}

function render(snapshot) {
  if (!snapshot) {
    return;
  }

  elements.trayDot.className = `dot tray-state-${snapshot.tray_state}`;
  elements.trayState.textContent = titleCase(snapshot.tray_state);
  elements.floorState.textContent = snapshot.floor;
  elements.transcript.textContent = snapshot.activity.transcript;
  elements.caption.textContent = snapshot.activity.caption;

  const tasks = Array.isArray(snapshot.tasks) ? snapshot.tasks : [];
  elements.taskList.replaceChildren(...tasks.map(taskRow));
  elements.noTasks.hidden = tasks.length > 0;

  const inSetup = Boolean(snapshot.setup.required);
  elements.setup.hidden = !inSetup;
  elements.activity.hidden = inSetup;
  elements.tasks.hidden = inSetup;
  if (inSetup) {
    renderSetup(snapshot.setup);
  }

  if (snapshot.shortcut.accelerator) {
    elements.shortcutHint.textContent = `Shortcut: ${snapshot.shortcut.accelerator}`;
    elements.copyCommand.hidden = true;
    elements.setupShortcutHint.textContent = `Shortcut: ${snapshot.shortcut.accelerator}`;
    elements.setupCopyCommand.hidden = true;
  } else if (snapshot.shortcut.fallback) {
    elements.shortcutHint.textContent = `Toggle command: ${snapshot.shortcut.fallback}`;
    elements.copyCommand.hidden = false;
    elements.setupShortcutHint.textContent = `Toggle command: ${snapshot.shortcut.fallback}`;
    elements.setupCopyCommand.hidden = false;
  } else {
    elements.shortcutHint.textContent = "No global shortcut is available.";
    elements.copyCommand.hidden = true;
    elements.setupShortcutHint.textContent = "No global shortcut is available.";
    elements.setupCopyCommand.hidden = true;
  }

  if (snapshot.error) {
    elements.shortcutHint.hidden = true;
    elements.serviceErrorText.textContent = snapshot.error;
    elements.serviceError.hidden = false;
  } else {
    elements.shortcutHint.hidden = false;
    elements.serviceErrorText.textContent = "";
    elements.serviceError.hidden = true;
  }
}

function showError(error) {
  elements.serviceErrorText.textContent = `Panel error: ${error}`;
  elements.serviceError.hidden = false;
}

function showSetupError(error) {
  elements.setupError.textContent = `Setup error: ${error}`;
  elements.setupError.hidden = false;
}

elements.hidePanel.addEventListener("click", () => {
  invoke("hide_panel");
});

elements.retryService.addEventListener("click", () => {
  invoke("restart_service");
});

elements.saveKey.addEventListener("click", async () => {
  const key = elements.keyField.value;
  if (!key) {
    showSetupError("Enter the provider key first.");
    return;
  }
  try {
    const snapshot = await invoke("save_provider_key", { key });
    // The key is never shown again after it is saved.
    elements.keyField.value = "";
    render(snapshot);
  } catch (error) {
    showSetupError(error);
  }
});

elements.downloadModel.addEventListener("click", async () => {
  try {
    render(await invoke("start_model_download"));
  } catch (error) {
    showSetupError(error);
  }
});

elements.autostartChoice.addEventListener("change", async () => {
  try {
    render(await invoke("set_autostart", { enabled: elements.autostartChoice.checked }));
  } catch (error) {
    showSetupError(error);
  }
});

elements.finishSetup.addEventListener("click", async () => {
  try {
    render(await invoke("finish_setup"));
  } catch (error) {
    showSetupError(error);
  }
});

function showCopyStatus(message) {
  elements.copyStatus.textContent = message || "";
  elements.copyStatus.hidden = !message;
}

function selectText(element) {
  const range = document.createRange();
  range.selectNodeContents(element);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

// The clipboard API needs permission the webview may refuse. The command
// stays selectable text, so it can always be copied by hand.
function copyCommand(text, source) {
  const clipboard = navigator.clipboard;
  if (!clipboard || typeof clipboard.writeText !== "function") {
    selectText(source);
    showCopyStatus("Clipboard access is not available. Select the command and copy it.");
    return;
  }
  clipboard.writeText(text).then(
    () => showCopyStatus("Command copied."),
    () => {
      selectText(source);
      showCopyStatus("Clipboard access was denied. Select the command and copy it.");
    },
  );
}

elements.copyCommand.addEventListener("click", () => {
  copyCommand(elements.shortcutHint.textContent.replace(/^Toggle command:\s*/, ""), elements.shortcutHint);
});

elements.setupCopyCommand.addEventListener("click", () => {
  copyCommand(
    elements.setupShortcutHint.textContent.replace(/^Toggle command:\s*/, ""),
    elements.setupShortcutHint,
  );
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    invoke("hide_panel");
  }
});

async function start() {
  await listen("shell://state", (event) => {
    render(event.payload);
  });
  render(await invoke("get_state"));
}

start().catch(showError);
