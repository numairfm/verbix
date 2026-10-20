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
  shortcutHint: document.getElementById("shortcut-hint"),
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

  elements.setup.hidden = !snapshot.setup.required;

  if (snapshot.shortcut.accelerator) {
    elements.shortcutHint.textContent = `Shortcut: ${snapshot.shortcut.accelerator}`;
  } else if (snapshot.shortcut.fallback) {
    elements.shortcutHint.textContent = `Toggle command: ${snapshot.shortcut.fallback}`;
  } else {
    elements.shortcutHint.textContent = "No global shortcut is available.";
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

elements.hidePanel.addEventListener("click", () => {
  invoke("hide_panel");
});

elements.retryService.addEventListener("click", () => {
  invoke("restart_service");
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
