# Verbix

Verbix is a voice-operated desktop assistant: the user talks, keeps talking
while work happens in the background, and hears each result in the same voice.

## Language

### Product surface

**Shell**:
The Rust desktop process that owns the tray, the global shortcut, autostart,
the panel, and the voice service process.
_Avoid_: app, frontend, UI.

**Voice service**:
The Python process that owns the runtime, voice stack, tools, and IPC.
_Avoid_: backend, daemon, server, agent process.

**Sidecar**:
The voice service running as a child of the shell.
_Avoid_: subprocess, background process.

**Session**:
The time from shell start to quit. History is not kept across sessions.
_Avoid_: run, instance.

**Tray**:
The OS status area where the shell lives when no panel is open.
_Avoid_: notification area, systray.

**Tray state**:
What the tray icon shows at a glance: idle, listening, speaking, working, or
error. Working and error describe tasks or the service, not the floor.
_Avoid_: status, mode.

**Panel**:
The shell's single window: floor state, captions, and tasks. Opened from the
tray; closing it hides the shell.
_Avoid_: window, dashboard, HUD.

**Overlay**:
A click-through or always-on-top surface drawn over other applications. Not
part of v1.
_Avoid_: HUD, pill.

**Transcript**:
The user's finalized speech as recognized. Shown in the panel; partial
transcripts do not exist.
_Avoid_: prompt, caption.

**Caption**:
Assistant text shown in the panel while it is spoken.
_Avoid_: subtitle, transcript.

**First-run setup**:
The one-time panel for the provider key, the speech model download, and the
autostart choice. The only settings surface in v1.
_Avoid_: onboarding, settings UI, wizard.

**Global shortcut**:
The key combination that toggles listening while Verbix is not focused.
Registered by the shell, not by the voice service.
_Avoid_: hotkey, PTT key.

**Shortcut fallback**:
A desktop-level shortcut bound to the shell's toggle command, used when the
OS refuses a global grab. On Wayland this is the supported path.
_Avoid_: workaround, manual shortcut.

**Autostart**:
Launching the shell at login, hidden in the tray.
_Avoid_: run on startup, login item, service.

**Single instance**:
One shell per user. Launching again forwards the command (toggle, show) to
the running shell and exits.
_Avoid_: lock, singleton.

### Voice loop

**Floor**:
The right to produce audio. Exactly one holder at a time.
_Avoid_: mic, turn.

**Floor state**:
Who holds the floor: idle, listening, or speaking.
_Avoid_: mode, status.

**Listening**:
The floor state in which the microphone is open.
_Avoid_: recording, capture.

**Speaking**:
The floor state in which Verbix audio is playing.
_Avoid_: playing, talking.

**Push-to-talk**:
The interaction style in which the user opens the mic, speaks, and closes it,
rather than an always-listening wake word. In v1 the toggle is press to start,
press to stop.
_Avoid_: continuous mode, hands-free.

**Toggle**:
The single user action that starts listening, stops listening, or interrupts
speaking and starts listening. The product's only voice control.
_Avoid_: PTT, mic button.

**Barge-in**:
Interrupting Verbix while it speaks; playback stops and listening starts.
_Avoid_: interrupt, cancel.

**Announcement**:
A completed worker result delivered by voice in the foreground's voice.
_Avoid_: notification, wake-up.

**Delivery state**:
How an announcement reached the user: pending (not yet spoken), heard, or
interrupted.
_Avoid_: status.

### Work

**Foreground**:
The one agent the user talks to. It answers directly or delegates.
_Avoid_: main agent, receptionist, assistant.

**Worker**:
A background agent that receives only an objective and works in the
workspace.
_Avoid_: subagent, task.

**Announcer**:
The toolless companion that speaks completed worker results.
_Avoid_: notifier, companion.

**Objective**:
The spawn text a worker receives; the only context it sees.
_Avoid_: prompt, task description.

**Task**:
A unit of background work started by the foreground.
_Avoid_: job, thread.

**Task state**:
A task's lifecycle: registered, running, completed, failed, or cancelled.
Orthogonal to floor state.
_Avoid_: status.

**Workspace**:
The directory given read-write to workers.
_Avoid_: cwd, sandbox.

### Data

**Global state**:
Machine-wide Verbix data that survives upgrades: configuration, persona,
memory, credentials, logs, and speech models.
_Avoid_: app data, user data.

**Persona**:
A user-written overlay appended to the system prompt. It never replaces core
prompts.
_Avoid_: character, custom prompt.

**Memory**:
Durable user facts that persist across sessions and are injected into
context.
_Avoid_: context, history.

### Shipping

**Bundle**:
A per-OS artifact built from the same codebase.
_Avoid_: installer, package.

**AppImage**:
The Linux bundle in v1. It runs from wherever the user puts it; installing it
is the user's choice.
_Avoid_: installer, deb, flatpak.

**Universal**:
One codebase that builds a bundle per OS. Not one binary that runs
everywhere.
_Avoid_: cross-platform build, single binary.
