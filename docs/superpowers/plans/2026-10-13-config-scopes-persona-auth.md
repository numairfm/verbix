# Config Scopes, Persona, Models, and Provider Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move user-tunable behavior (persona, hearing model, provider selection) out of code and flags into two config scopes with provider keys in `auth.json`.

**Architecture:** New `src/verbix/config.py` owns scope merging (builtin < global `~/.verbix/` < project `./.verbix/` < env < CLI) and `src/verbix/auth.py` owns provider lookup; `cli.py` consumes both at startup and fails fast with file-and-line errors. Secrets never enter config; agent tools hard-deny secret filenames.

**Tech Stack:** Python 3.13, stdlib only (yaml, json, pathlib, os), pytest.

**Spec:** No separate spec doc per owner preference; the design was approved in chat. Key decisions recapped: v1 surface is `stt_model` + `PERSONA.md` + `provider` name only; `audio_sink` deliberately excluded; project scope never auto-created; persona is overlay-only, never replacing core prompts.

## Global Constraints

- **No git commits unless explicitly instructed** (repo rule overrides the per-task commit steps below — treat them as ready-to-run commands awaiting instruction).
- **TDD:** failing test first for every behavior change.
- **Full suite (`uv run pytest -q`) + `uv run ruff check src tests` green** before any completion claim.
- **No new dependencies.** No new config keys beyond `stt_model`, `provider`, persona files.
- **PTT, agent, and history behavior unchanged** (config only adds new inputs; defaults preserve current behavior exactly).

## Review Focus

1. **Malformed `auth.json`** (bad JSON, missing key for a keyed provider): must fail at startup naming file and problem, never mid-turn — pinned in Task 4.
2. **Tool reading `~/.verbix/auth.json` or `./.env`**: must be denied with a clear error even though the path resolves inside the workspace — pinned in Task 5.
3. **Worker finish log leaking keys**: a summary containing `*_KEY=...` must be redacted in the debug log — pinned in Task 5.
4. **Unknown config keys**: must fail loudly naming file and line (repo convention), not silently ignored — pinned in Task 1.
5. **Missing everything** (no flag/env/file/entry): startup error must state exactly what to add where, with an example block — pinned in Task 4.

---

### Task 1: Config scope loader

**Files:**
- Create: `src/verbix/config.py`
- Test: `tests/test_config_scopes.py`

**Interfaces:**
- Consumes: `~/.verbix/config.yaml`, `<workspace>/.verbix/config.yaml` (both optional), env (`VERBIX_PROVIDER`), argparse namespace (optional).
- Produces: `load_config(workspace: Path | None, args: argparse.Namespace | None = None) -> dict` with keys subset of `{"provider": str | None, "stt_model": str | None}`; `read_persona(workspace) -> str` (global + project `PERSONA.md` concatenated project-after-global, guidance `<!-- -->` block stripped, missing files yield `""`).

- [x] **Step 1: Write the failing tests**

```python
def test_unknown_config_key_fails_loudly(tmp_path, monkeypatch):
    # global config with bogus key -> ValueError naming file and line-ish context
def test_project_overrides_global(tmp_path, monkeypatch):
    # global provider=openrouter, project provider=local -> "local"
def test_missing_files_yield_empty_defaults(tmp_path, monkeypatch):
    # no files, no env -> {"provider": None, "stt_model": None}, persona ""
def test_persona_strips_guidance_block(tmp_path, monkeypatch):
    # PERSONA.md with leading <!-- ... --> block -> block absent, user text kept
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_scopes.py -v`
Expected: FAIL with "No module named 'verbix.config'" (or collection error).

- [x] **Step 3: Implement `load_config` and `read_persona` in `src/verbix/config.py`**

Precedence: builtin defaults < global file < project file < `VERBIX_PROVIDER` env (provider only) < CLI namespace attrs (`provider`, `stt_model`) when set and not None. YAML parse errors raise ValueError naming the file. HOME honored via `Path.home()` (monkeypatchable in tests). Only `provider`/`stt_model` keys accepted; anything else raises ValueError naming file and key.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_scopes.py -v`
Expected: PASS.

- [x] **Step 5: Commit (only on explicit instruction)**

```bash
git add src/verbix/config.py tests/test_config_scopes.py
git commit -m "feat(config): two-scope config loader with loud unknown-key errors"
```

---

### Task 2: Wire persona and stt_model into startup

**Files:**
- Modify: `src/verbix/cli.py` (`build_runtime`, help text if a flag is added — none needed)
- Test: `tests/test_config_scopes.py` (append)

**Interfaces:**
- Consumes: Task 1 `load_config`/`read_persona`.
- Produces: unchanged `Runtime` behavior when no files exist (defaults preserve today exactly).

- [x] **Step 1: Write the failing tests**

```python
def test_persona_appends_as_section_not_replacement(tmp_path, monkeypatch):
    # PERSONA.md "Be chill." -> runtime system prompt contains core voice constraints AND "Be chill."
def test_cli_system_prompt_flag_beats_persona_file(tmp_path, monkeypatch):
    # --system-prompt set + PERSONA.md present -> flag text wins entirely
def test_stt_model_from_config_reaches_transcribe_stt(tmp_path, monkeypatch):
    # config stt_model=<path> -> TranscribeSTT constructed with that path (monkeypatch constructor to capture)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_scopes.py -v`
Expected: FAIL (persona text absent from prompt; default model path used).

- [x] **Step 3: Implement wiring in `src/verbix/cli.py`**

In `build_runtime`: load config for workspace; persona text appended to the system prompt as its own `# USER PERSONA` section (after core prompt, before or with existing `--system-prompt` handling — flag replaces file persona when set); resolved `stt_model` passed to `TranscribeSTT(model_path=...)` (explicit `--stt-model` flag still wins; then `VERBIX_STT_MODEL` inside `find_default_stt_model` stays as today for the fallback chain).

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_scopes.py tests/test_transcribe_stt.py -q`
Expected: PASS.

- [x] **Step 5: Commit (only on explicit instruction)**

```bash
git add src/verbix/cli.py tests/test_config_scopes.py
git commit -m "feat(config): wire persona overlay and stt_model into startup"
```

---

### Task 3: Model lookup order through .verbix scopes

**Files:**
- Modify: `src/verbix/voice/transcribe_stt.py` (`find_default_stt_model`)
- Test: `tests/test_transcribe_stt.py` (append)

**Interfaces:**
- Consumes: Task 1 config `stt_model` value (already threaded by Task 2); `workspace` for project scope.
- Produces: unchanged discovery behavior when scopes are empty.

- [x] **Step 1: Write the failing tests**

```python
def test_project_verbix_models_scope_wins_over_voice_runtime(tmp_path, monkeypatch):
    # ./.verbix/models/stt/a.gguf + voice-runtime fallback present -> resolves a.gguf
def test_global_verbix_models_scope_used(tmp_path, monkeypatch):
    # ~/.verbix/models/stt/b.gguf only -> resolves b.gguf
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcribe_stt.py -v`
Expected: FAIL (scopes ignored).

- [x] **Step 3: Implement scope lookup in `find_default_stt_model`**

Order: explicit path arg > `VERBIX_STT_MODEL` env (existing) > `<workspace>/.verbix/models/**/*.gguf` (needs workspace param — add optional `workspace: Path | None = None`, callers pass runtime workspace) > `~/.verbix/models/**/*.gguf` > existing `voice-runtime` fallback. Document `models/stt|tts|llm` reservation in a code comment; only `stt` is read.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcribe_stt.py -q`
Expected: PASS.

- [x] **Step 5: Commit (only on explicit instruction)**

```bash
git add src/verbix/voice/transcribe_stt.py tests/test_transcribe_stt.py
git commit -m "feat(config): stt model discovery through .verbix scopes"
```

---

### Task 4: auth.json provider selection

**Files:**
- Create: `src/verbix/auth.py`
- Modify: `src/verbix/cli.py` (`--provider` flag, `build_runtime` wiring)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `~/.verbix/auth.json` (`{provider: {type: "api", key, model?, base_url?}}`), Task 1 config `provider` name.
- Produces: `resolve_provider(name) -> dict(key, model, base_url)`; wired into `LitellmModelConfig` as today (`api_key`, `api_base`, model name precedence: `--model` > entry model > `--model` default).

- [x] **Step 1: Write the failing tests**

```python
def test_select_entry_by_name(tmp_path, monkeypatch):
    # two entries -> correct key/model/base_url for chosen name
def test_missing_entry_fails_startup_with_guidance(tmp_path, monkeypatch):
    # config names unknown provider -> ValueError with file path + example block
def test_env_keys_still_work_without_auth_file(tmp_path, monkeypatch):
    # no auth.json anywhere -> today's env behavior unchanged (no error)
def test_flag_model_beats_entry_model(...):
    # --model set + entry model set -> flag wins
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_auth.py -v`
Expected: FAIL (no module).

- [x] **Step 3: Implement `src/verbix/auth.py` + CLI wiring**

`auth.py`: read `~/.verbix/auth.json` (0600 expected, warn-not-fail on loose perms? No — YAGNI: read regardless, note in docs), validate entry shape (`type` must be `"api"`; key string, model/base_url optional strings; malformed → ValueError naming file). `cli.py`: add `--provider` flag + `VERBIX_PROVIDER` env read; resolution order flag > env > config `provider`; entry values feed `model_kwargs["api_key"/"api_base"]` and model name per precedence above; absolute failure (nothing anywhere and no env keys) raises ValueError with exact remediation (file path + example JSON block).

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_auth.py tests/test_ipc_and_cli.py -q`
Expected: PASS.

- [x] **Step 5: Commit (only on explicit instruction)**

```bash
git add src/verbix/auth.py src/verbix/cli.py tests/test_auth.py
git commit -m "feat(auth): per-provider auth.json selection with fail-fast guidance"
```

---

### Task 5: Secrets discipline (tool deny + log redaction)

**Files:**
- Modify: file-path gate used by tools (`src/verbix/tools/shared.py` `resolve_contained_path` or the read/edit/write tools — implementer picks the single choke point covering read+edit+write)
- Modify: `src/verbix/runtime/runtime.py` (`_on_task_finished` debug log)
- Test: `tests/test_tools.py` (append), `tests/test_runtime_e2e.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: unchanged behavior for all non-secret paths.

- [x] **Step 1: Write the failing tests**

```python
async def test_read_auth_json_denied(tmp_path):
    # workspace .verbix/auth.json + ~/.verbix/auth.json + .env reads -> is_error with clear message
async def test_finish_log_redacts_keys(tmp_path, caplog):
    # _on_task_finished with summary containing OPENROUTER_API_KEY=sk-secret -> log shows redaction, never the value
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py tests/test_runtime_e2e.py -q`
Expected: FAIL (reads succeed; key logged verbatim).

- [x] **Step 3: Implement deny + redaction**

Deny basenames `auth.json`, `auth.*`, `.env` at the single choke point (reads AND writes — workers must not overwrite secrets either). Redact `[A-Z_]*KEY\s*=\s*\S+` and `api_key`-valued JSON in the finish debug log only (payload itself untouched).

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py tests/test_runtime_e2e.py -q`
Expected: PASS.

- [x] **Step 5: Commit (only on explicit instruction)**

```bash
git add src/verbix/tools/shared.py src/verbix/runtime/runtime.py tests/test_tools.py tests/test_runtime_e2e.py
git commit -m "feat(security): deny secret-file tool access, redact keys in debug logs"
```

---

### Task 6: Global scaffold + docs

**Files:**
- Modify: `src/verbix/cli.py` (call site), `src/verbix/config.py` (`ensure_global_scaffold`)
- Modify: `README.md` (config section)
- Test: `tests/test_config_scopes.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2 formats.
- Produces: first-run scaffold; project scope never auto-created.

- [x] **Step 1: Write the failing tests**

```python
def test_scaffold_creates_global_once_with_guidance(tmp_path, monkeypatch):
    # empty HOME -> run -> config.yaml with commented guidance + PERSONA.md with stripped guidance block; second run adds nothing
def test_scaffold_never_touches_project_dir(tmp_path, monkeypatch):
    # workspace without .verbix -> run -> still absent
def test_deleted_scaffold_not_recreated(tmp_path, monkeypatch):
    # marker of prior scaffold exists but files deleted -> log pointer, no rewrite
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_scopes.py -v`
Expected: FAIL (no function).

- [x] **Step 3: Implement `ensure_global_scaffold` + CLI call + README**

Scaffold writes commented `config.yaml` (`stt_model` commented example only) and `PERSONA.md` with `<!-- guidance -->` + empty body; a marker records completion. Call once at CLI startup (never on import, never inside `build_runtime` so tests stay hermetic). README documents scopes, precedence, PERSONA.md, auth.json selection, and the `.gitignore-models` warning for project weights.

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_scopes.py -q`
Expected: PASS.

- [x] **Step 5: Commit (only on explicit instruction)**

```bash
git add src/verbix/config.py src/verbix/cli.py README.md tests/test_config_scopes.py
git commit -m "feat(config): global scaffold, docs, project scope stays user-created"
```
