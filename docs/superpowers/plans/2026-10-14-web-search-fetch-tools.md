# Web Search and Web Fetch Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement standalone `web_search` and `web_fetch` tools using the Python standard library with pluggable search providers (DuckDuckGo Lite, Tavily, Brave) and register them in the autonomous worker tool catalog.

**Architecture:** Create two self-contained, single-purpose tool modules under `src/verbix/tools/`: `web_search.py` (dispatches queries across DuckDuckGo Lite, Tavily, or Brave without external dependencies) and `web_fetch.py` (fetches web pages and strips HTML noise tags using `html.parser`). Both tools register in `TOOL_CATALOG` and automatically inherit output budgeting via `bind_output`.

**Tech Stack:** Python 3.13, standard library (`urllib.request`, `urllib.parse`, `html.parser`, `json`, `asyncio`), pytest, ruff. Zero new third-party dependencies.

**Spec:** [`docs/superpowers/specs/2026-10-14-web-search-fetch-tools-design.md`](file:///home/numair/dev/projects/verbix/docs/superpowers/specs/2026-10-14-web-search-fetch-tools-design.md)

## Global Constraints

- Python 3.13+ compatible.
- Zero new dependencies added to `pyproject.toml` (standard library only).
- Modular design per `AGENTS.md`: one thing does one thing, one tool per file.
- All unit tests must be hermetic and offline using mocked HTTP calls (no real network access in `pytest`).
- SSRF protection: reject non-HTTP schemes (`file://`, `ftp://`) and loopback/private addresses (`localhost`, `127.0.0.1`, `0.0.0.0`, `::1`).
- Output bounded by `bind_output` via `build_registry`.
- Full test suite plus `ruff check` must pass cleanly before any completion claim.

## Review Focus

1. **Non-HTTP or private host URL passed to `web_fetch`**: Must return an error message immediately without opening any network socket.
2. **DuckDuckGo Lite returns HTML with zero results or rate-limit challenge**: Must return a graceful, non-crashing status (`No results found...` or rate-limit guidance).
3. **Web page contains heavy script/style/nav boilerplates**: `web_fetch` must strip tags and extract human-readable text with intact paragraphs and headings.
4. **Provider selection resolution**: When `VERBIX_SEARCH_PROVIDER` is set, it overrides API keys; when unset, `TAVILY_API_KEY` / `BRAVE_API_KEY` take precedence over default DuckDuckGo.
5. **Tool catalog registration**: Adding `web_search` and `web_fetch` to `worker.yaml` allows `build_worker_registry` to instantiate both tools with `bind_output` attached.

---

### Task 1: `WebSearchTool` Implementation (`src/verbix/tools/web_search.py`)

**Files:**
- Create: `src/verbix/tools/web_search.py`
- Test: `tests/test_tools_web_search.py`

**Interfaces:**
- Consumes: `ToolContext`, `ToolResult` from `verbix.tools.base`.
- Produces: `WebSearchTool` with:
  - `name: str = "web_search"`
  - `side_effecting: bool = False`
  - `parameters_schema: dict[str, Any]` (accepts `query: str`, optional `max_results: int`)
  - `async execute(args: dict[str, Any], context: ToolContext) -> ToolResult`

- [ ] **Step 1: Write failing tests for `WebSearchTool`**

Write tests in `tests/test_tools_web_search.py` covering:
1. `test_web_search_duckduckgo_lite_parses_html`: Mock `urllib.request.urlopen` with DuckDuckGo Lite HTML snippet table and verify extracted `SearchResult` titles, URLs, and snippets.
2. `test_web_search_duckduckgo_no_results`: Mock DDG response with empty result table and verify `"No results found for query: '...'"` is returned.
3. `test_web_search_tavily_provider`: Mock Tavily JSON API response and verify structured output when `TAVILY_API_KEY` is set.
4. `test_web_search_brave_provider`: Mock Brave Search JSON API response and verify structured output when `BRAVE_API_KEY` is set.
5. `test_web_search_exa_provider`: Mock Exa JSON API response and verify structured output when `EXA_API_KEY` is set.
6. `test_web_search_provider_resolution`: Verify priority order (`VERBIX_SEARCH_PROVIDER` > keys > duckduckgo default).
7. `test_web_search_empty_query_error`: Verify `ToolResult` returns `is_error=True` when query is empty.
8. `test_web_search_http_error_handled`: Mock `urllib.error.HTTPError` (e.g. 500 or 429) and verify clean error message without unhandled exception.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_web_search.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'verbix.tools.web_search'`

- [ ] **Step 3: Implement `WebSearchTool` in `src/verbix/tools/web_search.py`**

Implement `WebSearchTool`:
- Schema: `query` (required string), `max_results` (integer, default 5, clamped between 1 and 10).
- Providers:
  - `_search_duckduckgo(query, max_results)`: POST `https://lite.duckduckgo.com/lite/` with `urllib.request` using `ssl.create_default_context(cafile=certifi.where())`, parse with `DuckDuckGoLiteParser(html.parser.HTMLParser)` skipping `<tr class="result-sponsored">` and extracting `<a class="result-link">` and `<td class="result-snippet">`.
  - `_search_tavily(query, max_results, api_key)`: POST `https://api.tavily.com/search` with JSON payload.
  - `_search_brave(query, max_results, api_key)`: GET `https://api.search.brave.com/res/v1/web/search` with `X-Subscription-Token` header.
  - `_search_exa(query, max_results, api_key)`: POST `https://api.exa.ai/search` with JSON payload and `x-api-key` header.
- Resolution logic:
  - Read `os.environ.get("VERBIX_SEARCH_PROVIDER")`.
  - Fall back to `"exa"` if `os.environ.get("EXA_API_KEY")`.
  - Fall back to `"tavily"` if `os.environ.get("TAVILY_API_KEY")`.
  - Fall back to `"brave"` if `os.environ.get("BRAVE_API_KEY")`.
  - Fall back to `"duckduckgo"`.
- Run synchronous HTTP requests in thread executor (`asyncio.to_thread`) to avoid blocking the event loop.
- Format results into numbered list:
  ```
  1. Title
     URL: https://...
     Snippet: ...
  ```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_web_search.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/web_search.py tests/test_tools_web_search.py
git commit -m "feat(tools): add web_search tool with pluggable providers"
```

---

### Task 2: `WebFetchTool` Implementation (`src/verbix/tools/web_fetch.py`)

**Files:**
- Create: `src/verbix/tools/web_fetch.py`
- Test: `tests/test_tools_web_fetch.py`

**Interfaces:**
- Consumes: `ToolContext`, `ToolResult` from `verbix.tools.base`.
- Produces: `WebFetchTool` with:
  - `name: str = "web_fetch"`
  - `side_effecting: bool = False`
  - `parameters_schema: dict[str, Any]` (accepts `url: str`, optional `format: str = "markdown"`)
  - `async execute(args: dict[str, Any], context: ToolContext) -> ToolResult`

- [ ] **Step 1: Write failing tests for `WebFetchTool`**

Write tests in `tests/test_tools_web_fetch.py` covering:
1. `test_web_fetch_extracts_markdown_by_default`: Mock `urlopen` returning HTML with headings, paragraphs, and list items; verify clean markdown formatting returned by default.
2. `test_web_fetch_formats`: Verify `format="text"` returns plain text without markdown markers, and `format="html"` returns raw HTML.
3. `test_web_fetch_cloudflare_challenge_retries_with_honest_ua`: Mock first request raising `HTTPError(403)` with `headers={"cf-mitigated": "challenge"}`; verify handler automatically retries with `User-Agent: verbix` and succeeds.
4. `test_web_fetch_blocks_ssrf`: Test with `file:///etc/passwd`, `http://localhost:8000/`, `http://127.0.0.1/`, `http://[::1]/`, `http://0.0.0.0/`; verify `is_error=True` without network calls.
5. `test_web_fetch_http_error_handled`: Mock `urllib.error.HTTPError(404, "Not Found")` and verify `ToolResult("error: failed to fetch '...': 404 Not Found", is_error=True)`.
6. `test_web_fetch_timeout_handled`: Mock `TimeoutError` and verify clean error message.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_web_fetch.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'verbix.tools.web_fetch'`

- [ ] **Step 3: Implement `WebFetchTool` in `src/verbix/tools/web_fetch.py`**

Implement `WebFetchTool`:
- Schema: `url` (required string), `format` (optional string, enum `["markdown", "text", "html"]`, default `"markdown"`).
- Validation: Validate scheme is `http` or `https`; parse hostname with `urllib.parse.urlparse` and reject `localhost`, loopback, or private hostnames.
- `HTMLTextExtractor(html.parser.HTMLParser)`:
  - Skip contents inside noise tags: `script`, `style`, `noscript`, `svg`, `nav`, `header`, `footer`, `aside`, `form`, and roles `navigation`/`banner`/`contentinfo`.
  - Maintain `ignored_stack` accounting for HTML void tags (`img`, `br`, `hr`, `input`, `meta`, `link`) to avoid void tags breaking end-tag tracking.
  - Track block elements (`p`, `h1`-`h6`, `li`, `div`, `article`, `section`, `tr`) to format appropriate markdown tags (`#`, `##`, `- `) or newlines.
  - Extract `<title>` tag and prepend `Title: <title>\nURL: <url>\n\n`.
  - Collapse excessive blank lines and whitespace.
- Fetch with `urllib.request.Request`:
  - Desktop Chrome User-Agent by default.
  - Catch `HTTPError(403)`: if header `cf-mitigated == "challenge"`, retry with `User-Agent: "verbix"`.
  - 30-second timeout, 5MB response size limit.
- Execute via `asyncio.to_thread`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_web_fetch.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/web_fetch.py tests/test_tools_web_fetch.py
git commit -m "feat(tools): add web_fetch tool with html noise stripping"
```

---

### Task 3: Catalog Registration and Worker Tool Configuration

**Files:**
- Modify: `src/verbix/tools/__init__.py`
- Modify: `src/verbix/agent/config/worker.yaml`
- Modify: `src/verbix/runtime/workers.py`
- Test: `tests/test_tools_catalog_web.py`

**Interfaces:**
- Consumes: `WebSearchTool` from `verbix.tools.web_search`, `WebFetchTool` from `verbix.tools.web_fetch`.
- Produces: `TOOL_CATALOG["web_search"]`, `TOOL_CATALOG["web_fetch"]`, updated `build_worker_registry`.

- [ ] **Step 1: Write failing test for catalog and worker registry registration**

Write `tests/test_tools_catalog_web.py`:
```python
from verbix.tools import TOOL_CATALOG, build_registry
from verbix.runtime.workers import build_worker_registry

def test_web_tools_in_catalog():
    assert "web_search" in TOOL_CATALOG
    assert "web_fetch" in TOOL_CATALOG

def test_build_registry_instantiates_web_tools(tmp_path):
    registry = build_registry(["web_search", "web_fetch"], workspace=tmp_path)
    definitions = registry.get_definitions()
    names = {d["function"]["name"] for d in definitions}
    assert "web_search" in names
    assert "web_fetch" in names

def test_build_worker_registry_includes_web_tools(tmp_path):
    registry = build_worker_registry(workspace=tmp_path)
    definitions = registry.get_definitions()
    names = {d["function"]["name"] for d in definitions}
    assert "web_search" in names
    assert "web_fetch" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools_catalog_web.py -v`  
Expected: FAIL with `AssertionError: assert 'web_search' in TOOL_CATALOG`

- [ ] **Step 3: Register web tools in catalog, worker.yaml, and workers.py**

1. In `src/verbix/tools/__init__.py`:
   - Import `WebSearchTool` from `verbix.tools.web_search`.
   - Import `WebFetchTool` from `verbix.tools.web_fetch`.
   - Add to `__all__`.
   - Add entries in `TOOL_CATALOG`:
     ```python
     "web_search": lambda runtime: WebSearchTool(),
     "web_fetch": lambda runtime: WebFetchTool(),
     ```
2. In `src/verbix/agent/config/worker.yaml`:
   - Add `web_search` and `web_fetch` to `tools:`.
   - Add description and prompt guidance for `web_search` and `web_fetch` in `format_error_template`.
3. In `src/verbix/runtime/workers.py`:
   - Add `"web_search"` and `"web_fetch"` to `build_worker_registry()` default tool names list.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools_catalog_web.py -v`  
Expected: PASS

- [ ] **Step 5: Run full test suite and linter**

Run:
```bash
uv run pytest
uv run ruff check
```
Expected: All tests pass, 0 linter errors.

- [ ] **Step 6: Commit**

```bash
git add src/verbix/tools/__init__.py src/verbix/agent/config/worker.yaml src/verbix/runtime/workers.py tests/test_tools_catalog_web.py
git commit -m "feat(worker): register web_search and web_fetch tools in worker catalog"
```
