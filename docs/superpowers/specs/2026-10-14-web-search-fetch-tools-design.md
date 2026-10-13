# Design Spec: Web Search and Web Fetch Tools (`web_search` & `web_fetch`)

**Date**: 2026-10-14  
**Status**: Draft for Review  
**Context**: Verbix background worker autonomous web exploration and document reading  

---

## 1. Executive Summary & Goals

### Problem
Verbix background workers currently possess rich workspace tools (`read`, `write`, `edit`, `grep`, `glob`, `bash`, `finish`), but lack direct web access. When a user asks Verbix to look up current documentation, search GitHub repositories, research external APIs, or fetch an online tutorial, the worker has to resort to running arbitrary `curl`/`wget` commands in `bash`, which:
1. Clutters the workspace and shell history.
2. Returns raw minified HTML, SVG, or JavaScript bundles that blow past token limits.
3. Fails when search engines block headless `curl` user agents.

### Goals
1. **Dedicated `web_search` Tool**:
   - Pluggable search providers: DuckDuckGo Lite (default: free, keyless, zero-config), Tavily (optional: `TAVILY_API_KEY`), and Brave Search (optional: `BRAVE_API_KEY`).
   - Provider resolution order: configured provider (`VERBIX_SEARCH_PROVIDER` / config) -> auto-detected API key (`TAVILY_API_KEY`, `BRAVE_API_KEY`) -> fallback to `duckduckgo`.
   - Returns structured, readable results: Title, URL, and snippet.
2. **Dedicated `web_fetch` Tool**:
   - Fetches public web pages using Python's standard library (`urllib.request`).
   - Strips non-content noise elements (`<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<svg>`, etc.) using standard library `html.parser`.
   - Returns clean, markdown-like text bounded by output limits (`bind_output`).
3. **Strict Zero External Dependencies**:
   - Implemented entirely with Python's standard library (`urllib.request`, `urllib.parse`, `html.parser`, `json`, `asyncio`). No third-party packages or native wheels added to `pyproject.toml`.
4. **Dynamic Catalog Integration**:
   - Integrated into `TOOL_CATALOG` in `src/verbix/tools/__init__.py`.
   - Added to default tools in `src/verbix/agent/config/worker.yaml` and `src/verbix/runtime/workers.py`.
5. **Hermetic Testing**:
   - 100% offline unit tests using mock HTTP handlers; no network requests during `pytest`.

---

## 2. Architecture & Tool Interfaces

### File Organization
```
src/verbix/
└── tools/
    ├── web_search.py        # WebSearchTool: provider dispatch, DDG Lite parser, Tavily/Brave wire
    ├── web_fetch.py         # WebFetchTool: URL fetcher + HTMLTextExtractor
    ├── __init__.py          # Register in TOOL_CATALOG
    └── shared.py            # Output budgeting via bind_output
```

Both tools live in single, modular files complying with `AGENTS.md` ("one thing does one thing, ideally one thing per file").

---

### Tool 1: `web_search`

**File**: `src/verbix/tools/web_search.py`  
**Side-effecting**: `False`

#### Parameters Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query string to look up on the web"
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of search results to return (default: 5, max: 10)",
      "default": 5
    }
  },
  "required": ["query"]
}
```

#### Provider Dispatching & Wire Behavior
A simple `SearchResult` dataclass:
```python
@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
```

The provider resolution logic:
1. If `VERBIX_SEARCH_PROVIDER` is set in environment (or in app config `search_provider`), use that provider (`duckduckgo`, `tavily`, `brave`).
2. Else if `TAVILY_API_KEY` is present, use `tavily`.
3. Else if `BRAVE_API_KEY` is present, use `brave`.
4. Else default to `duckduckgo`.

#### DuckDuckGo Lite Wire Implementation
- **Endpoint**: `POST https://lite.duckduckgo.com/lite/`
- **SSL Context**: Uses `ssl.create_default_context(cafile=certifi.where())` (using `certifi` from the venv) to ensure rock-solid TLS verification across all Linux/macOS/Windows systems.
- **Headers**:
  - `User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36`
  - `Content-Type: application/x-www-form-urlencoded`
  - `Accept: text/html`
- **Body**: `q=<url-encoded-query>`
- **Parsing**: An `HTMLParser` that ignores `<tr class="result-sponsored">` (ad rows) and extracts clean organic results from `<a class="result-link">` and `<td class="result-snippet">`. Direct organic destination URLs are used (skipping internal DDG redirects). This endpoint does not require JavaScript or browser cookies.

#### Tavily & Brave Wire Implementations
- **Tavily**: `POST https://api.tavily.com/search` with JSON payload `{"query": query, "max_results": max_results, "api_key": key}`.
- **Brave**: `GET https://api.search.brave.com/res/v1/web/search?q=<query>&count=<count>` with header `X-Subscription-Token: <key>`.

#### Output Format
```
1. Title of Result
   URL: https://example.com/page
   Snippet: Relevant snippet text from the search engine result.

2. Another Title
   URL: https://example.com/another
   Snippet: Another snippet...
```
If no results: `No results found for query: '<query>'.`

---

### Tool 2: `web_fetch`

**File**: `src/verbix/tools/web_fetch.py`  
**Side-effecting**: `False`

#### Parameters Schema
```json
{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "The absolute HTTP or HTTPS URL of the web page to fetch"
    }
  },
  "required": ["url"]
}
```

#### Wire & Content Processing
1. **Validation**: URL must begin with `http://` or `https://`. Reject local file protocols (`file://`), private IP addresses or localhost (`127.0.0.1`, `localhost`) to prevent SSRF vulnerabilities.
2. **HTTP Fetch**: `urllib.request.Request` with `User-Agent: Mozilla/5.0 ...` and a 15-second timeout.
3. **HTML Extraction (`HTMLTextExtractor`)**:
   - Subclasses `html.parser.HTMLParser`.
   - Ignores content inside noise tags: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<noscript>`, `<svg>`, `<form>`, `<aside>`, elements with roles `navigation`/`banner`/`contentinfo`, or navigational classes.
   - Maintains an `ignored_stack` accounting for HTML void elements (`img`, `br`, `hr`, `input`, `meta`, `link`) to avoid broken tag-depth tracking on void tags.
   - Formats block elements (`<h1>`-`<h6>`, `<p>`, `<li>`, `<tr>`, `<div>`, `<article>`, `<section>`) with appropriate spacing and newlines.
   - Extracts page `<title>` at the top.
   - Normalizes multiple spaces and consecutive blank lines into clean text.
4. **Output Budgeting**:
   - The returned text passes through `bind_output` automatically via `build_registry`.

#### Output Format
```
Title: Example Domain
URL: https://example.com

This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.
```

---

## 3. Dynamic Tool Registry & Worker YAML

### `src/verbix/tools/__init__.py`
Add to `TOOL_CATALOG`:
```python
"web_search": lambda runtime: WebSearchTool(),
"web_fetch": lambda runtime: WebFetchTool(),
```

### `src/verbix/agent/config/worker.yaml`
Add to `agent.tools`:
```yaml
  tools:
    - read
    - write
    - edit
    - grep
    - glob
    - web_search
    - web_fetch
    - bash
    - finish
```
And add usage guidance in `format_error_template`:
```yaml
Use 'web_search' to find information on the web:
- Tool: web_search
- Arguments: {"query": "python asyncio tutorial"}

Use 'web_fetch' to read the content of a web page:
- Tool: web_fetch
- Arguments: {"url": "https://docs.python.org/3/library/asyncio.html"}
```

### `src/verbix/runtime/workers.py`
Update `build_worker_registry` defaults to include `web_search` and `web_fetch`.

---

## 4. Error Handling & Edge Cases

1. **Network Errors & Timeouts**:
   - Catch `urllib.error.URLError`, `urllib.error.HTTPError`, `TimeoutError`, and `socket.timeout`.
   - Return clean `ToolResult(f"error: failed to fetch '{url}': {e}", is_error=True)`.
2. **SSRF & Non-HTTP Schemes**:
   - URLs not starting with `http://` or `https://` immediately return an error without attempting a connection.
   - Reject attempts to connect to `localhost`, `127.0.0.1`, `0.0.0.0`, `::1`.
3. **Non-HTML Responses (e.g., JSON or Plain Text)**:
   - If `Content-Type` is `text/plain` or `application/json`, return the raw text without HTML stripping.
4. **Empty Search Query or Missing Parameters**:
   - Return clear validation error message if `query` is empty or whitespace.
5. **Rate Limiting / CAPTCHA on DDG**:
   - If DDG returns a 403/429 status code or an empty document, catch cleanly and return a helpful error (`error: duckduckgo rate limited or blocked; consider configuring tavily or brave`).

---

## 5. Testing & Verification

1. **Hermetic Unit Tests**:
   - Test `web_search` with mocked HTTP responses for:
     - DuckDuckGo Lite HTML parsing (valid results extracted).
     - DuckDuckGo no results found.
     - Tavily provider response JSON parsing.
     - Brave provider response JSON parsing.
     - Provider resolution order (explicit env var vs key detection vs fallback).
     - Network timeout and HTTP 500 error handling.
   - Test `web_fetch` with mocked HTTP responses for:
     - Valid HTML parsing: title extraction, noise tag removal (`<script>`, `<nav>`, `<style>`), block spacing.
     - Plain text / JSON responses without stripping.
     - SSRF blocking (`localhost`, `127.0.0.1`, `file:///etc/passwd`).
     - Network failure and HTTP 404/500 handling.
2. **Catalog & Worker Integration Test**:
   - Verify `build_registry` registers `web_search` and `web_fetch`.
   - Verify `build_worker_registry` includes both tools.
3. **Full Suite**:
   - `uv run pytest` (all 425+ tests pass).
   - `uv run ruff check` clean.
