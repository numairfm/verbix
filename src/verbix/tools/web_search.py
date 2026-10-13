from __future__ import annotations

import asyncio
import html.parser
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except (ImportError, OSError):
    _SSL_CONTEXT = ssl.create_default_context()

from verbix.tools.base import ToolContext, ToolResult


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _clean_ddg_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urllib.parse.urlparse(raw_url)
    qs = urllib.parse.parse_qs(parsed.query)
    uddg = qs.get("uddg")
    if uddg:
        return uddg[0]
    return raw_url


class DuckDuckGoLiteParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_sponsored = False
        self._in_link = False
        self._in_snippet = False
        self._current_title = ""
        self._current_url = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        classes = (attr_dict.get("class") or "").split()

        if tag == "tr" and "result-sponsored" in classes:
            self._in_sponsored = True
            return

        if self._in_sponsored:
            return

        if tag == "a" and "result-link" in classes:
            if self._current_title and self._current_url:
                self.results.append(
                    SearchResult(
                        title=self._current_title,
                        url=self._current_url,
                        snippet="",
                    )
                )
            self._in_link = True
            self._title_parts = []
            self._current_url = _clean_ddg_url(attr_dict.get("href") or "")
            self._current_title = ""

        elif tag == "td" and "result-snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._in_sponsored:
            self._in_sponsored = False
            return

        if self._in_sponsored:
            return

        if tag == "a" and self._in_link:
            self._in_link = False
            self._current_title = " ".join("".join(self._title_parts).split())
            self._title_parts = []

        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join("".join(self._snippet_parts).split())
            self._snippet_parts = []
            if self._current_title and self._current_url:
                self.results.append(
                    SearchResult(
                        title=self._current_title,
                        url=self._current_url,
                        snippet=snippet,
                    )
                )
                self._current_title = ""
                self._current_url = ""

    def handle_data(self, data: str) -> None:
        if self._in_sponsored:
            return
        if self._in_link:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)

    def close(self) -> None:
        super().close()
        if self._current_title and self._current_url:
            self.results.append(
                SearchResult(
                    title=self._current_title,
                    url=self._current_url,
                    snippet="",
                )
            )
            self._current_title = ""
            self._current_url = ""


def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=15) as resp:
        html_content = resp.read().decode("utf-8", errors="replace")

    parser = DuckDuckGoLiteParser()
    parser.feed(html_content)
    parser.close()
    return parser.results[:max_results]


def _search_tavily(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    url = "https://api.tavily.com/search"
    payload = json.dumps({
        "query": query,
        "max_results": max_results,
        "api_key": api_key,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw_results = body.get("results", []) if isinstance(body, dict) else []
    results: list[SearchResult] = []
    for item in raw_results[:max_results]:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            link = str(item.get("url", "")).strip()
            snippet = str(item.get("content", "") or item.get("snippet", "")).strip()
            results.append(SearchResult(title=title, url=link, snippet=snippet))
    return results


def _search_brave(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    web_data = body.get("web", {}) if isinstance(body, dict) else {}
    raw_results = (
        web_data.get("results", [])
        if isinstance(web_data, dict)
        else body.get("results", [])
    )
    results: list[SearchResult] = []
    for item in raw_results[:max_results]:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            link = str(item.get("url", "")).strip()
            snippet = str(item.get("description", "") or item.get("snippet", "")).strip()
            results.append(SearchResult(title=title, url=link, snippet=snippet))
    return results


def _search_exa(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    url = "https://api.exa.ai/search"
    payload = json.dumps({
        "query": query,
        "numResults": max_results,
        "contents": {"text": {"maxCharacters": 500}},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw_results = body.get("results", []) if isinstance(body, dict) else []
    results: list[SearchResult] = []
    for item in raw_results[:max_results]:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            link = str(item.get("url", "")).strip()
            snippet = str(
                item.get("text", "")
                or item.get("snippet", "")
                or item.get("content", "")
            ).strip()
            results.append(SearchResult(title=title, url=link, snippet=snippet))
    return results


class WebSearchTool:
    name: str = "web_search"
    side_effecting: bool = False
    description: str = (
        "Searches the web using a search engine (DuckDuckGo Lite, Tavily, Brave, Exa) "
        "and returns titles, URLs, and snippets."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to look up on the web",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5, max: 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    @staticmethod
    def _resolve_provider() -> tuple[str, str | None]:
        def _get_key(name: str) -> str | None:
            val = os.environ.get(name, "").strip()
            return val if val else None

        explicit = os.environ.get("VERBIX_SEARCH_PROVIDER", "").strip().lower()
        if explicit:
            key = None
            if explicit == "exa":
                key = _get_key("EXA_API_KEY")
            elif explicit == "tavily":
                key = _get_key("TAVILY_API_KEY")
            elif explicit == "brave":
                key = _get_key("BRAVE_API_KEY")
            return explicit, key

        exa_key = _get_key("EXA_API_KEY")
        if exa_key:
            return "exa", exa_key
        tavily_key = _get_key("TAVILY_API_KEY")
        if tavily_key:
            return "tavily", tavily_key
        brave_key = _get_key("BRAVE_API_KEY")
        if brave_key:
            return "brave", brave_key
        return "duckduckgo", None

    def _dispatch_search(
        self, provider: str, query: str, max_results: int, api_key: str | None
    ) -> list[SearchResult]:
        if provider == "duckduckgo":
            return _search_duckduckgo(query, max_results)
        if provider == "tavily":
            if not api_key:
                raise ValueError("TAVILY_API_KEY environment variable is required for Tavily search")
            return _search_tavily(query, max_results, api_key)
        if provider == "brave":
            if not api_key:
                raise ValueError("BRAVE_API_KEY environment variable is required for Brave search")
            return _search_brave(query, max_results, api_key)
        if provider == "exa":
            if not api_key:
                raise ValueError("EXA_API_KEY environment variable is required for Exa search")
            return _search_exa(query, max_results, api_key)
        raise ValueError(f"Unknown search provider: '{provider}'")

    @staticmethod
    def _format_results(results: list[SearchResult], query: str) -> str:
        if not results:
            return f"No results found for query: '{query}'."
        entries = []
        for i, r in enumerate(results, 1):
            entries.append(f"{i}. {r.title}\n   URL: {r.url}\n   Snippet: {r.snippet}")
        return "\n\n".join(entries)

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult("error: query must not be empty", is_error=True)

        try:
            max_results = int(args.get("max_results", 5))
        except (ValueError, TypeError):
            max_results = 5
        max_results = max(1, min(10, max_results))

        provider, api_key = self._resolve_provider()

        try:
            results = await asyncio.to_thread(
                self._dispatch_search, provider, query, max_results, api_key
            )
            return ToolResult(self._format_results(results, query))
        except urllib.error.HTTPError as e:
            msg = f"error: search request failed with HTTP {e.code}: {e.reason}"
            if provider == "duckduckgo" and e.code in (403, 429):
                msg += " (set TAVILY_API_KEY, BRAVE_API_KEY, or EXA_API_KEY for reliable search API access)"
            return ToolResult(msg, is_error=True)
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                return ToolResult(
                    f"error: search request timed out for query '{query}'",
                    is_error=True,
                )
            return ToolResult(f"error: search network error: {e.reason}", is_error=True)
        except TimeoutError:
            return ToolResult(
                f"error: search request timed out for query '{query}'",
                is_error=True,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"error: search failed: {e}", is_error=True)
