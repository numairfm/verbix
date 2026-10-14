import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from verbix.tools.base import ToolContext
from verbix.tools.web_search import DuckDuckGoLiteParser, WebSearchTool


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


@pytest.fixture
def search_tool() -> WebSearchTool:
    return WebSearchTool()


def _mock_response(body_bytes: bytes, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body_bytes
    resp.status = status
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


@pytest.mark.asyncio
async def test_web_search_duckduckgo_lite_parses_html(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    html_payload = b"""<!DOCTYPE html>
<html>
<body>
<table>
  <tr class="result-sponsored">
    <td><a class="result-link" href="https://ad.example.com">Ad Title</a></td>
    <td class="result-snippet">Ad snippet that should be skipped</td>
  </tr>
  <tr>
    <td>
      <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org%2Fdoc&rut=123">Python Documentation</a>
    </td>
  </tr>
  <tr>
    <td class="result-snippet">
      Official documentation and tutorials for the Python programming language.
    </td>
  </tr>
  <tr>
    <td>
      <a class="result-link" href="https://realpython.com">Real Python</a>
    </td>
  </tr>
  <tr>
    <td class="result-snippet">
      Python tutorials and articles for developers.
    </td>
  </tr>
</table>
</body>
</html>"""

    with patch("urllib.request.urlopen", return_value=_mock_response(html_payload)) as mock_urlopen:
        res = await search_tool.execute({"query": "python"}, tool_context)

    assert res.is_error is False
    assert "1. Python Documentation" in res.observation
    assert "URL: https://python.org/doc" in res.observation
    assert "Snippet: Official documentation and tutorials for the Python programming language." in res.observation
    assert "2. Real Python" in res.observation
    assert "URL: https://realpython.com" in res.observation
    assert "Snippet: Python tutorials and articles for developers." in res.observation
    assert "Ad Title" not in res.observation
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "https://lite.duckduckgo.com/lite/"


@pytest.mark.asyncio
async def test_web_search_duckduckgo_no_results(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    html_payload = b"""<!DOCTYPE html>
<html>
<body>
<table>
  <tr><td>No results found.</td></tr>
</table>
</body>
</html>"""

    with patch("urllib.request.urlopen", return_value=_mock_response(html_payload)):
        res = await search_tool.execute({"query": "nonexistent query"}, tool_context)

    assert res.is_error is False
    assert res.observation == "No results found for query: 'nonexistent query'."


@pytest.mark.asyncio
async def test_web_search_tavily_provider(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    tavily_json = json.dumps({
        "results": [
          {
            "title": "Tavily Search Result",
            "url": "https://tavily.com/result",
            "content": "Tavily extracted snippet text."
          }
        ]
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_mock_response(tavily_json)) as mock_urlopen:
        res = await search_tool.execute({"query": "tavily search"}, tool_context)

    assert res.is_error is False
    assert "1. Tavily Search Result" in res.observation
    assert "URL: https://tavily.com/result" in res.observation
    assert "Snippet: Tavily extracted snippet text." in res.observation
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "https://api.tavily.com/search"
    sent_data = json.loads(req.data.decode("utf-8"))
    assert sent_data["query"] == "tavily search"
    assert sent_data["api_key"] == "tvly-test-key"


@pytest.mark.asyncio
async def test_web_search_brave_provider(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")

    brave_json = json.dumps({
        "web": {
            "results": [
                {
                    "title": "Brave Search Result",
                    "url": "https://search.brave.com/result",
                    "description": "Brave extracted description snippet."
                }
            ]
        }
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_mock_response(brave_json)) as mock_urlopen:
        res = await search_tool.execute({"query": "brave search"}, tool_context)

    assert res.is_error is False
    assert "1. Brave Search Result" in res.observation
    assert "URL: https://search.brave.com/result" in res.observation
    assert "Snippet: Brave extracted description snippet." in res.observation
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert "https://api.search.brave.com/res/v1/web/search" in req.full_url
    assert req.headers.get("X-subscription-token") == "brave-test-key" or req.headers.get("X-Subscription-Token") == "brave-test-key"


@pytest.mark.asyncio
async def test_web_search_exa_provider(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    exa_json = json.dumps({
        "results": [
            {
                "title": "Exa Search Result",
                "url": "https://exa.ai/result",
                "text": "Exa extracted text snippet."
            }
        ]
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_mock_response(exa_json)) as mock_urlopen:
        res = await search_tool.execute({"query": "exa search"}, tool_context)

    assert res.is_error is False
    assert "1. Exa Search Result" in res.observation
    assert "URL: https://exa.ai/result" in res.observation
    assert "Snippet: Exa extracted text snippet." in res.observation
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "https://api.exa.ai/search"
    assert req.headers.get("X-api-key") == "exa-test-key" or req.headers.get("x-api-key") == "exa-test-key"


def test_web_search_provider_resolution(search_tool: WebSearchTool, monkeypatch: pytest.MonkeyPatch):
    # 1. VERBIX_SEARCH_PROVIDER overrides all keys
    monkeypatch.setenv("VERBIX_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-key")
    provider, key = search_tool._resolve_provider()
    assert provider == "brave"
    assert key == "brave-key"

    # 2. EXA_API_KEY takes precedence when VERBIX_SEARCH_PROVIDER is not set
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    provider, key = search_tool._resolve_provider()
    assert provider == "exa"
    assert key == "exa-key"

    # 3. TAVILY_API_KEY takes precedence over BRAVE_API_KEY
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    provider, key = search_tool._resolve_provider()
    assert provider == "tavily"
    assert key == "tavily-key"

    # 4. BRAVE_API_KEY fallback
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider, key = search_tool._resolve_provider()
    assert provider == "brave"
    assert key == "brave-key"

    # 5. Default fallback to duckduckgo
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    provider, key = search_tool._resolve_provider()
    assert provider == "duckduckgo"
    assert key is None


@pytest.mark.asyncio
async def test_web_search_empty_query_error(search_tool: WebSearchTool, tool_context: ToolContext):
    res_empty = await search_tool.execute({"query": ""}, tool_context)
    assert res_empty.is_error is True
    assert "query" in res_empty.observation

    res_whitespace = await search_tool.execute({"query": "   \n\t "}, tool_context)
    assert res_whitespace.is_error is True

    res_missing = await search_tool.execute({}, tool_context)
    assert res_missing.is_error is True


@pytest.mark.asyncio
async def test_web_search_http_error_handled(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    http_error = urllib.error.HTTPError(
        url="https://lite.duckduckgo.com/lite/",
        code=500,
        msg="Internal Server Error",
        hdrs=Message(),
        fp=io.BytesIO(b"Server Error"),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        res = await search_tool.execute({"query": "python"}, tool_context)

    assert res.is_error is True
    assert "500" in res.observation or "Internal Server Error" in res.observation


@pytest.mark.asyncio
async def test_web_search_max_results_clamped(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    html_payload = b"""<!DOCTYPE html>
<html>
<body>
<table>
  <tr><td><a class="result-link" href="https://example.com/1">R1</a></td></tr>
  <tr><td class="result-snippet">S1</td></tr>
  <tr><td><a class="result-link" href="https://example.com/2">R2</a></td></tr>
  <tr><td class="result-snippet">S2</td></tr>
  <tr><td><a class="result-link" href="https://example.com/3">R3</a></td></tr>
  <tr><td class="result-snippet">S3</td></tr>
</table>
</body>
</html>"""

    with patch("urllib.request.urlopen", return_value=_mock_response(html_payload)):
        res = await search_tool.execute({"query": "test", "max_results": 2}, tool_context)

    assert res.is_error is False
    assert "1. R1" in res.observation
    assert "2. R2" in res.observation
    assert "3. R3" not in res.observation


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [403, 429])
async def test_web_search_duckduckgo_rate_limited_tip(
    search_tool: WebSearchTool,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    http_error = urllib.error.HTTPError(
        url="https://lite.duckduckgo.com/lite/",
        code=status_code,
        msg="Too Many Requests" if status_code == 429 else "Forbidden",
        hdrs=Message(),
        fp=io.BytesIO(b"Blocked"),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        res = await search_tool.execute({"query": "python"}, tool_context)

    assert res.is_error is True
    assert f"HTTP {status_code}" in res.observation
    assert "TAVILY_API_KEY" in res.observation
    assert "BRAVE_API_KEY" in res.observation
    assert "EXA_API_KEY" in res.observation


@pytest.mark.asyncio
async def test_web_search_timeout_error_handling(
    search_tool: WebSearchTool,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    url_err = urllib.error.URLError(TimeoutError("connection timed out"))
    with patch("urllib.request.urlopen", side_effect=url_err):
        res = await search_tool.execute({"query": "test query"}, tool_context)

    assert res.is_error is True
    assert res.observation == "error: search request timed out for query 'test query'"


def test_ddg_parser_structural_markers():
    parser = DuckDuckGoLiteParser()
    assert parser.has_search_form is False
    assert parser.has_zero_results_marker is False

    valid_html = """
    <form action="/lite/" method="post">
      <input type="text" name="q" value="foo">
    </form>
    <table><tr><td>No results.</td></tr></table>
    """
    parser.feed(valid_html)
    assert parser.has_search_form is True
    assert parser.has_zero_results_marker is True


@pytest.mark.asyncio
async def test_ddg_bot_challenge_returns_error(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    html_payload = b"""<!DOCTYPE html>
<html>
<body>
<div id="challenge-form" class="anomaly-modal">
  <h2>Bot Detection Challenge</h2>
  <p>Please complete security check</p>
</div>
</body>
</html>"""

    with patch("urllib.request.urlopen", return_value=_mock_response(html_payload)):
        res = await search_tool.execute({"query": "python"}, tool_context)

    assert res.is_error is True
    assert "blocked" in res.observation.lower() or "challenge" in res.observation.lower()
    assert "BRAVE_API_KEY" in res.observation


@pytest.mark.asyncio
async def test_ddg_unrecognized_response_returns_error(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    html_payload = b"""<!DOCTYPE html>
<html>
<body>
<div>Unexpected error occurred.</div>
</body>
</html>"""

    with patch("urllib.request.urlopen", return_value=_mock_response(html_payload)):
        res = await search_tool.execute({"query": "python"}, tool_context)

    assert res.is_error is True
    assert "unrecognized or bot-challenged" in res.observation.lower()
    assert "BRAVE_API_KEY" in res.observation


@pytest.mark.asyncio
async def test_ddg_genuine_zero_results_returns_no_results(
    search_tool: WebSearchTool, tool_context: ToolContext, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VERBIX_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    html_payload = b"""<!DOCTYPE html>
<html>
<body>
<form action="/lite/" method="post">
  <input type="text" name="q" value="weird query 12345" />
</form>
<table>
  <tr><td>No results.</td></tr>
</table>
</body>
</html>"""

    with patch("urllib.request.urlopen", return_value=_mock_response(html_payload)):
        res = await search_tool.execute({"query": "weird query 12345"}, tool_context)

    assert res.is_error is False
    assert res.observation == "No results found for query: 'weird query 12345'."

