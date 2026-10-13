import io
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from verbix.tools.base import ToolContext
from verbix.tools.web_fetch import HTMLTextExtractor, WebFetchTool, _SafeRedirectHandler


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


@pytest.fixture
def fetch_tool() -> WebFetchTool:
    return WebFetchTool()


def _mock_response(
    body_bytes: bytes,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body_bytes
    resp.status = status
    msg = Message()
    if headers:
        for k, v in headers.items():
            msg[k] = v
    resp.headers = msg
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


@pytest.mark.asyncio
async def test_web_fetch_extracts_markdown_by_default(
    fetch_tool: WebFetchTool, tool_context: ToolContext
):
    html_content = b"""<!DOCTYPE html>
<html>
<head><title>Documentation Guide</title></head>
<body>
  <h1>Getting Started</h1>
  <p>Welcome to the guide. Here is an introductory paragraph.</p>
  <h2>Features</h2>
  <ul>
    <li>Fast parsing</li>
    <li>Zero dependencies</li>
  </ul>
</body>
</html>"""

    with patch.object(
        urllib.request.OpenerDirector, "open", return_value=_mock_response(html_content)
    ):
        res = await fetch_tool.execute({"url": "https://example.com/docs"}, tool_context)

    assert not res.is_error
    assert "Title: Documentation Guide" in res.observation
    assert "URL: https://example.com/docs" in res.observation
    assert "# Getting Started" in res.observation
    assert "Welcome to the guide. Here is an introductory paragraph." in res.observation
    assert "## Features" in res.observation
    # Tighter list formatting: adjacent items without empty lines in between
    assert "- Fast parsing\n- Zero dependencies" in res.observation


@pytest.mark.asyncio
async def test_web_fetch_formats(fetch_tool: WebFetchTool, tool_context: ToolContext):
    html_content = b"""<!DOCTYPE html>
<html>
<head><title>Format Test</title></head>
<body>
  <h1>Header One</h1>
  <p>Paragraph content.</p>
  <ul>
    <li>Bullet item</li>
  </ul>
</body>
</html>"""

    # Test format="text"
    with patch.object(
        urllib.request.OpenerDirector, "open", return_value=_mock_response(html_content)
    ):
        res_text = await fetch_tool.execute(
            {"url": "https://example.com/page", "format": "text"},
            tool_context,
        )

    assert not res_text.is_error
    assert "Title: Format Test" in res_text.observation
    assert "URL: https://example.com/page" in res_text.observation
    assert "Header One" in res_text.observation
    assert "# Header One" not in res_text.observation
    assert "- Bullet item" not in res_text.observation
    assert "Bullet item" in res_text.observation

    # Test format="html"
    with patch.object(
        urllib.request.OpenerDirector, "open", return_value=_mock_response(html_content)
    ):
        res_html = await fetch_tool.execute(
            {"url": "https://example.com/page", "format": "html"},
            tool_context,
        )

    assert not res_html.is_error
    assert res_html.observation == html_content.decode("utf-8")


@pytest.mark.asyncio
async def test_web_fetch_cloudflare_challenge_retries_with_honest_ua(
    fetch_tool: WebFetchTool, tool_context: ToolContext
):
    cf_headers = Message()
    cf_headers["cf-mitigated"] = "challenge"
    cf_error = urllib.error.HTTPError(
        url="https://example.com/protected",
        code=403,
        msg="Forbidden",
        hdrs=cf_headers,
        fp=None,
    )

    success_resp = _mock_response(
        b"<html><head><title>Passed</title></head><body><p>Access Granted</p></body></html>"
    )

    calls = []

    def mock_open(self, req, *args, **kwargs):
        calls.append(req)
        if len(calls) == 1:
            raise cf_error
        return success_resp

    with patch.object(urllib.request.OpenerDirector, "open", mock_open):
        res = await fetch_tool.execute({"url": "https://example.com/protected"}, tool_context)

    assert not res.is_error
    assert "Access Granted" in res.observation
    assert len(calls) == 2

    # First request: Chrome desktop UA
    first_ua = calls[0].headers.get("User-agent") or calls[0].headers.get("User-Agent")
    assert "Mozilla" in first_ua

    # Second request: honest "verbix" UA
    second_ua = calls[1].headers.get("User-agent") or calls[1].headers.get("User-Agent")
    assert second_ua == "verbix"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssrf_url",
    [
        "file:///etc/passwd",
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://0.0.0.0/",
        "http://192.168.1.100/admin",
        "http://10.0.0.1/",
    ],
)
async def test_web_fetch_blocks_ssrf(
    fetch_tool: WebFetchTool, tool_context: ToolContext, ssrf_url: str
):
    with patch.object(urllib.request.OpenerDirector, "open") as mock_open:
        res = await fetch_tool.execute({"url": ssrf_url}, tool_context)

    assert res.is_error
    assert "error" in res.observation.lower()
    mock_open.assert_not_called()


def test_safe_redirect_handler_blocks_private_ip():
    handler = _SafeRedirectHandler()
    req = urllib.request.Request("https://example.com/initial")
    headers = Message()
    headers["Location"] = "http://127.0.0.1/admin"

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        handler.http_error_302(req, None, 302, "Found", headers)

    assert exc_info.value.code == 302
    assert "redirect blocked" in exc_info.value.reason


@pytest.mark.asyncio
async def test_web_fetch_blocks_ssrf_on_redirect(
    fetch_tool: WebFetchTool, tool_context: ToolContext
):
    class MockRedirectHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            resp = urllib.request.addinfourl(io.BytesIO(b""), Message(), req.full_url, 302)
            resp.msg = "Found"
            resp.headers["Location"] = "http://127.0.0.1/sensitive"
            return self.parent.error("http", req, resp, 302, "Found", resp.headers)

    mock_opener = urllib.request.build_opener(MockRedirectHandler, _SafeRedirectHandler)

    with patch("verbix.tools.web_fetch._build_opener", return_value=mock_opener):
        res = await fetch_tool.execute({"url": "http://example.com/redirect"}, tool_context)

    assert res.is_error
    assert "redirect blocked" in res.observation
    assert "127.0.0.1" in res.observation


@pytest.mark.asyncio
async def test_web_fetch_http_error_handled(
    fetch_tool: WebFetchTool, tool_context: ToolContext
):
    error = urllib.error.HTTPError(
        url="https://example.com/missing",
        code=404,
        msg="Not Found",
        hdrs=Message(),
        fp=None,
    )

    with patch.object(urllib.request.OpenerDirector, "open", side_effect=error):
        res = await fetch_tool.execute({"url": "https://example.com/missing"}, tool_context)

    assert res.is_error
    assert "error: failed to fetch 'https://example.com/missing': 404 Not Found" in res.observation


@pytest.mark.asyncio
async def test_web_fetch_timeout_handled(
    fetch_tool: WebFetchTool, tool_context: ToolContext
):
    with patch.object(
        urllib.request.OpenerDirector,
        "open",
        side_effect=TimeoutError("Connection timed out"),
    ):
        res = await fetch_tool.execute({"url": "https://example.com/slow"}, tool_context)

    assert res.is_error
    assert "timed out" in res.observation.lower()


@pytest.mark.asyncio
async def test_web_fetch_url_error_timeout_handled(
    fetch_tool: WebFetchTool, tool_context: ToolContext
):
    url_err = urllib.error.URLError(TimeoutError("operation timed out"))
    with patch.object(urllib.request.OpenerDirector, "open", side_effect=url_err):
        res = await fetch_tool.execute({"url": "https://example.com/slow_url"}, tool_context)

    assert res.is_error
    assert "timed out" in res.observation.lower()


def test_html_text_extractor_strips_noise_and_preserves_structure():
    html_doc = """<!DOCTYPE html>
<html>
<head><title>Noise Test</title></head>
<body>
  <header>Site Header to Ignore</header>
  <nav>Navigation links to Ignore</nav>
  <div role="navigation">Role navigation to Ignore</div>
  <aside>Sidebar to Ignore</aside>
  <article>
    <h1>Article Title</h1>
    <p>Good content paragraph with <img src="pic.jpg"> void tag inside.</p>
    <div role="banner">Ad Banner to Ignore</div>
    <form action="/search"><input type="text"><button>Search</button></form>
    <script>var x = 1;</script>
    <style>.css { color: red; }</style>
    <noscript>Please enable JS</noscript>
    <svg><circle r="5"/></svg>
  </article>
  <footer>Site Footer to Ignore</footer>
  <div role="contentinfo">Contentinfo to Ignore</div>
</body>
</html>"""

    extractor = HTMLTextExtractor(mode="markdown")
    extractor.feed(html_doc)
    extractor.close()
    text = extractor.get_text("https://example.com")

    assert "Title: Noise Test" in text
    assert "# Article Title" in text
    assert "Good content paragraph with void tag inside." in text
    assert "Site Header" not in text
    assert "Navigation links" not in text
    assert "Role navigation" not in text
    assert "Sidebar to Ignore" not in text
    assert "Ad Banner" not in text
    assert "Search" not in text
    assert "var x = 1" not in text
    assert "color: red" not in text
    assert "Please enable JS" not in text
    assert "Site Footer" not in text
    assert "Contentinfo" not in text


@pytest.mark.asyncio
async def test_web_fetch_invalid_args(fetch_tool: WebFetchTool, tool_context: ToolContext):
    res_empty = await fetch_tool.execute({"url": ""}, tool_context)
    assert res_empty.is_error

    res_format = await fetch_tool.execute(
        {"url": "https://example.com", "format": "unsupported"}, tool_context
    )
    assert res_format.is_error
