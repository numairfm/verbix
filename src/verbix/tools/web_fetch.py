from __future__ import annotations

import asyncio
import html.parser
import ipaddress
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except (ImportError, OSError):
    _SSL_CONTEXT = ssl.create_default_context()

from verbix.tools.base import ToolContext, ToolResult

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HONEST_USER_AGENT = "verbix"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5MB limit
TIMEOUT_SECONDS = 30

NOISE_TAGS = {
    "aside",
    "footer",
    "form",
    "header",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
}

NOISE_ROLES = {
    "banner",
    "contentinfo",
    "navigation",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def _validate_url(url: str) -> str | None:
    if not url:
        return "error: url must not be empty"

    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as e:
        return f"error: invalid url '{url}': {e}"

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return f"error: unsupported scheme '{scheme}'; only http and https are allowed"

    hostname = parsed.hostname
    if not hostname:
        return f"error: url '{url}' is missing a valid hostname"

    host = hostname.lower().strip("[]")
    if host in ("localhost", "0.0.0.0") or host.endswith(".localhost"):
        return f"error: blocked host '{hostname}'"

    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            ip = None

    if ip is not None and (
        ip.is_private
        or ip.is_loopback
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_link_local
    ):
        return f"error: blocked private/loopback host '{hostname}'"

    return None


def _is_cf_challenge(error: urllib.error.HTTPError) -> bool:
    if error.code != 403 or not error.headers:
        return False
    for k, v in error.headers.items():
        if k.lower() == "cf-mitigated" and str(v).lower() == "challenge":
            return True
    return False


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Rejects HTTP redirects to blocked or private/loopback hostnames."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validation_error = _validate_url(newurl)
        if validation_error:
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"redirect blocked: {validation_error}",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
        _SafeRedirectHandler(),
    )


def _fetch_html(
    url: str, opener: urllib.request.OpenerDirector | None = None
) -> tuple[str, str]:
    active_opener = opener or _build_opener()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with active_opener.open(req, timeout=TIMEOUT_SECONDS) as resp:
            content_bytes = resp.read(MAX_RESPONSE_BYTES)
            content_type = resp.headers.get("Content-Type", "") if resp.headers else ""
    except urllib.error.HTTPError as e:
        if _is_cf_challenge(e):
            retry_req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": HONEST_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                method="GET",
            )
            with active_opener.open(retry_req, timeout=TIMEOUT_SECONDS) as resp:
                content_bytes = resp.read(MAX_RESPONSE_BYTES)
                content_type = resp.headers.get("Content-Type", "") if resp.headers else ""
        else:
            raise

    charset = "utf-8"
    if "charset=" in content_type.lower():
        charset = (
            content_type.lower()
            .split("charset=")[-1]
            .split(";")[0]
            .strip()
            .strip("\"'")
        )
    try:
        return content_bytes.decode(charset, errors="replace"), content_type
    except (LookupError, UnicodeDecodeError):
        return content_bytes.decode("utf-8", errors="replace"), content_type


class HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self, mode: str = "markdown") -> None:
        super().__init__()
        self.mode = mode
        self.title = ""
        self._title_parts: list[str] = []
        self._in_title = False
        self._in_pre = False
        self._ignored_stack: list[str] = []
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return

        attr_dict = {k.lower(): (v.lower() if v else "") for k, v in attrs}
        role = attr_dict.get("role", "")
        is_noise = tag in NOISE_TAGS or role in NOISE_ROLES

        if self._ignored_stack or is_noise:
            if tag not in VOID_TAGS:
                self._ignored_stack.append(tag)
            return

        if tag == "pre":
            self._chunks.append("\n\n\x00PRE_START\x00")
            self._in_pre = True
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._chunks.append("\n\n")
            if self.mode == "markdown":
                level = int(tag[1])
                self._chunks.append("#" * level + " ")
        elif tag == "li":
            self._chunks.append("\n")
            if self.mode == "markdown":
                self._chunks.append("- ")
        elif tag in ("p", "article", "section"):
            self._chunks.append("\n\n")
        elif tag in ("div", "tr", "br"):
            self._chunks.append("\n")
        elif tag == "hr":
            self._chunks.append("\n\n---\n\n" if self.mode == "markdown" else "\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            self.title = " ".join("".join(self._title_parts).split())
            return

        if self._ignored_stack:
            if tag in self._ignored_stack:
                while self._ignored_stack and self._ignored_stack[-1] != tag:
                    self._ignored_stack.pop()
                if self._ignored_stack:
                    self._ignored_stack.pop()
            return

        if tag == "pre":
            self._in_pre = False
            self._chunks.append("\x00PRE_END\x00\n\n")
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "article", "section", "ul", "ol"):
            self._chunks.append("\n\n")
        elif tag in ("div", "tr"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._ignored_stack:
            return
        if not data:
            return
        if self._in_pre:
            self._chunks.append(data)
            return
        normalized = re.sub(r"[ \t\r\n]+", " ", data)
        self._chunks.append(normalized)

    def close(self) -> None:
        super().close()
        if self._in_pre:
            self._in_pre = False
            self._chunks.append("\x00PRE_END\x00\n\n")
        if self._title_parts and not self.title:
            self.title = " ".join("".join(self._title_parts).split())

    def get_text(self, url: str) -> str:
        raw_text = "".join(self._chunks)
        blocks: list[str] = []
        for part in re.split(r"(\x00PRE_START\x00[\s\S]*?\x00PRE_END\x00)", raw_text):
            if not part:
                continue
            if part.startswith("\x00PRE_START\x00") and part.endswith("\x00PRE_END\x00"):
                pre_content = part[len("\x00PRE_START\x00") : -len("\x00PRE_END\x00")].strip("\r\n")
                if pre_content:
                    blocks.append(pre_content)
            else:
                lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in part.splitlines()]
                cleaned_lines: list[str] = []
                blank = False
                for line in lines:
                    if line:
                        cleaned_lines.append(line)
                        blank = False
                    elif not blank:
                        cleaned_lines.append("")
                        blank = True
                normal_text = "\n".join(cleaned_lines).strip()
                if normal_text:
                    blocks.append(normal_text)

        body = "\n\n".join(blocks).strip()
        title_header = f"Title: {self.title}\n" if self.title else ""
        return f"{title_header}URL: {url}\n\n{body}"


class WebFetchTool:
    name: str = "web_fetch"
    side_effecting: bool = False
    description: str = (
        "Fetches a web page URL and extracts readable content (in markdown, plain text, or raw HTML), "
        "stripping navigation, ads, and noise."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to fetch",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: 'markdown' (default clean markdown), 'text' (plain text), or 'html' (raw HTML)",
                    "enum": ["markdown", "text", "html"],
                    "default": "markdown",
                },
            },
            "required": ["url"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        url = str(args.get("url") or "").strip()
        validation_error = _validate_url(url)
        if validation_error:
            return ToolResult(validation_error, is_error=True)

        fmt = str(args.get("format", "markdown")).strip().lower()
        if fmt not in ("markdown", "text", "html"):
            return ToolResult(
                f"error: unsupported format '{fmt}'; must be 'markdown', 'text', or 'html'",
                is_error=True,
            )

        return await asyncio.to_thread(self._fetch_and_process, url, fmt)

    def _fetch_and_process(self, url: str, fmt: str) -> ToolResult:
        try:
            content_text, content_type = _fetch_html(url)
        except urllib.error.HTTPError as e:
            return ToolResult(
                f"error: failed to fetch '{url}': {e.code} {e.reason}",
                is_error=True,
            )
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                return ToolResult(
                    f"error: request timed out fetching '{url}'",
                    is_error=True,
                )
            return ToolResult(
                f"error: network error fetching '{url}': {e.reason}",
                is_error=True,
            )
        except TimeoutError:
            return ToolResult(
                f"error: request timed out fetching '{url}'",
                is_error=True,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"error: failed to fetch '{url}': {e}", is_error=True)

        if fmt == "html":
            return ToolResult(content_text)

        ct = content_type.lower()
        if any(t in ct for t in ("application/json", "text/plain", "text/markdown", "+json")):
            return ToolResult(content_text)

        extractor = HTMLTextExtractor(mode=fmt)
        extractor.feed(content_text)
        extractor.close()
        return ToolResult(extractor.get_text(url))
