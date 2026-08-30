from dataclasses import replace

import pytest
import requests

from toxiscan_core.config import WEB_LIMITS
from toxiscan_core.web import (
    RobotsDeniedError,
    UnsafeUrlError,
    WebExtractionError,
    _render_page,
    extract_blocks_from_html,
    scrape_url,
    validate_public_url,
)


def public_resolver(hostname: str, port: int):
    return [(2, 1, 6, "", ("93.184.216.34", port))]


class StubResponse:
    def __init__(self, body=b"", status=200, headers=None) -> None:
        self.body = body
        self.status_code = status
        self.headers = headers or {}
        self.encoding = "utf-8"
        self.closed = False

    def iter_content(self, chunk_size: int):
        yield self.body

    def close(self) -> None:
        self.closed = True


class StubSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StubDriver:
    def __init__(self, html: str, final_url="https://example.com/page") -> None:
        self.page_source = html
        self.current_url = final_url
        self.quit_called = False
        self.heights = iter((100, 100, 100, 100))

    def set_page_load_timeout(self, seconds):
        self.page_load_timeout = seconds

    def set_script_timeout(self, seconds):
        self.script_timeout = seconds

    def get(self, url):
        self.requested_url = url

    def find_elements(self, by, xpath):
        return []

    def execute_script(self, script, *args):
        if script.startswith("return"):
            return next(self.heights, 100)
        return None

    def quit(self):
        self.quit_called = True


def test_validate_public_url_adds_https_and_removes_fragment():
    result = validate_public_url("example.com/path#section", public_resolver)

    assert result == "https://example.com/path"


@pytest.mark.parametrize(
    "url",
    (
        "file:///etc/passwd",
        "https://user:password@example.com",
        "https://example.com:invalid",
    ),
)
def test_validate_public_url_rejects_unsafe_syntax(url):
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url, public_resolver)


def test_validate_public_url_rejects_private_dns_result():
    def private_resolver(hostname: str, port: int):
        return [(2, 1, 6, "", ("127.0.0.1", port))]

    with pytest.raises(UnsafeUrlError, match="private"):
        validate_public_url("https://example.com", private_resolver)


def test_extract_blocks_from_html_finds_comments_and_removes_duplicates():
    html = """
        <html><head><title>Discussion</title></head><body>
          <nav>Navigation</nav>
          <main><p>Article body</p></main>
          <div class="comment">First comment</div>
          <div data-testid="comment-card">Second comment</div>
          <p>Article body</p>
          <script>hidden()</script>
        </body></html>
    """

    result = extract_blocks_from_html(html)

    assert result.blocks == (
        "Discussion",
        "First comment",
        "Second comment",
        "Article body",
    )


def test_scrape_url_obeys_robots_denial():
    session = StubSession(
        StubResponse(
            b"User-agent: ToxiScan\nDisallow: /private",
            headers={"Content-Type": "text/plain"},
        )
    )

    with pytest.raises(RobotsDeniedError):
        scrape_url(
            "https://example.com/private",
            session=session,
            resolver=public_resolver,
            webdriver_factory=lambda: pytest.fail("browser should not start"),
        )


def test_scrape_url_revalidates_redirect_destination():
    session = StubSession(
        StubResponse(status=404),
        StubResponse(status=302, headers={"Location": "http://internal.example/secret"}),
    )

    def resolver(hostname: str, port: int):
        address = "10.0.0.5" if hostname == "internal.example" else "93.184.216.34"
        return [(2, 1, 6, "", (address, port))]

    with pytest.raises(UnsafeUrlError):
        scrape_url("https://example.com", session=session, resolver=resolver)


def test_scrape_url_uses_static_content_when_browser_fails():
    session = StubSession(
        StubResponse(status=404),
        StubResponse(
            b"<html><body><p>Readable initial content</p></body></html>",
            headers={"Content-Type": "text/html"},
        ),
    )

    result = scrape_url(
        "https://example.com",
        session=session,
        resolver=public_resolver,
        webdriver_factory=lambda: (_ for _ in ()).throw(RuntimeError("no browser")),
    )

    assert result.text.blocks == ("Readable initial content",)
    assert result.used_browser is False
    assert "initial content" in result.warnings[0]


def test_scrape_url_uses_rendered_comments_and_closes_browser():
    session = StubSession(
        StubResponse(status=404),
        StubResponse(
            b"<html><body><p>Initial content</p></body></html>",
            headers={"Content-Type": "text/html"},
        ),
    )
    driver = StubDriver("<html><body><div class='comment'>Loaded comment</div></body></html>")
    limits = replace(WEB_LIMITS, scroll_pause_seconds=0)

    result = scrape_url(
        "https://example.com/page",
        limits,
        session=session,
        resolver=public_resolver,
        webdriver_factory=lambda: driver,
    )

    assert result.text.blocks == ("Loaded comment",)
    assert result.used_browser is True
    assert driver.quit_called is True


def test_scrape_url_rejects_oversized_response_from_header():
    limits = replace(WEB_LIMITS, max_response_bytes=10)
    session = StubSession(
        StubResponse(status=404),
        StubResponse(
            b"small",
            headers={"Content-Length": "11", "Content-Type": "text/html"},
        ),
    )

    with pytest.raises(WebExtractionError, match="too large"):
        scrape_url(
            "https://example.com",
            limits,
            session=session,
            resolver=public_resolver,
        )


def test_render_page_closes_browser_when_navigation_fails():
    class FailingDriver(StubDriver):
        def get(self, url):
            raise RuntimeError("navigation failed")

    driver = FailingDriver("")

    with pytest.raises(RuntimeError, match="navigation failed"):
        _render_page(
            "https://example.com",
            WEB_LIMITS,
            public_resolver,
            lambda: driver,
        )

    assert driver.quit_called is True


def test_scrape_url_denies_when_robots_file_is_unreachable():
    session = StubSession(requests.ConnectionError("offline"))

    with pytest.raises(RobotsDeniedError):
        scrape_url(
            "https://example.com",
            session=session,
            resolver=public_resolver,
        )
