"""Bounded extraction of readable content and comments from public web pages."""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .config import WEB_LIMITS, WebLimits
from .text import ExtractedText, prepare_blocks

logger = logging.getLogger(__name__)

USER_AGENT_TOKEN = "ToxiScan"
USER_AGENT = "ToxiScan/1.0 (+https://github.com/Daveokw/ToxiScan)"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_LOAD_MORE_PHRASES = (
    "load more comments",
    "show more comments",
    "view more comments",
    "more comments",
    "show previous comments",
)


class WebExtractionError(ValueError):
    """Raised when a page cannot be fetched and extracted safely."""


class UnsafeUrlError(WebExtractionError):
    """Raised when a URL could reach a non-public network target."""


class RobotsDeniedError(WebExtractionError):
    """Raised when the site's published crawler rules deny access."""


class ResponseLike(Protocol):
    status_code: int
    headers: dict[str, str]
    encoding: str | None

    def iter_content(self, chunk_size: int): ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ScrapeResult:
    """Extracted page content and non-fatal scrape information."""

    text: ExtractedText
    final_url: str
    used_browser: bool
    warnings: tuple[str, ...] = ()


def _default_resolver(hostname: str, port: int):
    return socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)


def validate_public_url(
    value: str,
    resolver: Callable[[str, int], list[tuple]] = _default_resolver,
) -> str:
    """Normalise an HTTP URL and reject credentials or non-public addresses."""

    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlsplit(candidate)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise UnsafeUrlError("Only public HTTP and HTTPS pages can be analysed.")
    if not parsed.hostname:
        raise UnsafeUrlError("Enter a complete public web address.")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("Web addresses containing credentials are not accepted.")

    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost":
        raise UnsafeUrlError("Local and private network addresses are not accepted.")

    try:
        supplied_port = parsed.port
    except ValueError as error:
        raise UnsafeUrlError("The web address contains an invalid port.") from error
    port = supplied_port or (443 if parsed.scheme.casefold() == "https" else 80)
    try:
        resolved = resolver(hostname, port)
    except (OSError, socket.gaierror) as error:
        raise WebExtractionError("The web address could not be resolved.") from error

    if not resolved:
        raise WebExtractionError("The web address did not resolve to a public server.")

    for entry in resolved:
        address = ipaddress.ip_address(entry[4][0])
        if not address.is_global:
            raise UnsafeUrlError("Local and private network addresses are not accepted.")

    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if supplied_port is not None:
        netloc = f"{netloc}:{supplied_port}"
    return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path or "/", parsed.query, ""))


def _new_session() -> requests.Session:
    session = requests.Session()
    # Hosted requests must not inherit proxy credentials or a developer's .netrc.
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"})
    return session


def _read_response(response: ResponseLike, limit: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = 0
        if declared_length > limit:
            raise WebExtractionError("The page is too large to process safely.")

    body = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        body.extend(chunk)
        if len(body) > limit:
            raise WebExtractionError("The page is too large to process safely.")
    return bytes(body)


def _request_bytes(
    url: str,
    session: requests.Session,
    limits: WebLimits,
    resolver: Callable[[str, int], list[tuple]],
) -> tuple[bytes, str, str, int]:
    current_url = validate_public_url(url, resolver)

    for redirect_count in range(limits.max_redirects + 1):
        response = None
        try:
            response = session.get(
                current_url,
                allow_redirects=False,
                stream=True,
                timeout=(limits.connect_timeout_seconds, limits.read_timeout_seconds),
            )
            status = response.status_code
            if status in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location:
                    raise WebExtractionError("The page returned an invalid redirect.")
                if redirect_count >= limits.max_redirects:
                    raise WebExtractionError("The page redirected too many times.")
                current_url = validate_public_url(urljoin(current_url, location), resolver)
                continue

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
            body = _read_response(response, limits.max_response_bytes)
            return body, current_url, content_type, status
        except requests.RequestException as error:
            raise WebExtractionError("The page could not be downloaded.") from error
        finally:
            if response is not None:
                response.close()

    raise WebExtractionError("The page redirected too many times.")


def _robots_allowed(
    url: str,
    session: requests.Session,
    limits: WebLimits,
    resolver: Callable[[str, int], list[tuple]],
) -> bool:
    parsed = urlsplit(url)
    robots_url = urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))

    try:
        body, final_url, content_type, status = _request_bytes(
            robots_url, session, limits, resolver
        )
    except WebExtractionError:
        # RFC 9309 treats an unreachable robots file conservatively. A transient
        # failure must not become permission to crawl the site.
        return False

    if status in {401, 403, 429} or status >= 500:
        return False
    if status >= 400:
        return True
    if content_type and content_type != "text/plain":
        return True

    parser = RobotFileParser()
    parser.set_url(final_url)
    parser.parse(body.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(USER_AGENT_TOKEN, url)


def extract_blocks_from_html(
    html: str | bytes, limits: WebLimits = WEB_LIMITS
) -> ExtractedText:
    """Extract visible article, list, and comment-like blocks from HTML."""

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(
        ["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form"]
    ):
        element.decompose()

    candidates = []
    title = soup.find("title")
    if title:
        candidates.append(title.get_text(" ", strip=True))

    selectors = (
        "article",
        '[role="article"]',
        '[class*="comment" i]',
        '[id*="comment" i]',
        '[data-testid*="comment" i]',
        "main p",
        "main li",
        "blockquote",
        "p",
        "li",
    )
    for selector in selectors:
        for element in soup.select(selector):
            candidates.append(element.get_text(" ", strip=True))

    if not candidates:
        candidates.append(soup.get_text("\n", strip=True))
    return prepare_blocks(candidates, limits.text)


def _create_webdriver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument(f"--user-agent={USER_AGENT}")
    options.set_capability("acceptInsecureCerts", False)

    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if os.path.exists("/usr/bin/chromium"):
        chromium = "/usr/bin/chromium"
    if chromium:
        options.binary_location = chromium

    driver_path = shutil.which("chromedriver")
    service = Service(executable_path=driver_path) if driver_path else Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Network.setBlockedURLs",
        {
            "urls": [
                "file://*",
                "http://127.*",
                "https://127.*",
                "http://10.*",
                "https://10.*",
                "http://169.254.*",
                "https://169.254.*",
                "http://localhost/*",
                "https://localhost/*",
            ]
        },
    )
    driver.execute_cdp_cmd("Network.enable", {})
    return driver


def _click_more_comments(driver, remaining: int) -> int:
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.common.by import By

    clicked = 0
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lowercase = uppercase.casefold()
    for phrase in _LOAD_MORE_PHRASES:
        xpath = (
            "//*[self::button or @role='button'][contains("
            f"translate(normalize-space(.), '{uppercase}', '{lowercase}'), '{phrase}')]"
        )
        for element in driver.find_elements(By.XPATH, xpath):
            if clicked >= remaining:
                return clicked
            try:
                if element.is_displayed() and element.is_enabled():
                    driver.execute_script("arguments[0].click();", element)
                    clicked += 1
            except WebDriverException as error:
                logger.debug(
                    "A comment expansion control could not be activated: %s",
                    type(error).__name__,
                )
                continue
    return clicked


def _render_page(
    url: str,
    limits: WebLimits,
    resolver: Callable[[str, int], list[tuple]],
    webdriver_factory: Callable[[], object],
) -> tuple[str, str]:
    driver = webdriver_factory()
    started = time.monotonic()
    try:
        driver.set_page_load_timeout(limits.page_load_timeout_seconds)
        driver.set_script_timeout(limits.page_load_timeout_seconds)
        driver.get(url)
        final_url = validate_public_url(driver.current_url, resolver)

        previous_height = -1
        stable_scrolls = 0
        load_more_clicks = 0

        for _ in range(limits.max_scrolls):
            if time.monotonic() - started >= limits.scrape_timeout_seconds:
                break
            if load_more_clicks < limits.max_load_more_clicks:
                load_more_clicks += _click_more_comments(
                    driver, limits.max_load_more_clicks - load_more_clicks
                )

            height = int(driver.execute_script("return document.body.scrollHeight") or 0)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(limits.scroll_pause_seconds)

            if height == previous_height:
                stable_scrolls += 1
                if stable_scrolls >= limits.stable_scrolls_before_stop:
                    break
            else:
                stable_scrolls = 0
                previous_height = height

        page_source = driver.page_source
        if len(page_source.encode("utf-8")) > limits.max_response_bytes:
            page_source = page_source.encode("utf-8")[: limits.max_response_bytes].decode(
                "utf-8", errors="ignore"
            )
        return page_source, final_url
    finally:
        driver.quit()


def scrape_url(
    url: str,
    limits: WebLimits = WEB_LIMITS,
    *,
    session: requests.Session | None = None,
    resolver: Callable[[str, int], list[tuple]] = _default_resolver,
    webdriver_factory: Callable[[], object] = _create_webdriver,
) -> ScrapeResult:
    """Extract a public page with dynamic rendering and a static fallback."""

    safe_url = validate_public_url(url, resolver)
    client = session or _new_session()
    if not _robots_allowed(safe_url, client, limits, resolver):
        raise RobotsDeniedError("The site's crawler rules do not permit this page.")

    body, final_url, content_type, status = _request_bytes(
        safe_url, client, limits, resolver
    )
    if status >= 400:
        raise WebExtractionError(f"The page returned HTTP status {status}.")
    if content_type and not content_type.startswith(_ALLOWED_CONTENT_TYPES):
        raise WebExtractionError("The web address did not return a readable page.")

    warnings: list[str] = []

    try:
        rendered_html, final_url = _render_page(
            final_url, limits, resolver, webdriver_factory
        )
        extracted = extract_blocks_from_html(rendered_html, limits)
        used_browser = True
    except Exception:
        logger.exception(
            "Dynamic page rendering failed; using static content",
            extra={"page_host": urlsplit(final_url).hostname},
        )
        extracted = extract_blocks_from_html(body, limits)
        used_browser = False
        warnings.append(
            "Dynamic content could not be rendered, so only the page's initial content was analysed."
        )

    if not extracted.blocks:
        raise WebExtractionError(
            "No readable public content was found. The page may require authentication."
        )
    if extracted.truncated:
        warnings.append("The page reached the safe analysis limit and was truncated.")
    return ScrapeResult(extracted, final_url, used_browser, tuple(warnings))
