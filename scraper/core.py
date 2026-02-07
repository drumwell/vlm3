"""
Core scraper functionality: session management, rate limiting, checkpointing.

This module provides the foundational components for the forum scraper:
- ForumConfig: Multi-forum configuration with platform presets
- ScraperSession: HTTP session with rate limiting and authentication
- Checkpoint: Resume capability for long-running scrapes
- Data validation utilities
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

# Optional cloudscraper for Cloudflare bypass
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (parent of scraper/)
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Selectors:
    """CSS selectors for parsing forum pages."""

    # Forum index
    forum_container: str = ""
    forum_link: str = ""
    forum_description: str = ""

    # Forum page
    subforum_link: str = ""
    thread_container: str = ""
    thread_link: str = ""
    thread_author: str = ""
    thread_stats: str = ""
    thread_sticky_class: str = ""
    thread_locked_class: str = ""

    # Thread page
    post_container: str = ""
    post_author: str = ""
    post_content: str = ""
    post_timestamp: str = ""
    post_quote: str = ""

    # Pagination
    pagination_container: str = ""
    pagination_current: str = ""
    pagination_next: str = ""
    pagination_prev: str = ""
    pagination_last: str = ""


@dataclass
class AuthConfig:
    """Authentication configuration."""

    enabled: bool = False
    login_url: str = ""
    login_form_selector: str = ""
    username_field: str = ""
    password_field: str = ""
    csrf_field: str = ""
    success_indicator: str = ""
    username: str = ""  # Loaded from env
    password: str = ""  # Loaded from env


@dataclass
class ProxyConfig:
    """Proxy configuration for residential proxy services (e.g., Oxylabs)."""

    enabled: bool = False
    provider: str = "oxylabs"
    host: str = "pr.oxylabs.io"
    port: int = 7777
    username: str = ""  # Loaded from env
    password: str = ""  # Loaded from env
    country: str = ""   # e.g., "US"
    city: str = ""
    state: str = ""
    sticky_session: bool = False
    session_id: str = ""
    fallback_on_failure: bool = True
    max_proxy_retries: int = 3
    track_bandwidth: bool = True

    def build_proxy_url(self) -> Optional[str]:
        """
        Build Oxylabs proxy URL with targeting options.

        Returns:
            Proxy URL string or None if disabled/misconfigured
        """
        if not self.enabled or not self.username or not self.password:
            return None

        # Build username with targeting options
        parts = [f"customer-{self.username}"]

        if self.country:
            parts.append(f"cc-{self.country.upper()}")

        if self.city:
            parts.append(f"city-{self.city.lower().replace(' ', '_')}")

        if self.state:
            parts.append(f"st-{self.state.lower().replace(' ', '_')}")

        if self.sticky_session and self.session_id:
            parts.append(f"sessid-{self.session_id}")

        username = "-".join(parts)
        return f"http://{username}:{self.password}@{self.host}:{self.port}"


class ProxyManager:
    """
    Manages proxy sessions, rotation, and bandwidth tracking.

    Features:
    - Proxy URL generation with session control
    - IP rotation on failure
    - Bandwidth tracking for cost monitoring
    """

    def __init__(self, config: ProxyConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize proxy manager.

        Args:
            config: Proxy configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.request_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.bytes_downloaded = 0
        self.rotation_count = 0

        # Generate initial session ID if sticky sessions enabled but no ID set
        if self.config.sticky_session and not self.config.session_id:
            self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate a random session ID."""
        import uuid
        self.config.session_id = uuid.uuid4().hex[:16]
        return self.config.session_id

    def get_proxy_dict(self, force_sticky: bool = False) -> Optional[Dict[str, str]]:
        """
        Get proxy dictionary for requests.

        Args:
            force_sticky: Force sticky session even if not configured

        Returns:
            Dict with 'http' and 'https' proxy URLs, or None if disabled
        """
        if not self.config.enabled:
            return None

        # Handle forced sticky session
        original_sticky = self.config.sticky_session
        original_session_id = self.config.session_id

        if force_sticky and not self.config.sticky_session:
            self.config.sticky_session = True
            if not self.config.session_id:
                self._generate_session_id()

        proxy_url = self.config.build_proxy_url()

        # Restore original settings if we forced sticky
        if force_sticky and not original_sticky:
            self.config.sticky_session = original_sticky
            self.config.session_id = original_session_id

        if not proxy_url:
            return None

        return {
            "http": proxy_url,
            "https": proxy_url,
        }

    def rotate_session(self):
        """Rotate to a new proxy session (new IP)."""
        old_id = self.config.session_id
        self._generate_session_id()
        self.rotation_count += 1
        self.consecutive_errors = 0
        self.logger.info(f"Rotated proxy session: {old_id[:8]}... -> {self.config.session_id[:8]}...")

    def track_bandwidth(self, response) -> int:
        """
        Track bandwidth usage from response.

        Args:
            response: requests.Response object

        Returns:
            Bytes tracked
        """
        if not self.config.track_bandwidth:
            return 0

        # Try Content-Length header first
        content_length = response.headers.get("Content-Length")
        if content_length:
            bytes_count = int(content_length)
        else:
            # Fall back to actual content length
            bytes_count = len(response.content) if hasattr(response, 'content') else 0

        self.bytes_downloaded += bytes_count
        return bytes_count

    def increment_error(self):
        """Record a proxy error and potentially rotate session."""
        self.error_count += 1
        self.consecutive_errors += 1

        # Rotate session after consecutive errors exceed threshold
        if self.consecutive_errors >= self.config.max_proxy_retries:
            self.rotate_session()

    def reset_errors(self):
        """Reset consecutive error count after successful request."""
        self.consecutive_errors = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return proxy statistics."""
        return {
            "proxy_requests": self.request_count,
            "proxy_errors": self.error_count,
            "bytes_downloaded": self.bytes_downloaded,
            "mb_downloaded": self.bytes_downloaded / (1024 * 1024),
            "session_rotations": self.rotation_count,
        }


@dataclass
class ForumConfig:
    """Configuration for a specific forum."""

    # Identity
    name: str
    forum_id: str
    base_url: str
    platform: str

    # Rate limiting
    min_delay: float
    max_delay: float
    retry_delay: float
    max_retries: int
    backoff_multiplier: float

    # HTTP
    timeout: int
    user_agent: str
    headers: Dict[str, str]

    # Storage
    storage_base: Path
    checkpoint_interval: int

    # Selectors
    selectors: Selectors

    # Authentication
    auth: AuthConfig

    # Proxy
    proxy: ProxyConfig = field(default_factory=ProxyConfig)

    # Image settings
    image_skip_patterns: List[str] = field(default_factory=list)


# Legacy alias for backward compatibility
ScraperConfig = ForumConfig


def load_forum_config(config_path: Path, forum_id: Optional[str] = None) -> ForumConfig:
    """
    Load configuration for a specific forum.

    Args:
        config_path: Path to scraper_config.yaml
        forum_id: Forum ID to load (uses active_forum if not specified)

    Returns:
        ForumConfig for the specified forum
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine which forum to load
    if forum_id is None:
        forum_id = cfg.get("active_forum", "e30m3datasrc")

    if forum_id not in cfg.get("forums", {}):
        raise ValueError(f"Forum '{forum_id}' not found in config. Available: {list(cfg.get('forums', {}).keys())}")

    forum_cfg = cfg["forums"][forum_id]
    defaults = cfg.get("defaults", {})
    platform_id = forum_cfg.get("platform", "vbulletin6")
    platform_cfg = cfg.get("platforms", {}).get(platform_id, {})

    # Merge rate limiting (forum overrides defaults)
    rate_cfg = {**defaults.get("rate_limiting", {}), **forum_cfg.get("rate_limiting", {})}

    # Merge HTTP settings
    http_cfg = {**defaults.get("http", {}), **forum_cfg.get("http", {})}

    # Merge checkpoint settings
    checkpoint_cfg = {**defaults.get("checkpoints", {}), **forum_cfg.get("checkpoints", {})}

    # Build selectors from platform + overrides
    platform_selectors = platform_cfg.get("selectors", {})
    selector_overrides = forum_cfg.get("selector_overrides", {})
    merged_selectors = {**platform_selectors, **selector_overrides}
    selectors = Selectors(**{k: v for k, v in merged_selectors.items() if hasattr(Selectors, k)})

    # Build auth config
    platform_auth = platform_cfg.get("auth", {})
    forum_auth = forum_cfg.get("auth", {})
    auth_enabled = forum_auth.get("enabled", False)

    auth = AuthConfig(
        enabled=auth_enabled,
        login_url=forum_auth.get("login_url", platform_auth.get("login_url", "")),
        login_form_selector=platform_auth.get("login_form_selector", ""),
        username_field=platform_auth.get("username_field", "username"),
        password_field=platform_auth.get("password_field", "password"),
        csrf_field=platform_auth.get("csrf_field", ""),
        success_indicator=platform_auth.get("success_indicator", ""),
        username=os.environ.get(forum_auth.get("username_env", ""), ""),
        password=os.environ.get(forum_auth.get("password_env", ""), ""),
    )

    # Build proxy config
    default_proxy = defaults.get("proxy", {})
    forum_proxy = forum_cfg.get("proxy", {})
    proxy_cfg = {**default_proxy, **forum_proxy}

    # Load credentials from environment variables
    proxy_username_env = proxy_cfg.get("username_env", "OXYLABS_USERNAME")
    proxy_password_env = proxy_cfg.get("password_env", "OXYLABS_PASSWORD")
    proxy_username = os.environ.get(proxy_username_env, "")
    proxy_password = os.environ.get(proxy_password_env, "")

    # Auto-enable proxy if credentials are present and not explicitly disabled
    proxy_enabled = proxy_cfg.get("enabled", False)
    if proxy_username and proxy_password and not proxy_cfg.get("enabled") is False:
        proxy_enabled = proxy_cfg.get("enabled", True)

    proxy = ProxyConfig(
        enabled=proxy_enabled,
        provider=proxy_cfg.get("provider", "oxylabs"),
        host=proxy_cfg.get("host", "pr.oxylabs.io"),
        port=proxy_cfg.get("port", 7777),
        username=proxy_username,
        password=proxy_password,
        country=proxy_cfg.get("country", ""),
        city=proxy_cfg.get("city", ""),
        state=proxy_cfg.get("state", ""),
        sticky_session=proxy_cfg.get("sticky_session", False),
        session_id=proxy_cfg.get("session_id", ""),
        fallback_on_failure=proxy_cfg.get("fallback_on_failure", True),
        max_proxy_retries=proxy_cfg.get("max_proxy_retries", 3),
        track_bandwidth=proxy_cfg.get("track_bandwidth", True),
    )

    # Image skip patterns
    image_cfg = {**defaults.get("images", {}), **forum_cfg.get("images", {})}
    image_skip_patterns = image_cfg.get("skip_patterns", [])

    # Base URL: check environment variable first, then config
    # Environment variable name can be specified in config (e.g., "E30M3_FORUM_URL")
    base_url_env = forum_cfg.get("base_url_env", f"{forum_id.upper()}_FORUM_URL")
    base_url = os.environ.get(base_url_env) or forum_cfg.get("base_url", "")
    if not base_url:
        raise ValueError(
            f"Forum '{forum_id}' has no base_url configured. "
            f"Set the {base_url_env} environment variable or add base_url to config."
        )

    # Storage path - include domain from base_url for organization
    storage_base_dir = Path(forum_cfg.get("storage", {}).get("base_dir", "data_src/forum"))
    domain = urlparse(base_url).netloc if base_url else forum_id
    storage_base = storage_base_dir / domain

    return ForumConfig(
        name=forum_cfg.get("name", forum_id),
        forum_id=forum_id,
        base_url=base_url,
        platform=platform_id,
        min_delay=rate_cfg.get("min_delay_seconds", 1.5),
        max_delay=rate_cfg.get("max_delay_seconds", 2.5),
        retry_delay=rate_cfg.get("retry_delay_seconds", 5),
        max_retries=rate_cfg.get("max_retries", 3),
        backoff_multiplier=rate_cfg.get("backoff_multiplier", 2),
        timeout=http_cfg.get("timeout_seconds", 30),
        user_agent=http_cfg.get("user_agent", "Mozilla/5.0"),
        headers=http_cfg.get("headers", {}),
        storage_base=storage_base,
        checkpoint_interval=checkpoint_cfg.get("save_interval", 10),
        selectors=selectors,
        auth=auth,
        proxy=proxy,
        image_skip_patterns=image_skip_patterns,
    )


# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(name: str, log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with console and optional file output.

    Args:
        name: Logger name
        log_dir: Optional directory for log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

    return logger


# ============================================================================
# HTTP Session with Rate Limiting and Authentication
# ============================================================================


class ScraperSession:
    """
    HTTP session with built-in rate limiting, retry logic, and authentication.

    Features:
    - Randomized delays between requests
    - Exponential backoff on failures
    - Automatic retry with configurable limits
    - Cookie-based authentication
    - Request/response logging
    """

    def __init__(self, config: ForumConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize scraper session.

        Args:
            config: Forum configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.session = self._create_session()
        self.last_request_time: Optional[float] = None
        self.request_count = 0
        self.error_count = 0
        self.authenticated = False

        # Initialize proxy manager
        self.proxy_manager = ProxyManager(config.proxy, self.logger)
        if config.proxy.enabled:
            geo_info = []
            if config.proxy.country:
                geo_info.append(f"country={config.proxy.country}")
            if config.proxy.city:
                geo_info.append(f"city={config.proxy.city}")
            if config.proxy.state:
                geo_info.append(f"state={config.proxy.state}")
            geo_str = f" ({', '.join(geo_info)})" if geo_info else ""
            self.logger.info(f"Proxy enabled: {config.proxy.provider}{geo_str}")

    def _create_session(self) -> requests.Session:
        """Create configured requests session with retry adapter.

        Uses cloudscraper if available to bypass Cloudflare protection.
        Falls back to standard requests.Session otherwise.
        """
        # Use cloudscraper for Cloudflare-protected sites
        if CLOUDSCRAPER_AVAILABLE:
            self.logger.info("Using cloudscraper for Cloudflare bypass")
            session = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'desktop': True
                }
            )
        else:
            self.logger.debug("Using standard requests session")
            session = requests.Session()

        # Set headers
        session.headers.update(self.config.headers)
        session.headers["User-Agent"] = self.config.user_agent

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_multiplier,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _wait_for_rate_limit(self):
        """Wait appropriate time before next request."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            # Randomize delay for politeness
            target_delay = random.uniform(self.config.min_delay, self.config.max_delay)
            if elapsed < target_delay:
                sleep_time = target_delay - elapsed
                self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

    def login(self) -> bool:
        """
        Authenticate with the forum if auth is enabled.

        Returns:
            True if login successful or not required, False on failure
        """
        auth = self.config.auth
        if not auth.enabled:
            self.logger.debug("Authentication not enabled")
            return True

        if not auth.username or not auth.password:
            self.logger.error("Authentication enabled but credentials not provided")
            self.logger.error("Set environment variables specified in config (username_env, password_env)")
            return False

        login_url = urljoin(self.config.base_url, auth.login_url)
        self.logger.info(f"Authenticating at {login_url}")

        try:
            # Use sticky session for login flow (same IP for GET and POST)
            proxy_dict = self.proxy_manager.get_proxy_dict(force_sticky=True)

            # First, fetch the login page to get any CSRF token
            self._wait_for_rate_limit()
            login_page = self.session.get(
                login_url,
                timeout=self.config.timeout,
                proxies=proxy_dict,
            )
            self.last_request_time = time.time()
            self.request_count += 1
            if proxy_dict:
                self.proxy_manager.request_count += 1
            login_page.raise_for_status()

            # Extract CSRF token if configured
            csrf_token = ""
            if auth.csrf_field:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(login_page.text, "html.parser")

                # Try hidden input field
                csrf_input = soup.select_one(f"input[name='{auth.csrf_field}']")
                if csrf_input:
                    csrf_token = csrf_input.get("value", "")

                # Try meta tag
                if not csrf_token:
                    csrf_meta = soup.select_one(f"meta[name='{auth.csrf_field}']")
                    if csrf_meta:
                        csrf_token = csrf_meta.get("content", "")

            # Build login payload
            payload = {
                auth.username_field: auth.username,
                auth.password_field: auth.password,
            }
            if csrf_token:
                payload[auth.csrf_field] = csrf_token

            # Submit login form (same proxy session for consistent IP)
            self._wait_for_rate_limit()
            response = self.session.post(
                login_url,
                data=payload,
                timeout=self.config.timeout,
                allow_redirects=True,
                proxies=proxy_dict,
            )
            self.last_request_time = time.time()
            self.request_count += 1
            if proxy_dict:
                self.proxy_manager.request_count += 1

            # Check for success
            if auth.success_indicator:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                if soup.select_one(auth.success_indicator):
                    self.logger.info("Authentication successful")
                    self.authenticated = True
                    return True
                else:
                    self.logger.error("Authentication failed - success indicator not found")
                    return False

            # Fallback: assume success if no error
            self.logger.info("Authentication completed (no success indicator to verify)")
            self.authenticated = True
            return True

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False

    def get(self, url: str, allow_redirects: bool = True, _use_proxy: bool = True) -> Optional[requests.Response]:
        """
        Make a GET request with rate limiting, error handling, and proxy support.

        Args:
            url: URL to fetch (can be relative to base_url)
            allow_redirects: Whether to follow redirects
            _use_proxy: Internal flag for fallback (don't set manually)

        Returns:
            Response object or None on failure
        """
        # Resolve relative URLs
        if not url.startswith("http"):
            url = urljoin(self.config.base_url, url)

        # Rate limiting
        self._wait_for_rate_limit()

        # Get proxy settings
        proxy_dict = self.proxy_manager.get_proxy_dict() if _use_proxy else None

        try:
            self.logger.debug(f"GET {url}" + (" (via proxy)" if proxy_dict else ""))
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                allow_redirects=allow_redirects,
                proxies=proxy_dict,
            )
            self.last_request_time = time.time()
            self.request_count += 1
            if proxy_dict:
                self.proxy_manager.request_count += 1
                self.proxy_manager.track_bandwidth(response)
                self.proxy_manager.reset_errors()

            # Check for rate limiting response
            if response.status_code == 429:
                self.logger.warning(f"Rate limited (429), waiting {self.config.retry_delay}s")
                time.sleep(self.config.retry_delay)
                return self.get(url, allow_redirects, _use_proxy)  # Retry

            response.raise_for_status()
            self.logger.debug(f"Success: {response.status_code}, {len(response.content)} bytes")
            return response

        except requests.exceptions.ProxyError as e:
            self.logger.warning(f"Proxy error fetching {url}: {e}")
            self.error_count += 1
            if proxy_dict:
                self.proxy_manager.increment_error()

            # Fallback to direct connection if enabled
            if self.config.proxy.fallback_on_failure and _use_proxy:
                self.logger.info("Falling back to direct connection (no proxy)")
                return self.get(url, allow_redirects, _use_proxy=False)
            return None

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching {url}")
            self.error_count += 1
            if proxy_dict:
                self.proxy_manager.increment_error()
            return None

        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error fetching {url}: {e}")
            self.error_count += 1
            return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error fetching {url}: {e}")
            self.error_count += 1
            if proxy_dict:
                self.proxy_manager.increment_error()
            return None

    def get_html(self, url: str) -> Optional[str]:
        """
        Fetch URL and return HTML content as string.

        Args:
            url: URL to fetch

        Returns:
            HTML string or None on failure
        """
        response = self.get(url)
        if response is not None:
            return response.text
        return None

    def download_file(self, url: str, dest_path: Path, timeout: int = 60) -> bool:
        """
        Download a file (e.g., image) to disk.

        Args:
            url: URL to download
            dest_path: Destination file path
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        if not url.startswith("http"):
            url = urljoin(self.config.base_url, url)

        self._wait_for_rate_limit()

        # Get proxy settings
        proxy_dict = self.proxy_manager.get_proxy_dict()

        try:
            self.logger.debug(f"Downloading {url} -> {dest_path}" + (" (via proxy)" if proxy_dict else ""))
            response = self.session.get(url, timeout=timeout, stream=True, proxies=proxy_dict)
            self.last_request_time = time.time()
            self.request_count += 1
            if proxy_dict:
                self.proxy_manager.request_count += 1

            response.raise_for_status()

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            total_bytes = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_bytes += len(chunk)

            # Track bandwidth for file downloads
            if proxy_dict and self.proxy_manager.config.track_bandwidth:
                self.proxy_manager.bytes_downloaded += total_bytes
                self.proxy_manager.reset_errors()

            self.logger.debug(f"Downloaded {dest_path.stat().st_size} bytes")
            return True

        except requests.exceptions.ProxyError as e:
            self.logger.warning(f"Proxy error downloading {url}: {e}")
            self.error_count += 1
            if proxy_dict:
                self.proxy_manager.increment_error()
            return False

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            self.error_count += 1
            if proxy_dict:
                self.proxy_manager.increment_error()
            return False

    def save_cookies(self, path: Path):
        """Save session cookies to file for later reuse."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cookies = requests.utils.dict_from_cookiejar(self.session.cookies)
        with open(path, "w") as f:
            json.dump(cookies, f)
        self.logger.debug(f"Saved {len(cookies)} cookies to {path}")

    def load_cookies(self, path: Path) -> bool:
        """Load session cookies from file."""
        if not path.exists():
            return False
        try:
            with open(path, "r") as f:
                cookies = json.load(f)
            self.session.cookies.update(cookies)
            self.logger.debug(f"Loaded {len(cookies)} cookies from {path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load cookies: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return session statistics including proxy metrics."""
        stats = {
            "requests": self.request_count,
            "errors": self.error_count,
        }
        # Add proxy stats if proxy is enabled
        if self.proxy_manager:
            stats.update(self.proxy_manager.get_stats())
        return stats


# ============================================================================
# Checkpoint System
# ============================================================================


@dataclass
class Checkpoint:
    """
    Checkpoint state for resumable scraping.

    Tracks:
    - Last processed item ID
    - List of completed items
    - Progress statistics
    - Timestamp information
    """

    stage: str
    last_item_id: Optional[str] = None
    completed_ids: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)
    total_processed: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_completed(self, item_id: str):
        """Mark an item as successfully processed."""
        if item_id not in self.completed_ids:
            self.completed_ids.append(item_id)
        self.last_item_id = item_id
        self.total_processed += 1
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, item_id: str):
        """Mark an item as failed."""
        if item_id not in self.failed_ids:
            self.failed_ids.append(item_id)
        self.updated_at = datetime.now().isoformat()

    def is_completed(self, item_id: str) -> bool:
        """Check if an item has been completed."""
        return item_id in self.completed_ids

    def save(self, path: Path):
        """Save checkpoint to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        """Load checkpoint from JSON file, or return None if not found."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Failed to load checkpoint {path}: {e}")
            return None

    @classmethod
    def load_or_create(cls, path: Path, stage: str) -> "Checkpoint":
        """Load existing checkpoint or create new one."""
        existing = cls.load(path)
        if existing is not None:
            logging.info(f"Resuming from checkpoint: {existing.total_processed} items processed")
            return existing
        return cls(stage=stage)


# ============================================================================
# Data Validation
# ============================================================================


THREAD_SCHEMA = {
    "required": ["thread_id", "title", "url", "forum_id"],
    "optional": ["author", "author_id", "created_at", "reply_count", "view_count",
                 "last_post_at", "last_poster", "is_sticky", "is_locked", "page_count"],
}

POST_SCHEMA = {
    "required": ["post_id", "thread_id", "author"],
    "optional": ["author_id", "content_html", "content_text", "created_at",
                 "post_number", "is_first_post", "images", "quotes", "thread_title"],
}


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_record(record: Dict, schema: Dict, record_type: str = "record") -> List[str]:
    """
    Validate a record against a schema.

    Args:
        record: Dictionary to validate
        schema: Schema with 'required' and 'optional' keys
        record_type: Type name for error messages

    Returns:
        List of warning messages (empty if valid)

    Raises:
        ValidationError: If required fields are missing
    """
    warnings = []

    # Check required fields
    missing = [f for f in schema["required"] if f not in record or record[f] is None]
    if missing:
        raise ValidationError(f"{record_type} missing required fields: {missing}")

    # Check for empty required fields
    for field in schema["required"]:
        if field in record and record[field] == "":
            warnings.append(f"{record_type} has empty required field: {field}")

    return warnings


def load_jsonl_validated(path: Path, schema: Dict, record_type: str = "record") -> List[Dict]:
    """
    Load and validate JSONL file.

    Args:
        path: Path to JSONL file
        schema: Validation schema
        record_type: Type name for error messages

    Returns:
        List of validated records

    Raises:
        ValidationError: If any record fails validation
    """
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, schema, f"{record_type} (line {line_num})")
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON at line {line_num}: {e}")

    return records


# ============================================================================
# Data Storage
# ============================================================================


def save_json(data: Any, path: Path, pretty: bool = True):
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)


def load_json(path: Path) -> Optional[Any]:
    """Load data from JSON file."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(data: Dict, path: Path):
    """Append a record to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    """Load all records from a JSON Lines file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_html(html: str, path: Path):
    """Save raw HTML to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================================
# URL Utilities
# ============================================================================


def extract_id_from_url(url: str, pattern: str = "threads") -> Optional[str]:
    """
    Extract numeric ID from vBulletin URL.

    Examples:
        /threads/12345-thread-title -> "12345"
        /forum/67-subforum-name -> "67"
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    for i, part in enumerate(parts):
        if part == pattern and i + 1 < len(parts):
            # Next part should be ID-slug
            next_part = parts[i + 1]
            if "-" in next_part:
                return next_part.split("-")[0]
            return next_part

    # Try to find ID-slug pattern anywhere
    for part in parts:
        if "-" in part:
            potential_id = part.split("-")[0]
            if potential_id.isdigit():
                return potential_id

    return None


def normalize_url(url: str, base_url: str) -> str:
    """Normalize URL to absolute form."""
    if url.startswith("//"):
        return "https:" + url
    if not url.startswith("http"):
        return urljoin(base_url, url)
    return url
