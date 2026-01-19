"""
Unit tests for proxy support in scraper/core.py

Tests the Oxylabs residential proxy integration including:
- ProxyConfig dataclass and URL building
- ProxyManager for session control and bandwidth tracking
- ScraperSession integration with proxy support

TDD: Write these tests first, then implement to pass.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from scraper.core import (
    ProxyConfig,
    ProxyManager,
    ForumConfig,
    ScraperSession,
    AuthConfig,
)


# =============================================================================
# ProxyConfig Tests
# =============================================================================


class TestProxyConfig:
    """Tests for ProxyConfig dataclass and URL building."""

    def test_build_proxy_url_basic(self):
        """Basic proxy URL with just username/password."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            host="pr.oxylabs.io",
            port=7777,
        )

        url = config.build_proxy_url()

        assert url == "http://customer-testuser:testpass@pr.oxylabs.io:7777"

    def test_build_proxy_url_with_country(self):
        """Proxy URL with country targeting."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            country="US",
        )

        url = config.build_proxy_url()

        assert "customer-testuser-cc-US" in url
        assert url == "http://customer-testuser-cc-US:testpass@pr.oxylabs.io:7777"

    def test_build_proxy_url_with_city_state(self):
        """Proxy URL with city and state targeting."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            country="US",
            city="Los Angeles",
            state="California",
        )

        url = config.build_proxy_url()

        assert "cc-US" in url
        assert "city-los_angeles" in url
        assert "st-california" in url

    def test_build_proxy_url_with_sticky_session(self):
        """Proxy URL with sticky session ID."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            sticky_session=True,
            session_id="abc123",
        )

        url = config.build_proxy_url()

        assert "sessid-abc123" in url
        assert url == "http://customer-testuser-sessid-abc123:testpass@pr.oxylabs.io:7777"

    def test_build_proxy_url_full_targeting(self):
        """Proxy URL with all targeting options."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            country="US",
            city="New York",
            state="New York",
            sticky_session=True,
            session_id="xyz789",
        )

        url = config.build_proxy_url()

        assert "customer-testuser" in url
        assert "cc-US" in url
        assert "city-new_york" in url
        assert "st-new_york" in url
        assert "sessid-xyz789" in url

    def test_build_proxy_url_disabled_returns_none(self):
        """Disabled proxy should return None."""
        config = ProxyConfig(
            enabled=False,
            username="testuser",
            password="testpass",
        )

        url = config.build_proxy_url()

        assert url is None

    def test_build_proxy_url_missing_credentials_returns_none(self):
        """Missing credentials should return None."""
        # Missing username
        config1 = ProxyConfig(enabled=True, username="", password="testpass")
        assert config1.build_proxy_url() is None

        # Missing password
        config2 = ProxyConfig(enabled=True, username="testuser", password="")
        assert config2.build_proxy_url() is None

        # Both missing
        config3 = ProxyConfig(enabled=True, username="", password="")
        assert config3.build_proxy_url() is None

    def test_build_proxy_url_custom_host_port(self):
        """Should support custom host and port."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            host="custom.proxy.io",
            port=8080,
        )

        url = config.build_proxy_url()

        assert "custom.proxy.io:8080" in url

    def test_default_values(self):
        """ProxyConfig should have sensible defaults."""
        config = ProxyConfig()

        assert config.enabled is False
        assert config.provider == "oxylabs"
        assert config.host == "pr.oxylabs.io"
        assert config.port == 7777
        assert config.fallback_on_failure is True
        assert config.max_proxy_retries == 3
        assert config.track_bandwidth is True


# =============================================================================
# ProxyManager Tests
# =============================================================================


class TestProxyManager:
    """Tests for ProxyManager class."""

    def test_get_proxy_dict_returns_http_and_https(self):
        """get_proxy_dict should return both http and https entries."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
        )
        manager = ProxyManager(config)

        proxy_dict = manager.get_proxy_dict()

        assert "http" in proxy_dict
        assert "https" in proxy_dict
        assert proxy_dict["http"] == proxy_dict["https"]
        assert "customer-testuser" in proxy_dict["http"]

    def test_get_proxy_dict_disabled_returns_none(self):
        """Disabled proxy should return None dict."""
        config = ProxyConfig(enabled=False)
        manager = ProxyManager(config)

        proxy_dict = manager.get_proxy_dict()

        assert proxy_dict is None

    def test_get_proxy_dict_sticky_override(self):
        """Should be able to force sticky session via parameter."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            sticky_session=False,  # Not sticky by default
        )
        manager = ProxyManager(config)

        # Without sticky
        normal_dict = manager.get_proxy_dict()
        assert "sessid" not in normal_dict["http"]

        # Force sticky
        sticky_dict = manager.get_proxy_dict(force_sticky=True)
        assert "sessid" in sticky_dict["http"]

    def test_rotate_session_changes_id(self):
        """rotate_session should generate new session ID."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            sticky_session=True,
            session_id="initial123",
        )
        manager = ProxyManager(config)

        old_id = manager.config.session_id
        manager.rotate_session()
        new_id = manager.config.session_id

        assert old_id != new_id
        assert len(new_id) > 0

    def test_bandwidth_tracking_from_response(self):
        """track_bandwidth should accumulate bytes from responses."""
        config = ProxyConfig(enabled=True, username="u", password="p", track_bandwidth=True)
        manager = ProxyManager(config)

        # Mock response with content
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1024"}

        manager.track_bandwidth(mock_response)

        assert manager.bytes_downloaded >= 1024

    def test_bandwidth_tracking_without_content_length(self):
        """Should handle missing Content-Length header."""
        config = ProxyConfig(enabled=True, username="u", password="p", track_bandwidth=True)
        manager = ProxyManager(config)

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.content = b"test content"

        manager.track_bandwidth(mock_response)

        assert manager.bytes_downloaded == len(b"test content")

    def test_bandwidth_tracking_disabled(self):
        """Should not track when disabled."""
        config = ProxyConfig(enabled=True, username="u", password="p", track_bandwidth=False)
        manager = ProxyManager(config)

        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1024"}

        manager.track_bandwidth(mock_response)

        assert manager.bytes_downloaded == 0

    def test_get_stats_returns_all_metrics(self):
        """get_stats should return comprehensive metrics."""
        config = ProxyConfig(enabled=True, username="u", password="p")
        manager = ProxyManager(config)

        # Simulate some activity
        manager.request_count = 10
        manager.error_count = 2
        manager.bytes_downloaded = 5000
        manager.rotation_count = 3

        stats = manager.get_stats()

        assert stats["proxy_requests"] == 10
        assert stats["proxy_errors"] == 2
        assert stats["bytes_downloaded"] == 5000
        assert stats["mb_downloaded"] == pytest.approx(5000 / (1024 * 1024), rel=0.01)
        assert stats["session_rotations"] == 3

    def test_increment_error_triggers_rotation(self):
        """Should rotate session after consecutive errors."""
        config = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            sticky_session=True,
            session_id="initial",
            max_proxy_retries=2,
        )
        manager = ProxyManager(config)
        initial_id = manager.config.session_id

        # First error
        manager.increment_error()
        assert manager.config.session_id == initial_id  # Not rotated yet

        # Second error (threshold reached)
        manager.increment_error()
        assert manager.config.session_id != initial_id  # Should have rotated

    def test_reset_error_count(self):
        """Successful request should reset error count."""
        config = ProxyConfig(enabled=True, username="u", password="p")
        manager = ProxyManager(config)

        manager.increment_error()
        manager.increment_error()
        assert manager.consecutive_errors == 2

        manager.reset_errors()
        assert manager.consecutive_errors == 0


# =============================================================================
# ScraperSession with Proxy Tests
# =============================================================================


class TestScraperSessionProxy:
    """Tests for ScraperSession with proxy support."""

    @pytest.fixture
    def mock_config_with_proxy(self):
        """Create a mock config with proxy enabled."""
        config = Mock(spec=ForumConfig)
        config.base_url = "https://example.com"
        config.min_delay = 0.01
        config.max_delay = 0.02
        config.retry_delay = 0.01
        config.max_retries = 2
        config.backoff_multiplier = 1.0
        config.timeout = 5
        config.user_agent = "TestBot/1.0"
        config.headers = {"Accept": "text/html"}
        config.auth = AuthConfig(enabled=False)
        config.proxy = ProxyConfig(
            enabled=True,
            username="testuser",
            password="testpass",
            country="US",
        )
        return config

    @pytest.fixture
    def mock_config_no_proxy(self):
        """Create a mock config with proxy disabled."""
        config = Mock(spec=ForumConfig)
        config.base_url = "https://example.com"
        config.min_delay = 0.01
        config.max_delay = 0.02
        config.retry_delay = 0.01
        config.max_retries = 2
        config.backoff_multiplier = 1.0
        config.timeout = 5
        config.user_agent = "TestBot/1.0"
        config.headers = {"Accept": "text/html"}
        config.auth = AuthConfig(enabled=False)
        config.proxy = ProxyConfig(enabled=False)
        return config

    def test_request_uses_proxy_when_enabled(self, mock_config_with_proxy):
        """GET request should use proxy when enabled."""
        session = ScraperSession(mock_config_with_proxy)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test"
            mock_response.headers = {"Content-Length": "4"}
            mock_get.return_value = mock_response

            session.get("https://example.com/page")

            # Check that proxies were passed
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert "proxies" in call_kwargs
            assert call_kwargs["proxies"] is not None
            assert "customer-testuser" in call_kwargs["proxies"]["http"]

    def test_request_without_proxy_when_disabled(self, mock_config_no_proxy):
        """GET request should not use proxy when disabled."""
        session = ScraperSession(mock_config_no_proxy)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test"
            mock_get.return_value = mock_response

            session.get("https://example.com/page")

            # Check that no proxies were passed (or None)
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            proxies = call_kwargs.get("proxies")
            assert proxies is None

    def test_fallback_on_proxy_error(self, mock_config_with_proxy):
        """Should fall back to direct connection on proxy error."""
        mock_config_with_proxy.proxy.fallback_on_failure = True
        session = ScraperSession(mock_config_with_proxy)

        with patch.object(session.session, "get") as mock_get:
            # First call fails with proxy error
            proxy_error = requests.exceptions.ProxyError("Proxy failed")
            success_response = Mock()
            success_response.status_code = 200
            success_response.content = b"test"
            mock_get.side_effect = [proxy_error, success_response]

            response = session.get("https://example.com/page")

            # Should have made two calls - first with proxy, second without
            assert mock_get.call_count == 2
            # Second call should be without proxy
            second_call_kwargs = mock_get.call_args_list[1][1]
            assert second_call_kwargs.get("proxies") is None
            assert response is not None

    def test_no_fallback_when_disabled(self, mock_config_with_proxy):
        """Should not fall back when fallback is disabled."""
        mock_config_with_proxy.proxy.fallback_on_failure = False
        session = ScraperSession(mock_config_with_proxy)

        with patch.object(session.session, "get") as mock_get:
            mock_get.side_effect = requests.exceptions.ProxyError("Proxy failed")

            response = session.get("https://example.com/page")

            # Should only have tried once
            assert mock_get.call_count == 1
            assert response is None

    def test_sticky_session_for_login(self, mock_config_with_proxy):
        """Login should use sticky session to maintain same IP."""
        mock_config_with_proxy.auth = AuthConfig(
            enabled=True,
            login_url="/login",
            username_field="user",
            password_field="pass",
            username="testuser",
            password="testpass",
        )
        session = ScraperSession(mock_config_with_proxy)

        with patch.object(session.session, "get") as mock_get, \
             patch.object(session.session, "post") as mock_post:

            # Mock login page fetch
            mock_login_page = Mock()
            mock_login_page.status_code = 200
            mock_login_page.text = "<html><form></form></html>"
            mock_login_page.headers = {}
            mock_get.return_value = mock_login_page

            # Mock login POST
            mock_login_response = Mock()
            mock_login_response.status_code = 200
            mock_login_response.text = "<html></html>"
            mock_post.return_value = mock_login_response

            session.login()

            # Both GET and POST should use sticky session (same sessid)
            get_proxies = mock_get.call_args[1].get("proxies", {})
            post_proxies = mock_post.call_args[1].get("proxies", {})

            if get_proxies and post_proxies:
                # Extract session IDs and verify they match
                assert "sessid" in get_proxies.get("http", "")
                assert "sessid" in post_proxies.get("http", "")

    def test_proxy_rotation_on_failure(self, mock_config_with_proxy):
        """Should rotate proxy session on repeated failures."""
        mock_config_with_proxy.proxy.sticky_session = True
        mock_config_with_proxy.proxy.session_id = "initial123"
        mock_config_with_proxy.proxy.max_proxy_retries = 2
        session = ScraperSession(mock_config_with_proxy)

        initial_session_id = session.proxy_manager.config.session_id

        with patch.object(session.session, "get") as mock_get:
            # Simulate repeated proxy failures
            mock_get.side_effect = requests.exceptions.ProxyError("Failed")

            # Make multiple failed requests
            session.get("https://example.com/page1")
            session.get("https://example.com/page2")

            # Session ID should have changed after failures
            assert session.proxy_manager.rotation_count >= 1

    def test_download_file_uses_proxy(self, mock_config_with_proxy):
        """download_file should use proxy when enabled."""
        session = ScraperSession(mock_config_with_proxy)

        with patch.object(session.session, "get") as mock_get, \
             patch("builtins.open", MagicMock()):

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Length": "1000"}
            mock_response.iter_content = Mock(return_value=[b"test"])
            mock_get.return_value = mock_response

            from pathlib import Path
            session.download_file("https://example.com/file.jpg", Path("/tmp/file.jpg"))

            # Check proxy was used
            call_kwargs = mock_get.call_args[1]
            assert "proxies" in call_kwargs
            assert call_kwargs["proxies"] is not None

    def test_stats_include_proxy_metrics(self, mock_config_with_proxy):
        """get_stats should include proxy-related metrics."""
        session = ScraperSession(mock_config_with_proxy)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test content"
            mock_response.headers = {"Content-Length": "12"}
            mock_get.return_value = mock_response

            session.get("https://example.com/page")

            stats = session.get_stats()

            assert "proxy_requests" in stats
            assert "bytes_downloaded" in stats
            assert stats["proxy_requests"] >= 1

    def test_has_proxy_manager_when_enabled(self, mock_config_with_proxy):
        """ScraperSession should have proxy_manager when proxy enabled."""
        session = ScraperSession(mock_config_with_proxy)

        assert hasattr(session, "proxy_manager")
        assert session.proxy_manager is not None
        assert isinstance(session.proxy_manager, ProxyManager)

    def test_no_proxy_manager_when_disabled(self, mock_config_no_proxy):
        """ScraperSession should have None proxy_manager when disabled."""
        session = ScraperSession(mock_config_no_proxy)

        assert hasattr(session, "proxy_manager")
        # proxy_manager exists but returns None for proxy_dict
        assert session.proxy_manager.get_proxy_dict() is None


# =============================================================================
# Integration Tests (Skipped unless OXYLABS credentials present)
# =============================================================================


@pytest.mark.skip(reason="Requires actual Oxylabs credentials")
class TestProxyIntegration:
    """Integration tests with actual proxy service."""

    def test_real_proxy_request(self):
        """Test actual request through Oxylabs proxy."""
        import os

        config = ProxyConfig(
            enabled=True,
            username=os.environ.get("OXYLABS_USERNAME", ""),
            password=os.environ.get("OXYLABS_PASSWORD", ""),
            country="US",
        )

        if not config.username or not config.password:
            pytest.skip("OXYLABS credentials not set")

        manager = ProxyManager(config)
        proxy_dict = manager.get_proxy_dict()

        # Make a request to IP check service
        response = requests.get(
            "https://httpbin.org/ip",
            proxies=proxy_dict,
            timeout=30,
        )

        assert response.status_code == 200
        data = response.json()
        assert "origin" in data
        # IP should be different from local IP (ideally US-based)
        print(f"Proxy IP: {data['origin']}")
