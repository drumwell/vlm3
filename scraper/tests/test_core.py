"""
Unit tests for scraper/core.py

Tests the core infrastructure: config loading, session management,
checkpointing, and utility functions.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import yaml

from scraper.core import (
    ForumConfig,
    ScraperConfig,
    ScraperSession,
    Checkpoint,
    Selectors,
    AuthConfig,
    load_forum_config,
    setup_logging,
    save_json,
    load_json,
    append_jsonl,
    save_html,
    extract_id_from_url,
    normalize_url,
    validate_record,
    load_jsonl_validated,
    ValidationError,
    THREAD_SCHEMA,
    POST_SCHEMA,
)


# =============================================================================
# ForumConfig Tests
# =============================================================================


class TestForumConfig:
    """Tests for configuration loading."""

    def test_load_forum_config(self, tmp_path):
        """Config should load all values from YAML file."""
        config_content = {
            "defaults": {
                "rate_limiting": {
                    "min_delay_seconds": 1.0,
                    "max_delay_seconds": 2.0,
                    "retry_delay_seconds": 5.0,
                    "max_retries": 3,
                    "backoff_multiplier": 2.0,
                },
                "http": {
                    "timeout_seconds": 30,
                    "user_agent": "TestBot/1.0",
                    "headers": {"Accept": "text/html"},
                },
                "checkpoints": {"save_interval": 10},
            },
            "platforms": {
                "test_platform": {
                    "selectors": {
                        "forum_link": ".forum-link",
                        "thread_link": ".thread-link",
                    },
                    "auth": {
                        "login_url": "/login",
                        "username_field": "user",
                        "password_field": "pass",
                    },
                }
            },
            "forums": {
                "test_forum": {
                    "name": "Test Forum",
                    "base_url": "https://example.com",
                    "platform": "test_platform",
                    "storage": {"base_dir": "test_archive"},
                    "auth": {"enabled": False},
                }
            },
            "active_forum": "test_forum",
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_forum_config(config_file)

        assert config.base_url == "https://example.com"
        assert config.min_delay == 1.0
        assert config.max_delay == 2.0
        assert config.retry_delay == 5.0
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.user_agent == "TestBot/1.0"
        assert config.headers == {"Accept": "text/html"}
        assert config.checkpoint_interval == 10
        assert config.selectors.forum_link == ".forum-link"

    def test_load_specific_forum(self, tmp_path):
        """Should load specific forum when forum_id provided."""
        config_content = {
            "defaults": {"rate_limiting": {"min_delay_seconds": 1.0, "max_delay_seconds": 2.0,
                                           "retry_delay_seconds": 5, "max_retries": 3, "backoff_multiplier": 2}},
            "platforms": {"vb": {"selectors": {}}},
            "forums": {
                "forum_a": {"name": "Forum A", "base_url": "https://a.com", "platform": "vb"},
                "forum_b": {"name": "Forum B", "base_url": "https://b.com", "platform": "vb"},
            },
            "active_forum": "forum_a",
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_forum_config(config_file, forum_id="forum_b")
        assert config.base_url == "https://b.com"

    def test_load_missing_file_raises(self, tmp_path):
        """Loading non-existent config should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_forum_config(tmp_path / "nonexistent.yaml")

    def test_load_missing_forum_raises(self, tmp_path):
        """Loading non-existent forum should raise ValueError."""
        config_content = {
            "forums": {"existing": {"base_url": "https://x.com", "platform": "vb"}},
            "platforms": {"vb": {}},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        with pytest.raises(ValueError, match="not found"):
            load_forum_config(config_file, forum_id="nonexistent")


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoint:
    """Tests for checkpoint save/load/resume functionality."""

    def test_mark_completed(self):
        """mark_completed should add item to completed list."""
        cp = Checkpoint(stage="test")

        cp.mark_completed("item_1")
        cp.mark_completed("item_2")

        assert "item_1" in cp.completed_ids
        assert "item_2" in cp.completed_ids
        assert cp.total_processed == 2
        assert cp.last_item_id == "item_2"

    def test_mark_completed_idempotent(self):
        """Marking same item complete multiple times shouldn't duplicate."""
        cp = Checkpoint(stage="test")

        cp.mark_completed("item_1")
        cp.mark_completed("item_1")

        assert cp.completed_ids.count("item_1") == 1
        # But total_processed increments each call (tracks attempts)
        assert cp.total_processed == 2

    def test_mark_failed(self):
        """mark_failed should add item to failed list."""
        cp = Checkpoint(stage="test")

        cp.mark_failed("bad_item")

        assert "bad_item" in cp.failed_ids
        assert "bad_item" not in cp.completed_ids

    def test_is_completed(self):
        """is_completed should return correct status."""
        cp = Checkpoint(stage="test")
        cp.mark_completed("done_item")

        assert cp.is_completed("done_item") is True
        assert cp.is_completed("other_item") is False

    def test_save_and_load(self, tmp_path):
        """Checkpoint should round-trip through save/load."""
        cp = Checkpoint(stage="test_stage")
        cp.mark_completed("item_1")
        cp.mark_completed("item_2")
        cp.mark_failed("bad_item")
        cp.metadata["custom_key"] = "custom_value"

        checkpoint_file = tmp_path / "checkpoint.json"
        cp.save(checkpoint_file)

        loaded = Checkpoint.load(checkpoint_file)

        assert loaded is not None
        assert loaded.stage == "test_stage"
        assert loaded.completed_ids == ["item_1", "item_2"]
        assert loaded.failed_ids == ["bad_item"]
        assert loaded.total_processed == 2
        assert loaded.metadata["custom_key"] == "custom_value"

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading non-existent checkpoint should return None."""
        result = Checkpoint.load(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_or_create_new(self, tmp_path):
        """load_or_create should create new checkpoint if none exists."""
        checkpoint_file = tmp_path / "new_checkpoint.json"

        cp = Checkpoint.load_or_create(checkpoint_file, "new_stage")

        assert cp.stage == "new_stage"
        assert cp.total_processed == 0

    def test_load_or_create_existing(self, tmp_path):
        """load_or_create should load existing checkpoint."""
        checkpoint_file = tmp_path / "existing.json"

        # Create existing checkpoint
        existing = Checkpoint(stage="existing_stage")
        existing.mark_completed("prior_item")
        existing.save(checkpoint_file)

        # Load it
        cp = Checkpoint.load_or_create(checkpoint_file, "ignored_stage")

        assert cp.stage == "existing_stage"
        assert "prior_item" in cp.completed_ids

    def test_save_creates_parent_dirs(self, tmp_path):
        """save should create parent directories if they don't exist."""
        checkpoint_file = tmp_path / "deep" / "nested" / "checkpoint.json"

        cp = Checkpoint(stage="test")
        cp.save(checkpoint_file)

        assert checkpoint_file.exists()


# =============================================================================
# ScraperSession Tests
# =============================================================================


class TestScraperSession:
    """Tests for HTTP session with rate limiting."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=ForumConfig)
        config.base_url = "https://example.com"
        config.min_delay = 0.01  # Fast for testing
        config.max_delay = 0.02
        config.retry_delay = 0.01
        config.max_retries = 2
        config.backoff_multiplier = 1.0
        config.timeout = 5
        config.user_agent = "TestBot/1.0"
        config.headers = {"Accept": "text/html"}
        config.auth = AuthConfig(enabled=False)
        return config

    def test_creates_session_with_headers(self, mock_config):
        """Session should be configured with user agent and headers."""
        session = ScraperSession(mock_config)

        assert session.session.headers["User-Agent"] == "TestBot/1.0"
        assert session.session.headers["Accept"] == "text/html"

    def test_rate_limiting_delays_requests(self, mock_config):
        """Subsequent requests should be delayed by rate limit."""
        mock_config.min_delay = 0.1
        mock_config.max_delay = 0.1

        session = ScraperSession(mock_config)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test"
            mock_get.return_value = mock_response

            # First request
            start = time.time()
            session.get("https://example.com/page1")

            # Second request should wait
            session.get("https://example.com/page2")
            elapsed = time.time() - start

            # Should have waited at least min_delay
            assert elapsed >= 0.1

    def test_get_resolves_relative_urls(self, mock_config):
        """Relative URLs should be resolved against base_url."""
        session = ScraperSession(mock_config)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test"
            mock_get.return_value = mock_response

            session.get("/forum/42")

            # Should have called with absolute URL
            mock_get.assert_called_once()
            called_url = mock_get.call_args[0][0]
            assert called_url == "https://example.com/forum/42"

    def test_get_returns_none_on_timeout(self, mock_config):
        """Timeout should return None, not raise."""
        session = ScraperSession(mock_config)

        with patch.object(session.session, "get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout()

            result = session.get("https://example.com/slow")

            assert result is None
            assert session.error_count == 1

    def test_get_html_returns_text(self, mock_config):
        """get_html should return response text."""
        session = ScraperSession(mock_config)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html>Test</html>"
            mock_response.content = b"<html>Test</html>"
            mock_get.return_value = mock_response

            html = session.get_html("https://example.com")

            assert html == "<html>Test</html>"

    def test_stats_tracking(self, mock_config):
        """Session should track request and error counts."""
        session = ScraperSession(mock_config)

        with patch.object(session.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test"
            mock_get.return_value = mock_response

            session.get("https://example.com/page1")
            session.get("https://example.com/page2")

            stats = session.get_stats()
            assert stats["requests"] == 2
            assert stats["errors"] == 0


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_save_and_load_json(self, tmp_path):
        """JSON should round-trip correctly."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_file = tmp_path / "test.json"

        save_json(data, json_file)
        loaded = load_json(json_file)

        assert loaded == data

    def test_load_json_nonexistent_returns_none(self, tmp_path):
        """Loading non-existent JSON should return None."""
        result = load_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_append_jsonl(self, tmp_path):
        """append_jsonl should append records to file."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl({"id": 1}, jsonl_file)
        append_jsonl({"id": 2}, jsonl_file)
        append_jsonl({"id": 3}, jsonl_file)

        with open(jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[2]) == {"id": 3}

    def test_save_html(self, tmp_path):
        """save_html should write HTML content."""
        html_file = tmp_path / "page.html"
        content = "<html><body>Test</body></html>"

        save_html(content, html_file)

        assert html_file.read_text() == content

    def test_save_html_creates_dirs(self, tmp_path):
        """save_html should create parent directories."""
        html_file = tmp_path / "deep" / "path" / "page.html"

        save_html("<html></html>", html_file)

        assert html_file.exists()

    def test_extract_id_from_thread_url(self):
        """Should extract thread ID from URL."""
        url = "/threads/12345-some-thread-title"
        assert extract_id_from_url(url, "threads") == "12345"

    def test_extract_id_from_forum_url(self):
        """Should extract forum ID from URL."""
        url = "https://example.com/forum/42-my-forum"
        assert extract_id_from_url(url, "forum") == "42"

    def test_extract_id_no_match(self):
        """Should return None if pattern not found."""
        url = "/members/100-username"
        result = extract_id_from_url(url, "threads")
        # Falls back to finding any ID-slug pattern
        assert result == "100"

    def test_normalize_url_absolute(self):
        """Absolute URLs should pass through."""
        url = "https://other.com/page"
        result = normalize_url(url, "https://example.com")
        assert result == "https://other.com/page"

    def test_normalize_url_relative(self):
        """Relative URLs should be resolved."""
        url = "/forum/42"
        result = normalize_url(url, "https://example.com")
        assert result == "https://example.com/forum/42"

    def test_normalize_url_protocol_relative(self):
        """Protocol-relative URLs should get https."""
        url = "//cdn.example.com/image.jpg"
        result = normalize_url(url, "https://example.com")
        assert result == "https://cdn.example.com/image.jpg"


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for logging setup."""

    def test_setup_logging_returns_logger(self):
        """setup_logging should return a configured logger."""
        logger = setup_logging("test_logger")

        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0

    def test_setup_logging_with_file(self, tmp_path):
        """setup_logging should create log file when log_dir specified."""
        logger = setup_logging("file_logger", log_dir=tmp_path)

        # Should have created a log file
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1

    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice shouldn't duplicate handlers."""
        logger1 = setup_logging("idempotent_logger")
        handler_count = len(logger1.handlers)

        logger2 = setup_logging("idempotent_logger")

        assert len(logger2.handlers) == handler_count


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for data validation functions."""

    def test_validate_record_valid(self):
        """Valid record should pass validation."""
        record = {"thread_id": "123", "title": "Test", "url": "/test", "forum_id": "1"}
        warnings = validate_record(record, THREAD_SCHEMA, "thread")
        assert warnings == []

    def test_validate_record_missing_required(self):
        """Missing required field should raise ValidationError."""
        record = {"thread_id": "123", "title": "Test"}  # Missing url, forum_id
        with pytest.raises(ValidationError, match="missing required fields"):
            validate_record(record, THREAD_SCHEMA, "thread")

    def test_validate_record_empty_required_warns(self):
        """Empty required field should produce warning."""
        record = {"thread_id": "123", "title": "", "url": "/test", "forum_id": "1"}
        warnings = validate_record(record, THREAD_SCHEMA, "thread")
        assert any("empty required field" in w for w in warnings)

    def test_load_jsonl_validated(self, tmp_path):
        """load_jsonl_validated should load and validate records."""
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"thread_id": "1", "title": "A", "url": "/a", "forum_id": "1"}\n')
            f.write('{"thread_id": "2", "title": "B", "url": "/b", "forum_id": "1"}\n')

        records = load_jsonl_validated(jsonl_file, THREAD_SCHEMA, "thread")
        assert len(records) == 2

    def test_load_jsonl_validated_invalid_record(self, tmp_path):
        """load_jsonl_validated should raise on invalid record."""
        jsonl_file = tmp_path / "bad.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"thread_id": "1"}\n')  # Missing required fields

        with pytest.raises(ValidationError):
            load_jsonl_validated(jsonl_file, THREAD_SCHEMA, "thread")

    def test_load_jsonl_validated_invalid_json(self, tmp_path):
        """load_jsonl_validated should raise on invalid JSON."""
        jsonl_file = tmp_path / "broken.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('not valid json\n')

        with pytest.raises(ValidationError, match="Invalid JSON"):
            load_jsonl_validated(jsonl_file, THREAD_SCHEMA, "thread")

    def test_load_jsonl_nonexistent_returns_empty(self, tmp_path):
        """load_jsonl_validated should return empty list for missing file."""
        records = load_jsonl_validated(tmp_path / "missing.jsonl", THREAD_SCHEMA)
        assert records == []


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for authentication functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create a config with auth enabled."""
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
        config.auth = AuthConfig(
            enabled=True,
            login_url="/login",
            username_field="user",
            password_field="pass",
            csrf_field="token",
            success_indicator=".logged-in",
            username="testuser",
            password="testpass",
        )
        return config

    def test_login_disabled(self):
        """Login should return True when auth disabled."""
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
        session = ScraperSession(config)

        result = session.login()
        assert result is True

    def test_login_missing_credentials(self):
        """Login should fail when credentials missing."""
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
        config.auth = AuthConfig(enabled=True, username="", password="")
        session = ScraperSession(config)

        result = session.login()
        assert result is False
