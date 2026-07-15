"""Tests for the Chatto platform adapter (plugins/platforms/chatto)."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake chattolib module — installed once for the whole test module so imports
# inside adapter code resolve without needing the real package on PYTHONPATH.
# ---------------------------------------------------------------------------

_CHATTO_ENV_VARS = (
    "CHATTO_BASE_URL", "CHATTO_TOKEN", "CHATTO_LOGIN", "CHATTO_PASSWORD",
    "CHATTO_ALLOWED_ROOMS", "CHATTO_ALLOWED_USERS", "CHATTO_ALLOW_ALL_USERS",
    "CHATTO_HOME_ROOM", "CHATTO_HOME_ROOM_NAME", "CHATTO_REPLY_MODE",
)


class _FakeChattoError(Exception):
    pass


class _FakeChattoConnectError(_FakeChattoError):
    def __init__(self, code: str, message: str, **_kw: Any) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class _FakeChattoRealtimeCloseError(_FakeChattoError):
    def __init__(self, code: str, message: str, *, reconnect: bool = False, retry_after_ms: int = 0) -> None:
        self.code = code
        self.message = message
        self.reconnect = reconnect
        self.retry_after_ms = retry_after_ms
        super().__init__(f"{code}: {message}")


class _FakeChattoRealtimeError(_FakeChattoError):
    def __init__(self, code: str, message: str, *, fatal: bool = False) -> None:
        self.code = code
        self.message = message
        self.fatal = fatal
        super().__init__(f"{code}: {message}")


class _FakeChattoClient:
    """Stand-in for chattolib.ChattoClient; hand-configured per test."""


def _install_fake_chattolib() -> None:
    """Register a stub `chattolib` in sys.modules if the real one is absent."""
    if "chattolib" in sys.modules:
        return
    mod = types.ModuleType("chattolib")
    mod.ChattoClient = _FakeChattoClient
    mod.ChattoError = _FakeChattoError
    mod.ChattoConnectError = _FakeChattoConnectError
    mod.ChattoRealtimeCloseError = _FakeChattoRealtimeCloseError
    mod.ChattoRealtimeError = _FakeChattoRealtimeError
    mod.stream_events = lambda client: _empty_async_iter()
    sys.modules["chattolib"] = mod


async def _empty_async_iter():
    if False:
        yield None  # pragma: no cover — never yields


_install_fake_chattolib()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with a clean CHATTO_* env."""
    for var in _CHATTO_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def platform_config():
    from gateway.config import PlatformConfig
    return PlatformConfig(enabled=True, extra={})


@pytest.fixture
def make_adapter(platform_config):
    from plugins.platforms.chatto.adapter import ChattoAdapter

    def _factory(**overrides):
        # Env overrides applied before construction so __init__ picks them up.
        for k, v in overrides.items():
            os.environ[k] = v
        return ChattoAdapter(platform_config)

    yield _factory


# ---------------------------------------------------------------------------
# Config bridge / is_connected / requirements
# ---------------------------------------------------------------------------


class TestChattoConfigBridge:
    def test_yaml_translates_scalar_and_list_keys_to_env(self, monkeypatch):
        from plugins.platforms.chatto.adapter import _apply_yaml_config
        _apply_yaml_config({}, {
            "base_url": "https://chat.example.com",
            "reply_mode": "THREAD",
            "allowed_rooms": ["r1", "r2"],
            "home_room": "r3",
            "home_room_name": "General",
        })
        assert os.environ["CHATTO_BASE_URL"] == "https://chat.example.com"
        assert os.environ["CHATTO_REPLY_MODE"] == "thread"
        assert os.environ["CHATTO_ALLOWED_ROOMS"] == "r1,r2"
        assert os.environ["CHATTO_HOME_ROOM"] == "r3"
        assert os.environ["CHATTO_HOME_ROOM_NAME"] == "General"

    def test_yaml_does_not_override_existing_env(self, monkeypatch):
        from plugins.platforms.chatto.adapter import _apply_yaml_config
        monkeypatch.setenv("CHATTO_REPLY_MODE", "off")
        _apply_yaml_config({}, {"reply_mode": "thread"})
        assert os.environ["CHATTO_REPLY_MODE"] == "off"


class TestChattoIsConnected:
    def test_needs_base_url_plus_token_or_login(self, monkeypatch):
        # Provide a stub for hermes_cli.gateway.get_env_value so the probe
        # uses actual env vars rather than any indirection.
        import hermes_cli.gateway as gw
        monkeypatch.setattr(gw, "get_env_value", lambda k, default="": os.getenv(k, default))
        from plugins.platforms.chatto.adapter import _is_connected
        assert _is_connected(None) is False
        monkeypatch.setenv("CHATTO_BASE_URL", "https://chat.example.com")
        assert _is_connected(None) is False  # no auth yet
        monkeypatch.setenv("CHATTO_TOKEN", "abc")
        assert _is_connected(None) is True
        monkeypatch.delenv("CHATTO_TOKEN")
        monkeypatch.setenv("CHATTO_LOGIN", "u")
        monkeypatch.setenv("CHATTO_PASSWORD", "p")
        assert _is_connected(None) is True

    def test_check_requirements_env_and_import(self, monkeypatch):
        from plugins.platforms.chatto.adapter import check_chatto_requirements
        assert check_chatto_requirements() is False  # nothing set
        monkeypatch.setenv("CHATTO_BASE_URL", "https://x")
        assert check_chatto_requirements() is False  # no creds
        monkeypatch.setenv("CHATTO_TOKEN", "t")
        assert check_chatto_requirements() is True  # env + stub chattolib import


# ---------------------------------------------------------------------------
# Adapter init
# ---------------------------------------------------------------------------


class TestChattoAdapterInit:
    def test_reads_env_vars_and_defaults(self, make_adapter):
        adapter = make_adapter(
            CHATTO_BASE_URL="https://chat.example.com/",
            CHATTO_TOKEN="tok",
        )
        assert adapter._base_url == "https://chat.example.com"  # trailing / stripped
        assert adapter._token == "tok"
        assert adapter._reply_mode == "reply"  # default
        assert adapter._allowed_rooms == set()

    def test_allowed_rooms_parsed_from_csv(self, make_adapter):
        adapter = make_adapter(
            CHATTO_BASE_URL="https://x",
            CHATTO_TOKEN="t",
            CHATTO_ALLOWED_ROOMS=" r1 , r2 ,",
        )
        assert adapter._allowed_rooms == {"r1", "r2"}

    def test_unknown_reply_mode_falls_back(self, make_adapter):
        adapter = make_adapter(
            CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t",
            CHATTO_REPLY_MODE="frobnicate",
        )
        assert adapter._reply_mode == "reply"


# ---------------------------------------------------------------------------
# send() — outbound path
# ---------------------------------------------------------------------------


class TestChattoAdapterSend:
    async def _make_ready(self, make_adapter, **env):
        adapter = make_adapter(
            CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t", **env,
        )
        adapter._client = MagicMock()
        adapter._client.post_message = AsyncMock(
            return_value=MagicMock(id="msg_1"),
        )
        return adapter

    @pytest.mark.asyncio
    async def test_returns_error_when_client_not_connected(self, make_adapter):
        adapter = make_adapter(CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t")
        result = await adapter.send("room_1", "hello")
        assert result.success is False
        assert "not connected" in (result.error or "")

    @pytest.mark.asyncio
    async def test_empty_content_returns_success_without_call(self, make_adapter):
        adapter = await self._make_ready(make_adapter)
        result = await adapter.send("room_1", "")
        assert result.success is True
        adapter._client.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_mode_reply_sets_in_reply_to(self, make_adapter):
        adapter = await self._make_ready(make_adapter, CHATTO_REPLY_MODE="reply")
        await adapter.send("room_1", "hi", reply_to="parent_ev")
        _, kwargs = adapter._client.post_message.call_args
        assert kwargs.get("in_reply_to") == "parent_ev"
        assert kwargs.get("thread_root_event_id", "") == ""

    @pytest.mark.asyncio
    async def test_reply_mode_thread_sets_thread_root(self, make_adapter):
        adapter = await self._make_ready(make_adapter, CHATTO_REPLY_MODE="thread")
        await adapter.send("room_1", "hi", reply_to="parent_ev")
        _, kwargs = adapter._client.post_message.call_args
        assert kwargs.get("thread_root_event_id") == "parent_ev"
        assert kwargs.get("in_reply_to", "") == ""

    @pytest.mark.asyncio
    async def test_attachment_ids_from_metadata(self, make_adapter):
        adapter = await self._make_ready(make_adapter)
        await adapter.send(
            "room_1", "hi",
            metadata={"attachment_asset_ids": ["a1", "a2"]},
        )
        _, kwargs = adapter._client.post_message.call_args
        assert kwargs.get("attachment_asset_ids") == ["a1", "a2"]

    @pytest.mark.asyncio
    async def test_chunks_long_content_and_returns_continuation_ids(self, make_adapter):
        adapter = await self._make_ready(make_adapter)
        # Simulate two IDs coming back so we can check continuation.
        adapter._client.post_message = AsyncMock(
            side_effect=[MagicMock(id="msg_1"), MagicMock(id="msg_2")],
        )
        # Force chunking with 5001 chars (> MAX_MESSAGE_LENGTH=4000).
        content = "x" * 5001
        result = await adapter.send("room_1", content)
        assert result.success is True
        assert adapter._client.post_message.call_count == 2
        assert result.message_id == "msg_2"
        assert result.continuation_message_ids == ("msg_2",)

    @pytest.mark.asyncio
    async def test_send_wraps_exception_as_error(self, make_adapter):
        adapter = await self._make_ready(make_adapter)
        adapter._client.post_message = AsyncMock(side_effect=RuntimeError("boom"))
        result = await adapter.send("room_1", "hi")
        assert result.success is False
        assert "boom" in (result.error or "")


# ---------------------------------------------------------------------------
# Inbound event dispatch
# ---------------------------------------------------------------------------


def _make_realtime_event(
    *,
    kind: str = "message_posted",
    room_id: str = "room_1",
    event_id: str = "ev_1",
    actor_id: str = "user_1",
):
    """Build a minimal RealtimeEvent-shaped MagicMock."""
    payload = MagicMock()
    payload.room_id = room_id
    payload.message_event_id = event_id
    payload.thread_root_event_id = ""
    event = MagicMock()
    event.kind = kind
    event.payload = payload
    event.actor_id = actor_id
    return event


class TestChattoInboundDispatch:
    @pytest.fixture
    def adapter(self, make_adapter):
        adapter = make_adapter(CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t")
        adapter._bot_user_id = "bot_1"
        adapter._client = MagicMock()
        adapter._client.get_message = AsyncMock()
        adapter._client.get_user = AsyncMock(return_value=None)
        adapter._client.get_room = AsyncMock(return_value=None)
        # Intercept dispatch above the full base pipeline (extract_media etc.).
        adapter.handle_message = AsyncMock()
        # Keep a convenience alias so existing assertions still read cleanly.
        adapter._message_handler = adapter.handle_message
        return adapter

    @pytest.mark.asyncio
    async def test_skips_self_echo(self, adapter):
        event = _make_realtime_event(actor_id="bot_1")
        await adapter._handle_realtime_event(event)
        adapter._client.get_message.assert_not_called()
        adapter._message_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_room_whitelist_filters_non_allowed(self, adapter):
        adapter._allowed_rooms = {"other_room"}
        event = _make_realtime_event(room_id="room_1", actor_id="alice")
        await adapter._handle_realtime_event(event)
        adapter._client.get_message.assert_not_called()
        adapter._message_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dedup_skips_duplicate_event(self, adapter):
        message = MagicMock()
        message.body = "hello"
        message.actor_id = "alice"
        message.deleted_at = None
        message.in_reply_to = ""
        message.thread_root_event_id = ""
        adapter._client.get_message = AsyncMock(return_value=message)

        event = _make_realtime_event(event_id="ev_dup", actor_id="alice")
        await adapter._handle_realtime_event(event)
        await adapter._handle_realtime_event(event)  # duplicate
        assert adapter._client.get_message.call_count == 1
        assert adapter._message_handler.call_count == 1

    @pytest.mark.asyncio
    async def test_hydration_failure_swallowed(self, adapter):
        adapter._client.get_message = AsyncMock(side_effect=RuntimeError("gone"))
        event = _make_realtime_event(actor_id="alice")
        await adapter._handle_realtime_event(event)  # should not raise
        adapter._message_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_body_skipped(self, adapter):
        message = MagicMock()
        message.body = ""
        message.actor_id = "alice"
        message.deleted_at = None
        adapter._client.get_message = AsyncMock(return_value=message)
        event = _make_realtime_event(actor_id="alice")
        await adapter._handle_realtime_event(event)
        adapter._message_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_posted_dispatches_message_event(self, adapter):
        message = MagicMock()
        message.body = "hello world"
        message.actor_id = "alice"
        message.deleted_at = None
        message.in_reply_to = "parent_ev"
        message.thread_root_event_id = ""
        adapter._client.get_message = AsyncMock(return_value=message)
        event = _make_realtime_event(actor_id="alice", event_id="ev_1")

        await adapter._handle_realtime_event(event)

        assert adapter._message_handler.call_count == 1
        (dispatched,), _ = adapter._message_handler.call_args
        assert dispatched.text == "hello world"
        assert dispatched.message_id == "ev_1"
        assert dispatched.reply_to_message_id == "parent_ev"
        assert dispatched.source.chat_id == "room_1"
        assert dispatched.source.user_id == "alice"


# ---------------------------------------------------------------------------
# Reconnect loop control flow
# ---------------------------------------------------------------------------


class TestChattoReconnect:
    @pytest.mark.asyncio
    async def test_close_with_reconnect_false_stops_loop(
        self, monkeypatch, make_adapter,
    ):
        """A ChattoRealtimeCloseError(reconnect=False) must terminate the loop."""
        adapter = make_adapter(CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t")
        adapter._client = MagicMock()

        import chattolib

        call_count = 0

        async def fake_stream(_client):
            nonlocal call_count
            call_count += 1
            raise chattolib.ChattoRealtimeCloseError(
                "server_shutdown", "bye", reconnect=False, retry_after_ms=0,
            )
            yield None  # unreachable — keeps this an async generator

        monkeypatch.setattr(chattolib, "stream_events", fake_stream)

        # Run the loop with a timeout — reconnect=False must return promptly.
        await asyncio.wait_for(adapter._run_event_stream(), timeout=2.0)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_fatal_realtime_error_stops_loop(self, monkeypatch, make_adapter):
        adapter = make_adapter(CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t")
        adapter._client = MagicMock()

        import chattolib

        async def fake_stream(_client):
            raise chattolib.ChattoRealtimeError("protocol_error", "bad", fatal=True)
            yield None

        monkeypatch.setattr(chattolib, "stream_events", fake_stream)
        await asyncio.wait_for(adapter._run_event_stream(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_closing_flag_short_circuits(self, make_adapter):
        adapter = make_adapter(CHATTO_BASE_URL="https://x", CHATTO_TOKEN="t")
        adapter._client = MagicMock()
        adapter._closing = True
        # Even with a broken stream, _closing=True must cause an immediate return.
        await asyncio.wait_for(adapter._run_event_stream(), timeout=1.0)
