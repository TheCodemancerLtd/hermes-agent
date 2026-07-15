"""Tests for the Chatto agent toolset (tools/chatto_tool.py)."""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_fake_chattolib() -> None:
    if "chattolib" in sys.modules:
        return
    mod = types.ModuleType("chattolib")
    mod.ChattoClient = MagicMock()
    mod.ChattoError = type("ChattoError", (Exception,), {})
    mod.ChattoConnectError = type("ChattoConnectError", (mod.ChattoError,), {})
    mod.ChattoRealtimeCloseError = type(
        "ChattoRealtimeCloseError", (mod.ChattoError,), {},
    )
    mod.ChattoRealtimeError = type("ChattoRealtimeError", (mod.ChattoError,), {})
    async def _empty(client):  # pragma: no cover
        if False:
            yield None
    mod.stream_events = _empty
    sys.modules["chattolib"] = mod


_install_fake_chattolib()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "CHATTO_BASE_URL", "CHATTO_TOKEN", "CHATTO_LOGIN", "CHATTO_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def env_ready(monkeypatch):
    monkeypatch.setenv("CHATTO_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("CHATTO_TOKEN", "tok")


@pytest.fixture
def patched_open_client(monkeypatch, env_ready):
    """Patch _open_client so tool handlers get a controllable async client."""
    from plugins.platforms.chatto import adapter as chatto_adapter

    client = MagicMock()
    client.post_message = AsyncMock(
        return_value=MagicMock(id="msg_1", room_id="room_1"),
    )
    client.add_reaction = AsyncMock(return_value=None)
    client.mark_room_as_read = AsyncMock(return_value=None)
    client.list_rooms = AsyncMock(return_value=[
        MagicMock(id="r1", display_name="General", kind="ROOM_KIND_CHANNEL",
                  room=None),
    ])
    client.start_dm = AsyncMock(
        return_value=MagicMock(id="dm_1", display_name="Alice"),
    )
    client.close = AsyncMock(return_value=None)

    async def _fake_open(**_kw):
        return client

    monkeypatch.setattr(chatto_adapter, "_open_client", _fake_open)
    return client


class TestChattoToolCheckRequirements:
    def test_needs_env_and_import(self, monkeypatch):
        from tools.chatto_tool import check_chatto_tool_requirements
        assert check_chatto_tool_requirements() is False
        monkeypatch.setenv("CHATTO_BASE_URL", "https://x")
        assert check_chatto_tool_requirements() is False
        monkeypatch.setenv("CHATTO_TOKEN", "t")
        assert check_chatto_tool_requirements() is True


class TestChattoToolDispatch:
    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self, env_ready):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({"action": "nope"}))
        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_post_message_requires_room_and_body(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        r1 = json.loads(await chatto_handler({"action": "post_message"}))
        assert "error" in r1
        r2 = json.loads(await chatto_handler({"action": "post_message", "room_id": "r1"}))
        assert "error" in r2
        patched_open_client.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_message_happy_path(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({
            "action": "post_message", "room_id": "room_1", "body": "hi",
        }))
        assert result == {"message_id": "msg_1", "room_id": "room_1"}
        patched_open_client.post_message.assert_awaited_once()
        args, kwargs = patched_open_client.post_message.call_args
        assert args == ("room_1", "hi")
        # Defaults: no reply/thread anchors.
        assert kwargs.get("in_reply_to", "") == ""
        assert kwargs.get("thread_root_event_id", "") == ""
        patched_open_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_post_message_forwards_reply_anchor(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        await chatto_handler({
            "action": "post_message", "room_id": "r", "body": "b",
            "in_reply_to": "parent_ev",
        })
        _, kwargs = patched_open_client.post_message.call_args
        assert kwargs["in_reply_to"] == "parent_ev"

    @pytest.mark.asyncio
    async def test_add_reaction_requires_all_fields(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({
            "action": "add_reaction", "room_id": "r1",
        }))
        assert "error" in result
        patched_open_client.add_reaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_reaction_happy_path(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({
            "action": "add_reaction", "room_id": "r1",
            "message_event_id": "ev_1", "emoji": "👍",
        }))
        assert result == {"ok": True}
        patched_open_client.add_reaction.assert_awaited_once_with("r1", "ev_1", "👍")

    @pytest.mark.asyncio
    async def test_start_dm_normalizes_csv_participants(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({
            "action": "start_dm", "participant_ids": "u1, u2, u3",
        }))
        assert result["room_id"] == "dm_1"
        patched_open_client.start_dm.assert_awaited_once_with(["u1", "u2", "u3"])

    @pytest.mark.asyncio
    async def test_mark_room_as_read_happy_path(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({
            "action": "mark_room_as_read", "room_id": "r_1",
        }))
        assert result == {"ok": True}
        patched_open_client.mark_room_as_read.assert_awaited_once_with("r_1")

    @pytest.mark.asyncio
    async def test_list_rooms_returns_summary(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        result = json.loads(await chatto_handler({"action": "list_rooms"}))
        assert result["count"] == 1
        assert result["rooms"][0]["id"] == "r1"
        assert result["rooms"][0]["name"] == "General"

    @pytest.mark.asyncio
    async def test_client_error_is_wrapped(self, patched_open_client):
        from tools.chatto_tool import chatto_handler
        patched_open_client.post_message = AsyncMock(side_effect=RuntimeError("boom"))
        result = json.loads(await chatto_handler({
            "action": "post_message", "room_id": "r", "body": "b",
        }))
        assert "error" in result
        assert "boom" in result["error"]
