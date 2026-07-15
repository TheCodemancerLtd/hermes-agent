"""Chatto webchat toolset — lets the agent post/read on a Chatto server.

Registered as toolset ``chatto``.  Zero cost for users not on Chatto: the
tools are only exposed when both ``CHATTO_BASE_URL`` and either
``CHATTO_TOKEN`` or ``CHATTO_LOGIN``+``CHATTO_PASSWORD`` are set, AND the
``chattolib`` package is importable.

Each action opens a short-lived ``ChattoClient`` (mirroring the adapter's
``_standalone_send`` path) rather than sharing state with a running
gateway adapter — keeps the tool usable from any context (agent, cron,
one-shot).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_TOOLSET = "chatto"


def check_chatto_tool_requirements() -> bool:
    """Same gate as the platform adapter: env + import."""
    base_url = os.getenv("CHATTO_BASE_URL", "").strip()
    if not base_url:
        return False
    has_token = bool(os.getenv("CHATTO_TOKEN", "").strip())
    has_login = bool(
        os.getenv("CHATTO_LOGIN", "").strip()
        and os.getenv("CHATTO_PASSWORD", "").strip()
    )
    if not (has_token or has_login):
        return False
    try:
        import chattolib  # noqa: F401
        return True
    except ImportError:
        return False


async def _client_ctx():
    """Async context manager yielding a short-lived ChattoClient.

    Delegates to ``plugins.platforms.chatto.adapter._open_client`` so the
    two entry points stay in sync.
    """
    from plugins.platforms.chatto.adapter import _open_client
    base_url = os.getenv("CHATTO_BASE_URL", "").rstrip("/")
    token = os.getenv("CHATTO_TOKEN", "")
    login = os.getenv("CHATTO_LOGIN", "")
    password = os.getenv("CHATTO_PASSWORD", "")
    return await _open_client(
        base_url=base_url, token=token, login=login, password=password,
    )


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


def _ok(payload: Dict[str, Any]) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Handlers — one per action, dispatched by the top-level ``action`` field.
# ---------------------------------------------------------------------------


async def _do_post_message(args: Dict[str, Any]) -> str:
    room_id = str(args.get("room_id", "") or "").strip()
    body = str(args.get("body", "") or "")
    if not room_id or not body:
        return _err("post_message requires 'room_id' and 'body'")
    in_reply_to = str(args.get("in_reply_to", "") or "")
    thread_root_event_id = str(args.get("thread_root_event_id", "") or "")
    client = await _client_ctx()
    try:
        msg = await client.post_message(
            room_id, body,
            in_reply_to=in_reply_to,
            thread_root_event_id=thread_root_event_id,
        )
    except Exception as exc:
        return _err(f"post_message failed: {exc}")
    finally:
        try:
            await client.close()
        except Exception:
            pass
    return _ok({
        "message_id": getattr(msg, "id", ""),
        "room_id": getattr(msg, "room_id", ""),
    })


async def _do_list_rooms(args: Dict[str, Any]) -> str:
    limit = int(args.get("limit", 50) or 50)
    client = await _client_ctx()
    try:
        rooms = await client.list_rooms()
    except Exception as exc:
        return _err(f"list_rooms failed: {exc}")
    finally:
        try:
            await client.close()
        except Exception:
            pass
    out = []
    for r in rooms[:limit]:
        # RoomWithViewerState wraps the actual Room at .room; direct Room
        # objects don't have that attribute — accept both.
        underlying = getattr(r, "room", None) or r
        out.append({
            "id": getattr(underlying, "id", ""),
            "name": (
                getattr(underlying, "display_name", "")
                or getattr(underlying, "name", "")
            ),
            "kind": str(getattr(underlying, "kind", "") or ""),
        })
    return _ok({"rooms": out, "count": len(out)})


async def _do_add_reaction(args: Dict[str, Any]) -> str:
    room_id = str(args.get("room_id", "") or "").strip()
    message_event_id = str(args.get("message_event_id", "") or "").strip()
    # Chatto wants a bare shortcode (e.g. "thumbsup", "+1"); reject unicode
    # or colon-wrapped forms models sometimes emit by peeling the colons
    # here rather than round-tripping a server-side invalid_argument.
    emoji = str(args.get("emoji", "") or "").strip().strip(":")
    if not (room_id and message_event_id and emoji):
        return _err("add_reaction requires 'room_id', 'message_event_id', 'emoji'")
    client = await _client_ctx()
    try:
        await client.add_reaction(room_id, message_event_id, emoji)
    except Exception as exc:
        return _err(f"add_reaction failed: {exc}")
    finally:
        try:
            await client.close()
        except Exception:
            pass
    return _ok({"ok": True})


async def _do_start_dm(args: Dict[str, Any]) -> str:
    raw_ids = args.get("participant_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
    participant_ids = [str(x) for x in raw_ids if x]
    if not participant_ids:
        return _err("start_dm requires 'participant_ids' (list of user ULIDs)")
    client = await _client_ctx()
    try:
        room = await client.start_dm(participant_ids)
    except Exception as exc:
        return _err(f"start_dm failed: {exc}")
    finally:
        try:
            await client.close()
        except Exception:
            pass
    return _ok({
        "room_id": getattr(room, "id", ""),
        "name": (
            getattr(room, "display_name", "")
            or getattr(room, "name", "")
        ),
    })


async def _do_mark_read(args: Dict[str, Any]) -> str:
    room_id = str(args.get("room_id", "") or "").strip()
    if not room_id:
        return _err("mark_room_as_read requires 'room_id'")
    client = await _client_ctx()
    try:
        await client.mark_room_as_read(room_id)
    except Exception as exc:
        return _err(f"mark_room_as_read failed: {exc}")
    finally:
        try:
            await client.close()
        except Exception:
            pass
    return _ok({"ok": True})


_ACTIONS = {
    "post_message": _do_post_message,
    "list_rooms": _do_list_rooms,
    "add_reaction": _do_add_reaction,
    "start_dm": _do_start_dm,
    "mark_room_as_read": _do_mark_read,
}


async def chatto_handler(args: Dict[str, Any], **_kw: Any) -> str:
    """Dispatch the requested action.

    Async handler — the registry awaits it because we register with
    ``is_async=True``.
    """
    action = str((args or {}).get("action", "") or "").strip()
    fn = _ACTIONS.get(action)
    if fn is None:
        return _err(
            f"Unknown action {action!r}. Valid: {sorted(_ACTIONS)}"
        )
    try:
        return await fn(args or {})
    except Exception as exc:
        logger.exception("chatto tool action %r failed unexpectedly", action)
        return _err(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# JSON schema exposed to the model.
# ---------------------------------------------------------------------------


_SCHEMA: Dict[str, Any] = {
    "name": "chatto",
    "description": (
        "Interact with a Chatto webchat server: post messages, list rooms, "
        "add reactions, open DMs, mark rooms as read.\n\n"
        "Available actions:\n"
        "  post_message(room_id, body, in_reply_to?, thread_root_event_id?)\n"
        "  list_rooms(limit?)\n"
        "  add_reaction(room_id, message_event_id, emoji)\n"
        "  start_dm(participant_ids)\n"
        "  mark_room_as_read(room_id)\n\n"
        "Use list_rooms first to discover room IDs. Room and message IDs "
        "are ULIDs; user IDs are ULIDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_ACTIONS.keys()),
                "description": "Which Chatto action to run.",
            },
            "room_id": {
                "type": "string",
                "description": "Room ULID (required for room-scoped actions).",
            },
            "body": {
                "type": "string",
                "description": "Message body (Markdown supported). Required for post_message.",
            },
            "in_reply_to": {
                "type": "string",
                "description": "Optional message event_id to reply to (flat reply).",
            },
            "thread_root_event_id": {
                "type": "string",
                "description": "Optional thread root event_id to post inside a thread.",
            },
            "message_event_id": {
                "type": "string",
                "description": "Target message event_id for add_reaction.",
            },
            "emoji": {
                "type": "string",
                "description": (
                    "Emoji shortcode NAME as accepted by Chatto — bare, without "
                    "colons or unicode (e.g. 'thumbsup', '+1', 'fire'). "
                    "Leading/trailing colons are stripped defensively."
                ),
            },
            "participant_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "User ULIDs to open a DM with (start_dm).",
            },
            "limit": {
                "type": "integer",
                "description": "Max rooms to return (list_rooms). Default 50.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


registry.register(
    name="chatto",
    toolset=_TOOLSET,
    schema=_SCHEMA,
    handler=chatto_handler,
    check_fn=check_chatto_tool_requirements,
    requires_env=["CHATTO_BASE_URL"],
    is_async=True,
    emoji="💬",
)
