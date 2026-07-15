"""Chatto webchat gateway adapter.

Connects to a Chatto server (self-hosted or the hosted chatto.run instance)
via the async ``chattolib`` client:

* ConnectRPC (JSON over HTTP) for request/response RPCs — auth, post_message,
  reactions, room queries, attachment uploads, etc.
* WebSocket realtime stream (``chattolib.stream_events``) for inbound events
  — message_posted, message_edited, reaction_added, typing, presence.

Environment variables:
    CHATTO_TOKEN               Bearer token (required unless CHATTO_LOGIN/PASSWORD set)
    CHATTO_BASE_URL            Server base URL (e.g. https://chat.example.com)
    CHATTO_LOGIN               Alternative to CHATTO_TOKEN: username for password login
    CHATTO_PASSWORD            Password used with CHATTO_LOGIN
    CHATTO_ALLOWED_USERS       Comma-separated user ULIDs allowed to trigger the bot
    CHATTO_HOME_ROOM           Default room ULID for cron / notification delivery
    CHATTO_ALLOWED_ROOMS       Whitelist of room ULIDs the bot will respond in
    CHATTO_REPLY_MODE          thread | reply | off  (default: reply)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.helpers import MessageDeduplicator

logger = logging.getLogger(__name__)


# Chatto's server-side default room-message body cap.  Kept conservative
# (matching Mattermost's readable threshold) rather than the raw protobuf
# max, so long agent responses chunk cleanly.
MAX_MESSAGE_LENGTH = 4000

# Realtime reconnect parameters (exponential backoff with jitter).  Used as
# a fallback when the server closes without a ``retry_after_ms`` hint.
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2

# Reply-mode env values.
_REPLY_MODE_THREAD = "thread"
_REPLY_MODE_REPLY = "reply"
_REPLY_MODE_OFF = "off"
_REPLY_MODE_DEFAULT = _REPLY_MODE_REPLY

# chattolib PresenceStatus.value for connect() health check.
_PRESENCE_ONLINE = "PRESENCE_STATUS_ONLINE"


def check_chatto_requirements() -> bool:
    """Return True if the Chatto adapter has enough config + deps to run."""
    base_url = os.getenv("CHATTO_BASE_URL", "").strip()
    if not base_url:
        logger.debug("Chatto: CHATTO_BASE_URL not set")
        return False
    has_token = bool(os.getenv("CHATTO_TOKEN", "").strip())
    has_login = bool(
        os.getenv("CHATTO_LOGIN", "").strip()
        and os.getenv("CHATTO_PASSWORD", "").strip()
    )
    if not (has_token or has_login):
        logger.debug("Chatto: neither CHATTO_TOKEN nor CHATTO_LOGIN/PASSWORD set")
        return False
    try:
        import chattolib  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "Chatto: chattolib not installed — `pip install hermes-agent[chatto]`"
        )
        return False


async def _open_client(
    *,
    base_url: str,
    token: str,
    login: str,
    password: str,
) -> Any:
    """Return a connected ``ChattoClient`` using token or login/password."""
    import chattolib

    if token:
        return chattolib.ChattoClient(token=token, base_url=base_url)
    return await chattolib.ChattoClient.login(login, password, base_url=base_url)


class ChattoAdapter(BasePlatformAdapter):
    """Gateway adapter for Chatto webchat servers."""

    splits_long_messages = True
    supports_code_blocks = True
    supports_async_delivery = True
    typed_command_prefix = "/"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("chatto"))

        self._base_url: str = (
            config.extra.get("base_url", "")
            or os.getenv("CHATTO_BASE_URL", "")
        ).rstrip("/")
        self._token: str = config.token or os.getenv("CHATTO_TOKEN", "")
        self._login: str = os.getenv("CHATTO_LOGIN", "")
        self._password: str = os.getenv("CHATTO_PASSWORD", "")

        self._client: Any = None
        self._stream_task: Optional[asyncio.Task] = None
        self._closing = False

        self._bot_user_id: str = ""
        self._bot_username: str = ""
        self._bot_display_name: str = ""

        self._reply_mode: str = (
            config.extra.get("reply_mode", "")
            or os.getenv("CHATTO_REPLY_MODE", "")
            or _REPLY_MODE_DEFAULT
        ).lower()
        if self._reply_mode not in (_REPLY_MODE_THREAD, _REPLY_MODE_REPLY, _REPLY_MODE_OFF):
            logger.warning(
                "Chatto: unknown CHATTO_REPLY_MODE=%r, falling back to %r",
                self._reply_mode, _REPLY_MODE_DEFAULT,
            )
            self._reply_mode = _REPLY_MODE_DEFAULT

        allowed_rooms = os.getenv("CHATTO_ALLOWED_ROOMS", "").strip()
        self._allowed_rooms: set[str] = (
            {r.strip() for r in allowed_rooms.split(",") if r.strip()}
            if allowed_rooms else set()
        )

        # Lightweight per-room cache for get_chat_info (populated lazily).
        self._room_info_cache: Dict[str, Dict[str, Any]] = {}

        # Dedup — chattolib may redeliver events across reconnects.
        self._dedup = MessageDeduplicator()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Log into Chatto and start the realtime event stream."""
        if not self._base_url:
            logger.error("Chatto: CHATTO_BASE_URL not configured")
            return False
        if not self._token and not (self._login and self._password):
            logger.error(
                "Chatto: neither CHATTO_TOKEN nor CHATTO_LOGIN/PASSWORD is set"
            )
            return False

        self._closing = False

        try:
            self._client = await _open_client(
                base_url=self._base_url,
                token=self._token,
                login=self._login,
                password=self._password,
            )
        except Exception as exc:
            logger.error("Chatto: failed to construct client: %s", exc)
            self._set_fatal_error(
                "chatto_login_failed",
                f"Chatto login failed: {exc}",
                retryable=False,
            )
            return False

        # Fetch bot identity — also verifies auth.
        try:
            me = await self._client.me()
        except Exception as exc:
            logger.error("Chatto: failed to fetch viewer profile: %s", exc)
            self._set_fatal_error(
                "chatto_auth_failed",
                f"Chatto auth failed: {exc}",
                retryable=False,
            )
            try:
                await self._client.close()
            finally:
                self._client = None
            return False

        self._bot_user_id = getattr(me, "id", "") or ""
        self._bot_username = getattr(me, "login", "") or ""
        self._bot_display_name = getattr(me, "display_name", "") or self._bot_username
        logger.info(
            "Chatto: authenticated as @%s (%s) on %s",
            self._bot_username or "?", self._bot_user_id or "?", self._base_url,
        )

        # Best-effort presence broadcast (non-fatal on failure).
        try:
            await self._client.update_presence(_PRESENCE_ONLINE)
        except Exception:
            logger.debug("Chatto: update_presence not supported / failed (non-fatal)")

        self._stream_task = asyncio.create_task(
            self._run_event_stream(), name="chatto-event-stream",
        )
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Stop the event stream and close the chattolib client."""
        self._closing = True

        task = self._stream_task
        self._stream_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.close()
            except Exception:  # pragma: no cover — best-effort teardown
                logger.exception("Chatto: error closing client")

        logger.info("Chatto: disconnected")

    # ------------------------------------------------------------------
    # Outbound send
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Post a message (or multiple chunks) to a Chatto room."""
        if self._client is None:
            return SendResult(success=False, error="Chatto client not connected")
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)

        # Only the FIRST chunk carries the reply/thread anchor; subsequent
        # chunks are flat follow-ups so we don't end up threading N times.
        first_kwargs: Dict[str, Any] = {}
        if reply_to and self._reply_mode == _REPLY_MODE_REPLY:
            first_kwargs["in_reply_to"] = reply_to
        elif reply_to and self._reply_mode == _REPLY_MODE_THREAD:
            first_kwargs["thread_root_event_id"] = reply_to

        # Attachment asset IDs may be pre-uploaded by callers (e.g. the
        # chatto toolset uploads via client.upload_attachment first, then
        # passes the resulting IDs through metadata).
        attachment_asset_ids: List[str] = []
        if metadata:
            raw_ids = metadata.get("attachment_asset_ids") or []
            if isinstance(raw_ids, (list, tuple)):
                attachment_asset_ids = [str(x) for x in raw_ids if x]

        last_id: Optional[str] = None
        continuation_ids: List[str] = []
        for idx, chunk in enumerate(chunks):
            kwargs = dict(first_kwargs) if idx == 0 else {}
            if idx == 0 and attachment_asset_ids:
                kwargs["attachment_asset_ids"] = attachment_asset_ids
            try:
                msg = await self._client.post_message(chat_id, chunk, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Chatto: post_message failed on chunk %d/%d: %s",
                    idx + 1, len(chunks), exc,
                )
                # Return partial success if we got at least one chunk out;
                # otherwise a clean failure.
                if last_id is None:
                    return SendResult(
                        success=False,
                        error=str(exc),
                        retryable=self._is_retryable(exc),
                    )
                return SendResult(
                    success=True,
                    message_id=last_id,
                    continuation_message_ids=tuple(continuation_ids),
                    error=f"partial: {exc}",
                )
            mid = getattr(msg, "id", "") or ""
            if last_id is not None:
                continuation_ids.append(mid)
            last_id = mid

        return SendResult(
            success=True,
            message_id=last_id,
            continuation_message_ids=tuple(continuation_ids),
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True for transient chattolib errors worth retrying."""
        try:
            from chattolib import ChattoConnectError
        except ImportError:
            return False
        if isinstance(exc, ChattoConnectError):
            # Connect protocol codes that are transient.  See
            # https://connectrpc.com/docs/protocol#error-codes.
            transient = {"unavailable", "deadline_exceeded", "resource_exhausted"}
            return (exc.code or "").lower() in transient
        return False

    # ------------------------------------------------------------------
    # Chat info (abstract in BasePlatformAdapter)
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return the Chatto room's display name and type."""
        cached = self._room_info_cache.get(chat_id)
        if cached is not None:
            return cached
        info: Dict[str, Any] = {"name": chat_id, "type": "channel"}
        if self._client is None:
            return info
        try:
            room = await self._client.get_room(chat_id)
        except Exception as exc:
            logger.debug("Chatto: get_room(%s) failed: %s", chat_id, exc)
            return info
        if room is None:
            return info
        # `room` is a RoomWithViewerState — the actual Room lives at .room.
        # Fall back to `room` itself when chattolib returns a bare Room.
        underlying = getattr(room, "room", None) or room
        display = getattr(underlying, "display_name", "") or getattr(underlying, "name", "") or chat_id
        kind = str(getattr(underlying, "kind", "") or "")
        chat_type = "dm" if kind.endswith("DM") else "channel"
        info = {"name": display, "type": chat_type}
        self._room_info_cache[chat_id] = info
        return info

    # ------------------------------------------------------------------
    # Realtime event stream (inbound)
    # ------------------------------------------------------------------

    async def _run_event_stream(self) -> None:
        """Consume chattolib.stream_events with server-directed reconnect."""
        try:
            from chattolib import (
                ChattoRealtimeCloseError,
                ChattoRealtimeError,
                stream_events,
            )
        except ImportError:
            logger.error("Chatto: chattolib[realtime] not installed — event stream disabled")
            return

        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                async for event in stream_events(self._client):
                    if self._closing:
                        return
                    await self._handle_realtime_event(event)
                # Iterator exited cleanly — treat as a normal close and reconnect
                # with the local backoff (no server hint available).
                logger.info("Chatto: realtime stream ended, reconnecting")
            except asyncio.CancelledError:
                return
            except ChattoRealtimeCloseError as exc:
                if not exc.reconnect:
                    logger.error(
                        "Chatto: realtime closed by server (%s: %s), not reconnecting",
                        exc.code, exc.message,
                    )
                    return
                wait = max(exc.retry_after_ms / 1000.0, _RECONNECT_BASE_DELAY)
                logger.warning(
                    "Chatto: realtime closed by server (%s), reconnecting in %.1fs",
                    exc.code, wait,
                )
                delay = _RECONNECT_BASE_DELAY  # server hint supersedes local backoff
                await self._sleep_interruptible(wait)
                continue
            except ChattoRealtimeError as exc:
                if getattr(exc, "fatal", False):
                    logger.error("Chatto: fatal realtime error (%s): %s", exc.code, exc.message)
                    return
                logger.warning(
                    "Chatto: realtime error (%s: %s), reconnecting in %.1fs",
                    exc.code, exc.message, delay,
                )
            except Exception as exc:
                logger.warning(
                    "Chatto: unexpected realtime error: %s, reconnecting in %.1fs",
                    exc, delay,
                )

            if self._closing:
                return

            jitter = delay * _RECONNECT_JITTER * random.random()
            await self._sleep_interruptible(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in short slices so disconnect() cancels promptly."""
        end = asyncio.get_running_loop().time() + seconds
        while not self._closing:
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.5))

    async def _handle_realtime_event(self, event: Any) -> None:
        """Dispatch a decoded RealtimeEvent to the right handler."""
        kind = getattr(event, "kind", "") or ""
        # First-party MVP scope: only message_posted is bridged to the agent.
        # Other kinds (message_edited, reaction_added, presence_changed, …)
        # are recognised for logging but not yet forwarded — extension point.
        if kind == "message_posted":
            await self._dispatch_message_posted(event)
        elif kind:
            logger.debug("Chatto: ignoring event kind=%s (not yet bridged)", kind)

    async def _dispatch_message_posted(self, event: Any) -> None:
        """Hydrate a message_posted event and dispatch as MessageEvent."""
        payload = getattr(event, "payload", None)
        if payload is None:
            return
        room_id = getattr(payload, "room_id", "") or ""
        event_id = getattr(payload, "message_event_id", "") or ""
        if not room_id or not event_id:
            return

        # Self-echo filter — the actor_id on the envelope is authoritative
        # (chattolib does NOT filter this itself; see chatto-bridge notes).
        actor_id = getattr(event, "actor_id", None) or ""
        if actor_id and actor_id == self._bot_user_id:
            return

        # Room whitelist.
        if self._allowed_rooms and room_id not in self._allowed_rooms:
            logger.debug("Chatto: ignoring message in non-allowed room: %s", room_id)
            return

        # Dedup (retries + reconnect redeliveries).
        if self._dedup.is_duplicate(event_id):
            return

        # Hydrate via GetMessage — the realtime payload is an invalidation
        # signal; the concrete body lives in the RPC response.
        if self._client is None:
            return
        try:
            message = await self._client.get_message(room_id, event_id)
        except Exception as exc:
            logger.debug("Chatto: get_message(%s, %s) failed: %s", room_id, event_id, exc)
            return
        if message is None or getattr(message, "deleted_at", None):
            return

        body = getattr(message, "body", "") or ""
        # Skip attachment-only messages for the MVP (mirrors chatto-bridge).
        if not body.strip():
            return

        # Resolve sender display name (best-effort — falls back to actor_id).
        sender_id = getattr(message, "actor_id", "") or actor_id
        sender_name = await self._resolve_display_name(sender_id)

        # Determine chat_type from the room kind.
        room_info = await self.get_chat_info(room_id)
        chat_type = room_info.get("type", "channel")

        # Thread anchoring — if the incoming message is inside a thread, we
        # keep that thread by default; otherwise leave thread_id unset so
        # replies land at the root.
        thread_id = (
            getattr(message, "thread_root_event_id", "") or None
        )
        if not thread_id and self._reply_mode == _REPLY_MODE_THREAD and chat_type != "dm":
            thread_id = event_id

        msg_type = MessageType.COMMAND if body.lstrip().startswith("/") else MessageType.TEXT

        source = self.build_source(
            chat_id=room_id,
            chat_name=room_info.get("name"),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            thread_id=thread_id,
            message_id=event_id,
        )

        msg_event = MessageEvent(
            text=body.lstrip() if msg_type == MessageType.COMMAND else body,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=event_id,
            reply_to_message_id=(getattr(message, "in_reply_to", "") or None),
        )

        await self.handle_message(msg_event)

    async def _resolve_display_name(self, user_id: str) -> str:
        """Look up a user's display name, falling back to their ID on failure."""
        if not user_id:
            return ""
        if self._client is None:
            return user_id
        try:
            user = await self._client.get_user(user_id)
        except Exception:
            return user_id
        if user is None:
            return user_id
        return (
            getattr(user, "display_name", "")
            or getattr(user, "login", "")
            or user_id
        )


# ---------------------------------------------------------------------------
# Cron / out-of-process delivery
# ---------------------------------------------------------------------------


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    reply_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SendResult:
    """Deliver a message to Chatto without a running gateway adapter.

    Used by cron / scheduled routines that run out-of-process.  Creates a
    short-lived chattolib client, posts, and closes.
    """
    base_url = (
        (getattr(pconfig, "extra", None) or {}).get("base_url", "")
        or os.getenv("CHATTO_BASE_URL", "")
    ).rstrip("/")
    token = getattr(pconfig, "token", "") or os.getenv("CHATTO_TOKEN", "")
    login = os.getenv("CHATTO_LOGIN", "")
    password = os.getenv("CHATTO_PASSWORD", "")
    if not base_url or not (token or (login and password)):
        return SendResult(success=False, error="Chatto: base URL or credentials missing")

    try:
        client = await _open_client(
            base_url=base_url, token=token, login=login, password=password,
        )
    except Exception as exc:
        return SendResult(success=False, error=f"Chatto login failed: {exc}")

    try:
        kwargs: Dict[str, Any] = {}
        if reply_to:
            kwargs["in_reply_to"] = reply_to
        if metadata and metadata.get("attachment_asset_ids"):
            kwargs["attachment_asset_ids"] = list(metadata["attachment_asset_ids"])
        try:
            posted = await client.post_message(chat_id, message, **kwargs)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=getattr(posted, "id", "") or None)
    finally:
        try:
            await client.close()
        except Exception:
            logger.debug("Chatto standalone: error closing short-lived client")


# ---------------------------------------------------------------------------
# YAML → env config bridge
# ---------------------------------------------------------------------------


def _apply_yaml_config(yaml_cfg: dict, chatto_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``chatto:`` keys into env vars.

    Env vars take precedence over YAML — every assignment is guarded by
    ``not os.getenv(...)`` so an explicit env survives a YAML update.
    """
    base_url = chatto_cfg.get("base_url") or chatto_cfg.get("url")
    if base_url and not os.getenv("CHATTO_BASE_URL"):
        os.environ["CHATTO_BASE_URL"] = str(base_url)

    if "reply_mode" in chatto_cfg and not os.getenv("CHATTO_REPLY_MODE"):
        os.environ["CHATTO_REPLY_MODE"] = str(chatto_cfg["reply_mode"]).lower()

    allowed_rooms = chatto_cfg.get("allowed_rooms")
    if allowed_rooms is not None and not os.getenv("CHATTO_ALLOWED_ROOMS"):
        if isinstance(allowed_rooms, list):
            allowed_rooms = ",".join(str(v) for v in allowed_rooms)
        os.environ["CHATTO_ALLOWED_ROOMS"] = str(allowed_rooms)

    home_room = chatto_cfg.get("home_room")
    if home_room and not os.getenv("CHATTO_HOME_ROOM"):
        os.environ["CHATTO_HOME_ROOM"] = str(home_room)

    home_room_name = chatto_cfg.get("home_room_name")
    if home_room_name and not os.getenv("CHATTO_HOME_ROOM_NAME"):
        os.environ["CHATTO_HOME_ROOM_NAME"] = str(home_room_name)

    return None


# ---------------------------------------------------------------------------
# is_connected probe
# ---------------------------------------------------------------------------


def _is_connected(config) -> bool:
    """Chatto is connected when BASE_URL + (TOKEN or LOGIN/PASSWORD) are set."""
    import hermes_cli.gateway as gateway_mod
    base_url = (gateway_mod.get_env_value("CHATTO_BASE_URL") or "").strip()
    token = (gateway_mod.get_env_value("CHATTO_TOKEN") or "").strip()
    login = (gateway_mod.get_env_value("CHATTO_LOGIN") or "").strip()
    password = (gateway_mod.get_env_value("CHATTO_PASSWORD") or "").strip()
    return bool(base_url and (token or (login and password)))


# ---------------------------------------------------------------------------
# Plugin registration entry point
# ---------------------------------------------------------------------------


def _build_adapter(config):
    """Factory wrapper — constructs ChattoAdapter from a PlatformConfig."""
    return ChattoAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="chatto",
        label="Chatto",
        adapter_factory=_build_adapter,
        check_fn=check_chatto_requirements,
        is_connected=_is_connected,
        required_env=["CHATTO_BASE_URL", "CHATTO_TOKEN"],
        install_hint="pip install hermes-agent[chatto]",
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="CHATTO_ALLOWED_USERS",
        allow_all_env="CHATTO_ALLOW_ALL_USERS",
        cron_deliver_env_var="CHATTO_HOME_ROOM",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        allow_update_command=True,
    )
