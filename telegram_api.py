"""Minimal Telegram Bot API client built on the standard library.

Kept dependency-free (urllib only) to match the rest of WanGP's plugins.
Every call returns a (ok, result_or_error) tuple so callers never have to
catch network exceptions themselves.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

_API_BASE = "https://api.telegram.org/bot{token}/{method}"


def _call(bot_token: str, method: str, params: dict, timeout: float = 15.0) -> tuple[bool, Any]:
    """POST a urlencoded request to the Bot API and decode the JSON envelope."""
    url = _API_BASE.format(token=bot_token, method=method)
    encoded = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            encoded[key] = json.dumps(value)
        elif isinstance(value, bool):
            encoded[key] = "true" if value else "false"
        else:
            encoded[key] = value
    data = urllib.parse.urlencode(encoded).encode()
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            return False, body.get("description", str(exc))
        except Exception:
            return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)
    if not body.get("ok"):
        return False, body.get("description", "unknown error")
    return True, body.get("result")


def send_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    reply_markup: Optional[dict] = None,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = True,
) -> tuple[bool, Any]:
    return _call(
        bot_token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
            "reply_markup": reply_markup,
        },
    )


def edit_message_text(
    bot_token: str,
    chat_id: str,
    message_id: int,
    text: str,
    *,
    reply_markup: Optional[dict] = None,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = True,
) -> tuple[bool, Any]:
    return _call(
        bot_token,
        "editMessageText",
        {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
            "reply_markup": reply_markup,
        },
    )


def edit_message_reply_markup(
    bot_token: str,
    chat_id: str,
    message_id: int,
    reply_markup: Optional[dict],
) -> tuple[bool, Any]:
    return _call(
        bot_token,
        "editMessageReplyMarkup",
        {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup},
    )


def answer_callback_query(
    bot_token: str,
    callback_query_id: str,
    text: str = "",
    show_alert: bool = False,
) -> tuple[bool, Any]:
    return _call(
        bot_token,
        "answerCallbackQuery",
        {"callback_query_id": callback_query_id, "text": text, "show_alert": show_alert},
    )


def get_updates(
    bot_token: str,
    offset: Optional[int] = None,
    timeout: int = 25,
    allowed_updates: Optional[list] = None,
) -> tuple[bool, Any]:
    # Network timeout must exceed the long-poll timeout so the socket does not
    # drop the connection while Telegram is still holding it open.
    return _call(
        bot_token,
        "getUpdates",
        {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": allowed_updates or ["message", "callback_query"],
        },
        timeout=timeout + 10,
    )


def delete_message(
    bot_token: str,
    chat_id: str,
    message_id: int,
) -> tuple[bool, Any]:
    return _call(
        bot_token,
        "deleteMessage",
        {"chat_id": chat_id, "message_id": message_id},
    )


def get_me(bot_token: str) -> tuple[bool, Any]:
    return _call(bot_token, "getMe", {})


def send_file(
    bot_token: str,
    chat_id: str,
    file_path: str,
    caption: str = "",
    *,
    reply_markup: Optional[dict] = None,
    parse_mode: str = "HTML",
) -> tuple[bool, Any]:
    """Upload a local video/image/document via multipart/form-data."""
    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or "application/octet-stream"
    if mime.startswith("video"):
        method, field = "sendVideo", "video"
    elif mime.startswith("image") and not mime.endswith("gif"):
        method, field = "sendPhoto", "photo"
    else:
        method, field = "sendDocument", "document"

    try:
        with open(file_path, "rb") as handle:
            file_data = handle.read()
    except Exception as exc:
        return False, f"Cannot read file: {exc}"

    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)

    def encode_field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    parts = [
        encode_field("chat_id", str(chat_id)),
        encode_field("parse_mode", parse_mode),
    ]
    if caption:
        parts.append(encode_field("caption", caption))
    if reply_markup is not None:
        parts.append(encode_field("reply_markup", json.dumps(reply_markup)))
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
    )
    body = b"".join(parts) + file_data + f"\r\n--{boundary}--\r\n".encode()

    url = _API_BASE.format(token=bot_token, method=method)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        if not payload.get("ok"):
            return False, payload.get("description", "upload failed")
        return True, payload.get("result")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace"))
            return False, payload.get("description", str(exc))
        except Exception:
            return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


# ----------------------------------------------------------------------------
# Inline-keyboard helpers
# ----------------------------------------------------------------------------

def inline_keyboard(rows: list[list[tuple[str, str]]]) -> dict:
    """Build an inline_keyboard markup from rows of (label, callback_data)."""
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data} for label, data in row]
            for row in rows
        ]
    }
