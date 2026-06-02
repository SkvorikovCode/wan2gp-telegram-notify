"""Minimal Telegram Bot API client built on the standard library.

Kept dependency-free (urllib only) to match the rest of WanGP's plugins.
Every call returns a (ok, result_or_error) tuple so callers never have to
catch network exceptions themselves.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_USER_AGENT = "Wan2GPTelegramNotify/2.0"
_OPENER: Optional[urllib.request.OpenerDirector] = None
_PROXY_MODE: str = "direct"
_PROXY_URL: str = ""
_PROXY_USER: str = ""
_PROXY_PASSWORD: str = ""
_PROXY_FINGERPRINT: tuple[str, str, str, str] = ("direct", "", "", "")


def configure_network(
    proxy_mode: str = "direct",
    proxy_url: str = "",
    proxy_user: str = "",
    proxy_password: str = "",
) -> None:
    """Apply proxy settings from the plugin UI (call before Telegram API requests)."""
    global _OPENER, _PROXY_MODE, _PROXY_URL, _PROXY_USER, _PROXY_PASSWORD, _PROXY_FINGERPRINT
    mode = (proxy_mode or "direct").strip().lower()
    url = (proxy_url or "").strip()
    user = (proxy_user or "").strip()
    password = proxy_password or ""
    fingerprint = (mode, url, user, password)
    if fingerprint == _PROXY_FINGERPRINT and _OPENER is not None:
        return
    _PROXY_MODE, _PROXY_URL, _PROXY_USER, _PROXY_PASSWORD = mode, url, user, password
    _PROXY_FINGERPRINT = fingerprint
    _OPENER = None


def _inject_proxy_auth(proxy_url: str, user: str, password: str) -> str:
    """Add login:password to proxy URL if not already embedded."""
    if not user and not password:
        return proxy_url
    parsed = urllib.parse.urlparse(proxy_url)
    if parsed.username:
        return proxy_url
    auth_user = urllib.parse.quote(user, safe="")
    auth_pass = urllib.parse.quote(password, safe="")
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = f"{auth_user}:{auth_pass}@{host}"
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc))


def _normalize_proxy_url(
    mode: str,
    url: str,
    *,
    user: str = "",
    password: str = "",
) -> Optional[str]:
    mode = (mode or "direct").strip().lower()
    if mode in ("", "direct", "none", "off"):
        return None
    if mode == "system":
        return None
    raw = (url or "").strip()
    if not raw:
        env_url = (os.environ.get("TG_BOT_PROXY") or "").strip()
        if env_url:
            raw = env_url if "://" in env_url else f"http://{env_url}"
        else:
            return None
    if "://" not in raw and "@" in raw:
        scheme = "socks5h" if mode in ("socks5", "socks5h", "socks") else "http"
        raw = f"{scheme}://{raw}"
    elif "://" not in raw:
        if mode in ("socks5", "socks5h", "socks"):
            raw = f"socks5h://{raw}"
        else:
            raw = f"http://{raw}"
    return _inject_proxy_auth(raw, user, password)


def _get_opener() -> urllib.request.OpenerDirector:
    global _OPENER
    if _OPENER is not None:
        return _OPENER

    mode = _PROXY_MODE
    if mode == "system":
        _OPENER = urllib.request.build_opener()
        return _OPENER

    proxy_url = _normalize_proxy_url(
        mode, _PROXY_URL, user=_PROXY_USER, password=_PROXY_PASSWORD
    )
    if not proxy_url:
        _OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return _OPENER

    if proxy_url.startswith("socks"):
        try:
            import socks  # noqa: F401  # PySocks
        except ImportError as exc:
            raise RuntimeError(
                "SOCKS5 proxy needs PySocks in the WanGP venv: pip install PySocks"
            ) from exc

    _OPENER = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
    )
    return _OPENER


def _request(
    req: urllib.request.Request,
    *,
    timeout: float,
    attempts: int = 3,
) -> bytes:
    """GET/POST with retries on transient connection resets."""
    opener = _get_opener()
    last_exc: Optional[Exception] = None
    for attempt in range(max(1, attempts)):
        try:
            with opener.open(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


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
        req = urllib.request.Request(
            url,
            data=data,
            headers={"User-Agent": _USER_AGENT},
        )
        raw = _request(req, timeout=timeout, attempts=3)
        body = json.loads(raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            return False, body.get("description", str(exc))
        except Exception:
            return False, f"HTTP {exc.code}"
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, _friendly_network_error(exc)
    if not body.get("ok"):
        return False, body.get("description", "unknown error")
    return True, body.get("result")


def _friendly_network_error(exc: Exception) -> str:
    text = str(exc).strip()
    if "10054" in text or "forcibly closed" in text.lower() or "connection reset" in text.lower():
        return (
            "Connection to Telegram was reset. "
            "If you use a VPN/proxy (Clash, V2Ray), allow direct access to api.telegram.org "
            "or disable the system proxy while WanGP runs."
        )
    return text or "network error"


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


def get_file(bot_token: str, file_id: str) -> tuple[bool, Any]:
    """Resolve a Telegram file_id to a file_path on Telegram servers."""
    return _call(bot_token, "getFile", {"file_id": file_id})


def download_telegram_file(
    bot_token: str,
    telegram_file_path: str,
    dest_path: str,
    *,
    timeout: float = 180.0,
) -> tuple[bool, str]:
    """Download a file from Telegram's CDN to a local path."""
    url = f"https://api.telegram.org/file/bot{bot_token}/{telegram_file_path.lstrip('/')}"
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        data = _request(req, timeout=timeout, attempts=5)
        with open(dest_path, "wb") as handle:
            handle.write(data)
        return True, dest_path
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, _friendly_network_error(exc)


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
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        raw = _request(req, timeout=300, attempts=3)
        payload = json.loads(raw.decode("utf-8", "replace"))
        if not payload.get("ok"):
            return False, payload.get("description", "upload failed")
        return True, payload.get("result")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace"))
            return False, payload.get("description", str(exc))
        except Exception:
            return False, f"HTTP {exc.code}"
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, _friendly_network_error(exc)


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
