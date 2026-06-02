"""Telegram bot for WanGP — group-chat edition.

One persistent "panel" message per chat that is always edited in-place.
User command messages are deleted immediately after processing.
Navigation is fully inline-keyboard driven; /cancel returns to the main menu
from any nested screen.

Supported commands (also available as inline buttons):
  /start, /help    – show main menu
  /hello           – show current settings and quick-launch panel
  /status          – current state
  /regen           – re-run last generation with a new seed
  /generate        – run with current settings
  /prompt <text>   – set prompt (or opens text-input mode)
  /model           – model picker (paginated)
  /settings        – settings hub (General / LoRAs / Post / Quality / …)
  /loras           – toggle LoRAs and edit multipliers
  /image           – set start image (i2v); then send a photo
  /endimage        – set end image; then send a photo
  /clearimage      – remove start/end images
  /set key value   – change one setting
  /abort           – stop running generation
  /cancel          – cancel any pending input and return to main menu
"""

from __future__ import annotations

import html
import os
import threading
import time
from typing import Any, Optional

from . import telegram_api as tg
from .engine import (
    EDITABLE_SETTINGS,
    SETTINGS_CODE_TO_SECTION,
    SETTINGS_SECTION_CODES,
    SETTINGS_SECTION_ORDER,
    SETTINGS_SECTIONS,
    GenerationEngine,
    _coerce,
    format_setting_display,
    section_keys,
)

_SETTINGS_PAGE_SIZE = 6
_LORA_PAGE_SIZE = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _section_emoji(section: str) -> str:
    return {
        "General": "⚙️",
        "LoRAs": "🎭",
        "Post Processing": "🎞️",
        "Quality": "✨",
        "Sliding Window": "🪟",
        "Misc": "🔧",
    }.get(section, "•")


def _settings_hub_keyboard() -> dict:
    rows = [
        [(f"{_section_emoji(s)} {s}", f"sec:{SETTINGS_SECTION_CODES[s]}")]
        for s in SETTINGS_SECTION_ORDER
    ]
    rows.append(_back_cancel_row())
    return tg.inline_keyboard(rows)


def _main_keyboard(busy: bool) -> dict:
    rows = [
        [("🔁 Regenerate", "act:regen"), ("▶️ Generate", "act:generate")],
        [("✏️ Prompt", "act:prompt"), ("🧩 Model", "act:model")],
        [("⚙️ Settings", "act:settings"), ("🎭 LoRAs", "act:loras")],
        [("📷 Start image", "act:image"), ("📊 Status", "act:status")],
    ]
    if busy:
        rows.append([("⏹ Abort", "act:abort")])
    return tg.inline_keyboard(rows)


def _cancel_keyboard() -> dict:
    return tg.inline_keyboard([[("✖ Cancel", "act:cancel")]])


def _back_keyboard() -> dict:
    return tg.inline_keyboard([[("◀️ Back", "act:back")]])


def _back_cancel_row() -> list:
    return [("◀️ Back", "act:back"), ("✖ Cancel", "act:cancel")]


def _extract_image_file_id(message: dict) -> Optional[str]:
    photos = message.get("photo") or []
    if photos:
        return str(photos[-1].get("file_id", "") or "") or None
    doc = message.get("document") or {}
    mime = str(doc.get("mime_type", "") or "").lower()
    if mime.startswith("image/"):
        return str(doc.get("file_id", "") or "") or None
    return None


# ---------------------------------------------------------------------------
# Per-chat state
# ---------------------------------------------------------------------------

class _ChatState:
    """Mutable state for one chat."""

    def __init__(self) -> None:
        self.panel_message_id: Optional[int] = None   # the one persistent message
        self.pending: Optional[dict] = None            # waiting for free-text input
        self.model_page: int = 0
        self.lock = threading.Lock()
        # Short-key → model_type map to stay within Telegram's 64-byte callback_data limit
        self._model_key_map: dict[str, str] = {}
        self._model_key_counter: int = 0
        self.settings_section: str = "General"
        self.settings_page: int = 0
        self.lora_page: int = 0
        self._lora_key_map: dict[str, str] = {}
        self._lora_key_counter: int = 0

    def register_model_key(self, model_type: str) -> str:
        """Return a short key for model_type, creating one if needed."""
        for k, v in self._model_key_map.items():
            if v == model_type:
                return k
        self._model_key_counter += 1
        key = str(self._model_key_counter)
        self._model_key_map[key] = model_type
        return key

    def resolve_model_key(self, key: str) -> Optional[str]:
        return self._model_key_map.get(key)

    def register_lora_key(self, basename: str) -> str:
        for k, v in self._lora_key_map.items():
            if v == basename:
                return k
        self._lora_key_counter += 1
        key = f"l{self._lora_key_counter}"
        self._lora_key_map[key] = basename
        return key

    def resolve_lora_key(self, key: str) -> Optional[str]:
        return self._lora_key_map.get(key)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class TelegramBot:
    def __init__(self, config_provider, engine: GenerationEngine) -> None:
        self._config_provider = config_provider
        self._engine = engine
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._offset: Optional[int] = None
        # chat_id (str) -> _ChatState
        self._chats: dict[str, _ChatState] = {}

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        # Clear stale panel ids from a previous session so old messages don't
        # accumulate with dead buttons after a restart.
        for state in self._chats.values():
            with state.lock:
                state.panel_message_id = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="tg-bot-poll")
        self._thread.start()

    def _drain_pending_updates(self, token: str) -> None:
        """Consume all queued updates without processing them to avoid replay on restart."""
        ok, result = tg.get_updates(token, offset=self._offset, timeout=0)
        if ok and result:
            self._offset = result[-1].get("update_id", 0) + 1

    def stop(self) -> None:
        self._stop.set()
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------------- config

    def _cfg(self) -> dict:
        try:
            return self._config_provider() or {}
        except Exception:
            return {}

    def _is_authorized(self, chat_id: str) -> bool:
        cfg = self._cfg()
        chat_id = str(chat_id)
        allowed = {str(cfg.get("chat_id", "")).strip()}
        for extra in str(cfg.get("allowed_chat_ids", "") or "").replace(";", ",").split(","):
            extra = extra.strip()
            if extra:
                allowed.add(extra)
        allowed.discard("")
        return bool(allowed) and chat_id in allowed

    # ---------------------------------------------------------------- chat state

    def _state(self, chat_id: str) -> _ChatState:
        if chat_id not in self._chats:
            self._chats[chat_id] = _ChatState()
        return self._chats[chat_id]

    # ---------------------------------------------------------------- panel management

    def _show_panel(self, token: str, chat_id: str, text: str, keyboard: dict) -> None:
        """Edit the existing panel message or create a new one."""
        state = self._state(chat_id)
        with state.lock:
            mid = state.panel_message_id
        if mid is not None:
            ok, err = tg.edit_message_text(token, chat_id, mid, text, reply_markup=keyboard)
            if ok:
                return
            # "message is not modified" — content identical, panel is still there, do nothing.
            if isinstance(err, str) and "not modified" in err.lower():
                return
            # Any other error (message deleted, too old, etc.) — fall through to send a new one.
        ok, result = tg.send_message(token, chat_id, text, reply_markup=keyboard)
        if ok and isinstance(result, dict):
            with state.lock:
                state.panel_message_id = result.get("message_id")

    def _delete_message(self, token: str, chat_id: str, message_id: int) -> None:
        tg.delete_message(token, chat_id, message_id)

    # ---------------------------------------------------------------- main loop

    def _run(self) -> None:
        # Flush any updates that accumulated while the bot was offline so they
        # are not replayed as fresh commands.
        cfg = self._cfg()
        token = str(cfg.get("bot_token", "")).strip()
        if token:
            self._drain_pending_updates(token)

        while not self._stop.is_set():
            cfg = self._cfg()
            token = str(cfg.get("bot_token", "")).strip()
            if not token or not cfg.get("bot_enabled"):
                self._stop.wait(2.0)
                continue
            ok, result = tg.get_updates(token, offset=self._offset, timeout=25)
            if not ok:
                self._stop.wait(3.0)
                continue
            for update in result or []:
                try:
                    self._handle_update(token, update)
                except Exception as exc:
                    print(f"[TelegramBot] error handling update: {exc}")
                self._offset = update.get("update_id", 0) + 1

    # ---------------------------------------------------------------- dispatch

    def _handle_update(self, token: str, update: dict) -> None:
        if "message" in update:
            self._handle_message(token, update["message"])
        elif "callback_query" in update:
            self._handle_callback(token, update["callback_query"])

    def _handle_message(self, token: str, message: dict) -> None:
        chat_id = str(message.get("chat", {}).get("id", ""))
        msg_id: int = message.get("message_id", 0)
        text = (message.get("text") or "").strip()
        file_id = _extract_image_file_id(message)
        if not chat_id:
            return

        if not self._is_authorized(chat_id):
            if msg_id:
                self._delete_message(token, chat_id, msg_id)
            tg.send_message(token, chat_id, "⛔ This chat is not authorized.")
            return

        state = self._state(chat_id)

        if file_id:
            pending = state.pending
            slot = "start"
            if isinstance(pending, dict):
                mode = pending.get("mode", "")
                if mode == "image_end":
                    slot = "end"
                elif mode == "image_start":
                    slot = "start"
                elif mode in ("prompt",) or str(mode).startswith("set:") or mode == "loras_mults":
                    if msg_id:
                        self._delete_message(token, chat_id, msg_id)
                    self._show_panel(
                        token, chat_id,
                        "⚠️ Finish the current input first, or tap ✖ Cancel.",
                        _cancel_keyboard(),
                    )
                    return
            if msg_id:
                self._delete_message(token, chat_id, msg_id)
            with state.lock:
                state.pending = None
            self._apply_telegram_image(token, chat_id, file_id, slot)
            return

        # Delete text commands to keep the chat clean.
        if msg_id:
            self._delete_message(token, chat_id, msg_id)

        # /cancel always wins.
        if text.lstrip("/").lower().split("@")[0] == "cancel":
            with state.lock:
                state.pending = None
            self._show_main_menu(token, chat_id)
            return

        # Free-text follow-up (prompt entry or /set value).
        pending = state.pending
        if pending and text and not text.startswith("/"):
            self._consume_pending(token, chat_id, pending, text)
            return

        if not text:
            return

        command, _, argument = text.partition(" ")
        command = command.lstrip("/").lower().split("@")[0]
        argument = argument.strip()

        handler = {
            "start":      self._cmd_help,
            "help":       self._cmd_help,
            "hello":      self._cmd_hello,
            "status":     self._cmd_status,
            "regen":      self._cmd_regen,
            "regenerate": self._cmd_regen,
            "generate":   self._cmd_generate,
            "prompt":     self._cmd_prompt,
            "model":      lambda t, c, a: self._show_model_picker(t, c, 0),
            "settings":   self._cmd_settings,
            "loras":      self._cmd_loras,
            "image":      self._cmd_image,
            "startimage": self._cmd_image,
            "endimage":   self._cmd_endimage,
            "clearimage": self._cmd_clearimage,
            "set":        self._cmd_set,
            "abort":      self._cmd_abort,
            "stop":       self._cmd_abort,
        }.get(command)

        if handler is None:
            # Unknown command — just show the main menu.
            self._show_main_menu(token, chat_id)
            return
        handler(token, chat_id, argument)

    def _handle_callback(self, token: str, query: dict) -> None:
        query_id = query.get("id", "")
        data = query.get("data", "") or ""
        message = query.get("message", {}) or {}
        chat_id = str(message.get("chat", {}).get("id", ""))
        msg_id: int = message.get("message_id", 0)

        if not self._is_authorized(chat_id):
            tg.answer_callback_query(token, query_id, "Not authorized", show_alert=True)
            return
        tg.answer_callback_query(token, query_id)

        # Keep track of which message is the panel (in case it was recreated).
        if msg_id:
            state = self._state(chat_id)
            with state.lock:
                state.panel_message_id = msg_id

        kind, _, payload = data.partition(":")
        if kind == "act":
            self._dispatch_action(token, chat_id, payload)
        elif kind == "model":
            self._select_model(token, chat_id, payload)
        elif kind == "mpage":
            self._show_model_picker(token, chat_id, int(payload or 0))
        elif kind == "set":
            self._prompt_for_setting(token, chat_id, payload)
        elif kind == "sec":
            section = SETTINGS_CODE_TO_SECTION.get(payload)
            if section:
                self._show_settings_section(token, chat_id, section, 0)
        elif kind == "spage":
            code, _, page_s = payload.partition(":")
            section = SETTINGS_CODE_TO_SECTION.get(code)
            if section:
                self._show_settings_section(token, chat_id, section, int(page_s or 0))
        elif kind == "sw":
            self._toggle_switch_setting(token, chat_id, payload)
        elif kind == "lt":
            self._toggle_lora(token, chat_id, payload)
        elif kind == "lpage":
            self._show_loras_panel(token, chat_id, int(payload or 0))
        elif kind == "loram":
            self._prompt_lora_multipliers(token, chat_id)

    def _dispatch_action(self, token: str, chat_id: str, action: str) -> None:
        if action == "cancel":
            state = self._state(chat_id)
            with state.lock:
                state.pending = None
            self._show_main_menu(token, chat_id)
        elif action == "back":
            state = self._state(chat_id)
            with state.lock:
                state.pending = None
            self._show_main_menu(token, chat_id)
        elif action == "regen":
            self._cmd_regen(token, chat_id, "")
        elif action == "generate":
            self._cmd_generate(token, chat_id, "")
        elif action == "hello":
            self._show_hello_panel(token, chat_id)
        elif action == "prompt":
            self._cmd_prompt(token, chat_id, "")
        elif action == "model":
            self._show_model_picker(token, chat_id, 0)
        elif action == "settings":
            self._cmd_settings(token, chat_id, "")
        elif action == "loras":
            self._cmd_loras(token, chat_id, "")
        elif action == "settings_hub":
            self._show_settings_hub(token, chat_id)
        elif action == "lora_clear":
            self._engine.set_active_loras([], "")
            state = self._state(chat_id)
            self._show_loras_panel(token, chat_id, state.lora_page)
        elif action == "image":
            self._cmd_image(token, chat_id, "")
        elif action == "endimage":
            self._cmd_endimage(token, chat_id, "")
        elif action == "clearimage":
            self._cmd_clearimage(token, chat_id, "")
        elif action == "status":
            self._cmd_status(token, chat_id, "")
        elif action == "abort":
            self._cmd_abort(token, chat_id, "")

    # ---------------------------------------------------------------- main menu

    def _show_main_menu(self, token: str, chat_id: str) -> None:
        busy = self._engine.is_busy()
        settings = self._engine.get_last_settings()
        lines = ["<b>WanGP Control Panel</b>"]
        if settings:
            model = settings.get("model_type", "")
            lines.append(f"Model: <code>{_esc(self._engine.model_display_name(model))}</code>")
            prompt = str(settings.get("prompt", ""))
            if prompt:
                lines.append(f"Prompt: {_esc(prompt[:120])}")
            img_start, img_end = self._engine.get_image_paths_summary()
            if img_start:
                lines.append(f"📷 Start: <code>{_esc(img_start)}</code>")
            if img_end:
                lines.append(f"📷 End: <code>{_esc(img_end)}</code>")
        lines.append("🟡 Busy generating…" if busy else "🟢 Idle")
        self._show_panel(token, chat_id, "\n".join(lines), _main_keyboard(busy))

    # ---------------------------------------------------------------- commands

    def _cmd_help(self, token: str, chat_id: str, _argument: str) -> None:
        self._show_main_menu(token, chat_id)

    def _cmd_hello(self, token: str, chat_id: str, _argument: str) -> None:
        self._show_hello_panel(token, chat_id)

    def _show_hello_panel(self, token: str, chat_id: str) -> None:
        busy = self._engine.is_busy()
        settings = self._engine.get_last_settings()

        lines = ["👋 <b>Ready to generate</b>"]

        if settings:
            model = settings.get("model_type", "")
            lines.append(f"\n🧩 <b>Model:</b> <code>{_esc(self._engine.model_display_name(model))}</code>")

            prompt = str(settings.get("prompt", "")).strip()
            lines.append(f"✏️ <b>Prompt:</b> {_esc(prompt[:200]) if prompt else '—'}")

            neg = str(settings.get("negative_prompt", "")).strip()
            if neg:
                lines.append(f"🚫 <b>Negative:</b> {_esc(neg[:100])}")

            lines.append(
                f"🔢 <b>Steps:</b> {_esc(settings.get('num_inference_steps', '?'))}  "
                f"🎲 <b>Seed:</b> {_esc(settings.get('seed', '?'))}  "
                f"📐 <b>Res:</b> {_esc(settings.get('resolution', '?'))}"
            )
            active_loras, mults = self._engine.get_active_loras()
            if active_loras:
                lines.append(f"🎭 <b>LoRAs:</b> {_esc(', '.join(active_loras[:5]))}"
                             + ("…" if len(active_loras) > 5 else ""))
                if mults.strip():
                    lines.append(f"   <i>mult:</i> <code>{_esc(mults[:80])}</code>")

            img_start, img_end = self._engine.get_image_paths_summary()
            if img_start:
                lines.append(f"📷 <b>Start image:</b> <code>{_esc(img_start)}</code>")
            if img_end:
                lines.append(f"📷 <b>End image:</b> <code>{_esc(img_end)}</code>")

            guidance = settings.get("guidance_scale")
            frames = settings.get("video_length")
            if guidance is not None or frames is not None:
                extra = []
                if guidance is not None:
                    extra.append(f"🎯 <b>Guidance:</b> {_esc(guidance)}")
                if frames is not None:
                    extra.append(f"🎞 <b>Frames:</b> {_esc(frames)}")
                lines.append("  ".join(extra))
        else:
            lines.append("\n⚠️ No settings yet — configure in the UI first or pick a model below.")

        lines.append("\n" + ("🟡 Busy generating…" if busy else "🟢 Idle — pick an action:"))

        keyboard = tg.inline_keyboard([
            [("▶️ Generate", "act:generate"), ("🔁 New seed", "act:regen")],
            [("✏️ Prompt", "act:prompt"), ("🧩 Model", "act:model")],
            [("⚙️ Settings", "act:settings"), ("🎭 LoRAs", "act:loras")],
            [("📷 Start image", "act:image"), ("📊 Status", "act:status")],
        ] + ([[("⏹ Abort", "act:abort")]] if busy else []))

        self._show_panel(token, chat_id, "\n".join(lines), keyboard)

    def _cmd_status(self, token: str, chat_id: str, _argument: str) -> None:
        busy = self._engine.is_busy()
        lines = ["<b>WanGP Status</b>", "🟡 Busy generating…" if busy else "🟢 Idle"]
        settings = self._engine.get_last_settings()
        if settings:
            model = settings.get("model_type", "")
            lines.append(f"Model: <code>{_esc(self._engine.model_display_name(model))}</code>")
            prompt = str(settings.get("prompt", ""))
            if prompt:
                lines.append(f"Prompt: {_esc(prompt[:200])}")
            lines.append(
                f"Steps {_esc(settings.get('num_inference_steps', '?'))} · "
                f"Seed {_esc(settings.get('seed', '?'))} · "
                f"{_esc(settings.get('resolution', '?'))}"
            )
            img_start, img_end = self._engine.get_image_paths_summary()
            if img_start or img_end:
                parts = []
                if img_start:
                    parts.append(f"start <code>{_esc(img_start)}</code>")
                if img_end:
                    parts.append(f"end <code>{_esc(img_end)}</code>")
                lines.append("📷 " + " · ".join(parts))
        else:
            lines.append("No generation captured yet. Run one from the UI first.")
        self._show_panel(token, chat_id, "\n".join(lines), _main_keyboard(busy))

    def _cmd_regen(self, token: str, chat_id: str, _argument: str) -> None:
        base = self._engine.get_last_settings()
        if not base:
            self._show_panel(
                token, chat_id,
                "⚠️ Nothing to regenerate yet. Generate once from the UI first.",
                _back_keyboard(),
            )
            return
        settings = self._engine.build_settings(base, {}, new_seed=True)
        self._launch(token, chat_id, settings, "🔁 Regenerating with a new seed…")

    def _cmd_generate(self, token: str, chat_id: str, _argument: str) -> None:
        base = self._engine.get_last_settings()
        if not base:
            self._show_panel(
                token, chat_id,
                "⚠️ No settings to generate from yet.",
                _back_keyboard(),
            )
            return
        settings = self._engine.build_settings(base, {}, new_seed=False)
        self._launch(token, chat_id, settings, "▶️ Generating with current settings…")

    def _cmd_image(self, token: str, chat_id: str, _argument: str) -> None:
        if not self._require_settings(token, chat_id):
            return
        state = self._state(chat_id)
        with state.lock:
            state.pending = {"mode": "image_start"}
        self._show_panel(
            token, chat_id,
            "📷 <b>Start image (i2v)</b>\n\nSend a photo in this chat.\n"
            "You can also send a photo anytime — it will be used as the start frame.",
            _cancel_keyboard(),
        )

    def _cmd_endimage(self, token: str, chat_id: str, _argument: str) -> None:
        if not self._require_settings(token, chat_id):
            return
        state = self._state(chat_id)
        with state.lock:
            state.pending = {"mode": "image_end"}
        self._show_panel(
            token, chat_id,
            "📷 <b>End image</b>\n\nSend a photo to use as the last frame.",
            _cancel_keyboard(),
        )

    def _cmd_clearimage(self, token: str, chat_id: str, _argument: str) -> None:
        if not self._require_settings(token, chat_id):
            return
        self._engine.clear_image_attachment("start")
        self._engine.clear_image_attachment("end")
        self._show_panel(
            token, chat_id,
            "🗑 Start and end images cleared.",
            tg.inline_keyboard([
                [("📷 Start image", "act:image"), ("◀️ Menu", "act:back")],
            ]),
        )

    def _apply_telegram_image(self, token: str, chat_id: str, file_id: str, slot: str) -> None:
        self._cfg()
        if not self._require_settings(token, chat_id):
            return
        ok, path_or_err = self._engine.save_telegram_image(token, file_id, chat_id, slot=slot)
        if not ok:
            self._show_panel(
                token, chat_id,
                f"⚠️ Could not download image: {_esc(path_or_err)}",
                _back_keyboard(),
            )
            return
        ok, msg = self._engine.set_image_attachment(slot, path_or_err)
        if not ok:
            self._show_panel(token, chat_id, f"⚠️ {_esc(msg)}", _back_keyboard())
            return
        label = "Start image" if slot == "start" else "End image"
        keyboard = tg.inline_keyboard([
            [("▶️ Generate", "act:generate"), ("🔁 New seed", "act:regen")],
            [("✏️ Prompt", "act:prompt"), ("◀️ Menu", "act:back")],
        ])
        self._show_panel(
            token, chat_id,
            f"✅ <b>{label}</b> set: <code>{_esc(msg)}</code>",
            keyboard,
        )

    def _cmd_prompt(self, token: str, chat_id: str, argument: str) -> None:
        if argument:
            self._apply_setting(token, chat_id, "prompt", argument, offer_generate=True)
            return
        state = self._state(chat_id)
        with state.lock:
            state.pending = {"mode": "prompt"}
        self._show_panel(
            token, chat_id,
            "✏️ <b>Set prompt</b>\n\nSend the new prompt text as a message.",
            _cancel_keyboard(),
        )

    def _require_settings(self, token: str, chat_id: str) -> Optional[dict]:
        settings = self._engine.get_last_settings()
        if not settings:
            self._show_panel(
                token, chat_id,
                "⚠️ No settings captured yet. Generate once from the UI or pick a model.",
                _back_keyboard(),
            )
            return None
        return settings

    def _cmd_settings(self, token: str, chat_id: str, _argument: str) -> None:
        if not self._require_settings(token, chat_id):
            return
        self._show_settings_hub(token, chat_id)

    def _cmd_loras(self, token: str, chat_id: str, _argument: str) -> None:
        if not self._require_settings(token, chat_id):
            return
        self._show_loras_panel(token, chat_id, 0)

    def _show_settings_hub(self, token: str, chat_id: str) -> None:
        self._show_panel(
            token, chat_id,
            "<b>⚙️ Settings</b>\nChoose a section:",
            _settings_hub_keyboard(),
        )

    def _show_settings_section(self, token: str, chat_id: str, section: str, page: int) -> None:
        if section == "LoRAs":
            self._show_loras_panel(token, chat_id, page)
            return
        settings = self._require_settings(token, chat_id)
        if not settings:
            return
        items = SETTINGS_SECTIONS.get(section, [])
        if not items:
            self._show_settings_hub(token, chat_id)
            return
        state = self._state(chat_id)
        with state.lock:
            state.settings_section = section
            state.settings_page = page
        page = max(0, min(page, max(0, (len(items) - 1) // _SETTINGS_PAGE_SIZE)))
        start = page * _SETTINGS_PAGE_SIZE
        chunk = items[start:start + _SETTINGS_PAGE_SIZE]
        lines = [f"<b>{_section_emoji(section)} {_esc(section)}</b>"]
        rows = []
        for key, meta in chunk:
            value = settings.get(key)
            shown = format_setting_display(key, value)
            hint = meta.get("hint")
            line = f"• <b>{_esc(meta['label'])}</b>: <code>{_esc(shown)}</code>"
            if hint:
                line += f"\n  <i>{_esc(hint)}</i>"
            lines.append(line)
            if meta.get("type") == "switch":
                rows.append([
                    (f"{'🟢' if int(value or 0) else '⚪'} {meta['label'][:28]}", f"sw:{key}"),
                    (f"✏️ Set", f"set:{key}"),
                ])
            else:
                rows.append([(f"✏️ {meta['label'][:32]}", f"set:{key}")])
        nav = []
        code = SETTINGS_SECTION_CODES[section]
        if page > 0:
            nav.append(("⬅️", f"spage:{code}:{page - 1}"))
        if start + _SETTINGS_PAGE_SIZE < len(items):
            nav.append(("➡️", f"spage:{code}:{page + 1}"))
        if nav:
            rows.append(nav)
        rows.append([("◀️ Sections", "act:settings_hub"), ("✖ Cancel", "act:cancel")])
        total_pages = max(1, (len(items) + _SETTINGS_PAGE_SIZE - 1) // _SETTINGS_PAGE_SIZE)
        lines.append(f"\n<i>Page {page + 1}/{total_pages}</i>")
        self._show_panel(token, chat_id, "\n".join(lines), tg.inline_keyboard(rows))

    def _toggle_switch_setting(self, token: str, chat_id: str, key: str) -> None:
        meta = EDITABLE_SETTINGS.get(key)
        if not meta or meta.get("type") != "switch":
            return
        base = self._require_settings(token, chat_id)
        if not base:
            return
        current = int(base.get(key, 0) or 0)
        base[key] = 0 if current else 1
        self._engine.set_last_settings(base)
        state = self._state(chat_id)
        with state.lock:
            section = state.settings_section
            page = state.settings_page
        self._show_settings_section(token, chat_id, section, page)

    def _show_loras_panel(self, token: str, chat_id: str, page: int) -> None:
        settings = self._require_settings(token, chat_id)
        if not settings:
            return
        model_type = settings.get("model_type", "")
        all_loras = self._engine.list_loras(model_type)
        active, mults = self._engine.get_active_loras()
        active_set = set(active)

        state = self._state(chat_id)
        with state.lock:
            state.lora_page = page
            state.settings_section = "LoRAs"

        lines = [
            f"<b>🎭 LoRAs</b> — <code>{_esc(self._engine.model_display_name(model_type))}</code>",
            f"Active: <b>{len(active)}</b>",
        ]
        if active:
            lines.append("✅ " + ", ".join(_esc(n) for n in active[:6])
                       + ("…" if len(active) > 6 else ""))
        if mults.strip():
            lines.append(f"Multipliers: <code>{_esc(mults[:120])}</code>")
        else:
            lines.append("<i>Multipliers: 1 per LoRA (default)</i>")

        if not all_loras:
            lines.append("\n⚠️ No LoRA files in this model's folder.")
            rows = [
                [("✏️ Edit multipliers", "loram:")],
                [("◀️ Sections", "act:settings_hub"), ("◀️ Menu", "act:back")],
            ]
            self._show_panel(token, chat_id, "\n".join(lines), tg.inline_keyboard(rows))
            return

        page = max(0, min(page, max(0, (len(all_loras) - 1) // _LORA_PAGE_SIZE)))
        start = page * _LORA_PAGE_SIZE
        chunk = all_loras[start:start + _LORA_PAGE_SIZE]
        rows = []
        for basename in chunk:
            key = state.register_lora_key(basename)
            mark = "✅" if basename in active_set else "⬜"
            short = basename if len(basename) <= 36 else basename[:33] + "…"
            rows.append([(f"{mark} {short}", f"lt:{key}")])
        nav = []
        if page > 0:
            nav.append(("⬅️", f"lpage:{page - 1}"))
        if start + _LORA_PAGE_SIZE < len(all_loras):
            nav.append(("➡️", f"lpage:{page + 1}"))
        if nav:
            rows.append(nav)
        rows.append([("✏️ Multipliers", "loram:"), ("🗑 Clear all", "act:lora_clear")])
        rows.append([("◀️ Sections", "act:settings_hub"), ("◀️ Menu", "act:back")])
        total_pages = max(1, (len(all_loras) + _LORA_PAGE_SIZE - 1) // _LORA_PAGE_SIZE)
        lines.append(f"\n<i>Page {page + 1}/{total_pages} · tap to toggle</i>")
        self._show_panel(token, chat_id, "\n".join(lines), tg.inline_keyboard(rows))

    def _toggle_lora(self, token: str, chat_id: str, lora_key: str) -> None:
        settings = self._require_settings(token, chat_id)
        if not settings:
            return
        state = self._state(chat_id)
        basename = state.resolve_lora_key(lora_key)
        if not basename:
            self._show_loras_panel(token, chat_id, state.lora_page)
            return
        active, mults = self._engine.get_active_loras()
        if basename in active:
            active = [n for n in active if n != basename]
        else:
            active = active + [basename]
        self._engine.set_active_loras(active, mults)
        self._show_loras_panel(token, chat_id, state.lora_page)

    def _prompt_lora_multipliers(self, token: str, chat_id: str) -> None:
        if not self._require_settings(token, chat_id):
            return
        active, mults = self._engine.get_active_loras()
        state = self._state(chat_id)
        with state.lock:
            state.pending = {"mode": "loras_mults"}
        example = "1 0.8 1" if len(active) > 1 else "1"
        self._show_panel(
            token, chat_id,
            "✏️ <b>LoRA multipliers</b>\n\n"
            f"Active ({len(active)}): {_esc(', '.join(active) or 'none')}\n\n"
            f"Current: <code>{_esc(mults or example)}</code>\n\n"
            "Send space-separated values (one per active LoRA), or WanGP strings like <code>0;1</code>.",
            _cancel_keyboard(),
        )

    def _cmd_set(self, token: str, chat_id: str, argument: str) -> None:
        key, _, value = argument.partition(" ")
        key = key.strip().lower()
        value = value.strip()
        if key not in EDITABLE_SETTINGS:
            valid = ", ".join(EDITABLE_SETTINGS.keys())
            self._show_panel(
                token, chat_id,
                f"⚠️ Unknown setting.\nEditable: <code>{_esc(valid)}</code>",
                _back_keyboard(),
            )
            return
        if not value:
            self._prompt_for_setting(token, chat_id, key)
            return
        self._apply_setting(token, chat_id, key, value, offer_generate=True)

    def _cmd_abort(self, token: str, chat_id: str, _argument: str) -> None:
        _ok, message = self._engine.abort()
        self._show_panel(token, chat_id, f"⏹ {_esc(message)}", _main_keyboard(False))

    # ---------------------------------------------------------------- setting helpers

    def _prompt_for_setting(self, token: str, chat_id: str, key: str) -> None:
        meta = EDITABLE_SETTINGS.get(key)
        if not meta:
            return
        state = self._state(chat_id)
        with state.lock:
            state.pending = {"mode": f"set:{key}"}
        settings = self._engine.get_last_settings() or {}
        current = _esc(str(settings.get(key, "—")))
        self._show_panel(
            token, chat_id,
            f"✏️ <b>{_esc(meta['label'])}</b>\n\nCurrent: <code>{current}</code>\n\nSend the new value.",
            _cancel_keyboard(),
        )

    def _consume_pending(self, token: str, chat_id: str, pending: dict, text: str) -> None:
        state = self._state(chat_id)
        with state.lock:
            state.pending = None
        mode = pending.get("mode", "")
        if mode == "prompt":
            self._apply_setting(token, chat_id, "prompt", text, offer_generate=True)
        elif mode == "loras_mults":
            active, _ = self._engine.get_active_loras()
            self._engine.set_active_loras(active, text.strip())
            state = self._state(chat_id)
            self._show_loras_panel(token, chat_id, state.lora_page)
        elif mode.startswith("set:"):
            self._apply_setting(token, chat_id, mode[4:], text, offer_generate=True)

    def _apply_setting(self, token: str, chat_id: str, key: str, raw: str, *, offer_generate: bool) -> None:
        meta = EDITABLE_SETTINGS.get(key)
        if not meta:
            return
        base = self._engine.get_last_settings()
        if not base:
            self._show_panel(
                token, chat_id,
                "⚠️ No base settings yet. Generate once from the UI first.",
                _back_keyboard(),
            )
            return
        try:
            value = _coerce(meta["type"], raw)
        except (ValueError, TypeError):
            self._show_panel(
                token, chat_id,
                f"⚠️ <code>{_esc(raw)}</code> is not a valid {meta['type']}.",
                _back_keyboard(),
            )
            return
        base[key] = value
        self._engine.set_last_settings(base)
        shown = format_setting_display(key, value)
        confirm = f"✅ <b>{_esc(meta['label'])}</b> → <code>{_esc(shown)}</code>"
        if offer_generate:
            keyboard = tg.inline_keyboard([
                [("▶️ Generate", "act:generate"), ("🔁 New seed", "act:regen")],
                [("⚙️ Settings", "act:settings"), ("◀️ Menu", "act:back")],
            ])
        else:
            keyboard = _back_keyboard()
        self._show_panel(token, chat_id, confirm, keyboard)

    # ---------------------------------------------------------------- model picker

    def _show_model_picker(self, token: str, chat_id: str, page: int) -> None:
        models = self._engine.list_models()
        if not models:
            self._show_panel(token, chat_id, "⚠️ No models available.", _back_keyboard())
            return
        page_size = 8
        page = max(0, min(page, (len(models) - 1) // page_size))
        state = self._state(chat_id)
        with state.lock:
            state.model_page = page
        start = page * page_size
        chunk = models[start:start + page_size]
        rows = []
        for model_type, name in chunk:
            key = state.register_model_key(model_type)
            rows.append([(name[:60], f"model:{key}")])
        nav = []
        if page > 0:
            nav.append(("⬅️ Prev", f"mpage:{page - 1}"))
        if start + page_size < len(models):
            nav.append(("Next ➡️", f"mpage:{page + 1}"))
        if nav:
            rows.append(nav)
        rows.append(_back_cancel_row())
        self._show_panel(
            token, chat_id,
            f"<b>🧩 Choose a model</b> (page {page + 1}, {len(models)} total)",
            tg.inline_keyboard(rows),
        )

    def _select_model(self, token: str, chat_id: str, key: str) -> None:
        state = self._state(chat_id)
        model_type = state.resolve_model_key(key)
        if not model_type:
            self._show_panel(token, chat_id, "⚠️ Unknown model key, please reopen the model picker.", _back_keyboard())
            return
        base = self._engine.get_last_settings()
        if not base:
            try:
                base = dict(self._engine.wgp.get_default_settings(model_type))
            except Exception:
                base = {}
        base["model_type"] = model_type
        self._engine.set_last_settings(base)
        name = self._engine.model_display_name(model_type)
        keyboard = tg.inline_keyboard([[
            ("▶️ Generate", "act:generate"),
            ("✏️ Prompt", "act:prompt"),
            ("◀️ Back", "act:back"),
        ]])
        self._show_panel(token, chat_id, f"✅ Model set to <b>{_esc(name)}</b>.", keyboard)

    # ---------------------------------------------------------------- generation

    def _launch(self, token: str, chat_id: str, settings: dict, intro: str) -> None:
        ok, status_msg = self._engine.run_generation(
            settings,
            on_progress=lambda payload: self._on_progress(token, chat_id, payload),
            on_done=lambda result: self._on_done(token, chat_id, result),
            requester_chat_id=chat_id,
        )
        if not ok:
            self._show_panel(token, chat_id, f"⚠️ {_esc(status_msg)}", _main_keyboard(False))
            return
        self._show_panel(token, chat_id, intro, tg.inline_keyboard([[("⏹ Abort", "act:abort")]]))
        # Store the intro so progress updates can detect no-change.
        state = self._state(chat_id)
        with state.lock:
            state._last_progress_text = intro
            state._last_progress_edit = 0.0

    def _on_progress(self, token: str, chat_id: str, payload: dict) -> None:
        state = self._state(chat_id)
        now = time.time()
        with state.lock:
            last_edit = getattr(state, "_last_progress_edit", 0.0)
            if now - last_edit < 4.0:
                return
        text = self._format_progress(payload)
        with state.lock:
            last_text = getattr(state, "_last_progress_text", "")
            if text == last_text:
                return
        ok, _ = self._edit_panel(token, chat_id, text, tg.inline_keyboard([[("⏹ Abort", "act:abort")]]))
        if ok:
            with state.lock:
                state._last_progress_text = text
                state._last_progress_edit = now

    def _edit_panel(self, token: str, chat_id: str, text: str, keyboard: dict):
        state = self._state(chat_id)
        with state.lock:
            mid = state.panel_message_id
        if mid is None:
            return False, None
        return tg.edit_message_text(token, chat_id, mid, text, reply_markup=keyboard)

    def _format_progress(self, payload: dict) -> str:
        progress = payload.get("progress")
        status = str(payload.get("status") or "").strip()
        lines = ["⏳ <b>Generating…</b>"]
        step, total = payload.get("current_step"), payload.get("total_steps")
        if isinstance(step, int) and isinstance(total, int) and total > 0:
            pct = int(step / total * 100)
            filled = int(10 * pct / 100)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"[{bar}] {pct}% ({step}/{total})")
        elif isinstance(progress, int):
            filled = int(10 * progress / 100)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"[{bar}] {progress}%")
        if status:
            lines.append(f"<i>{_esc(status[:120])}</i>")
        return "\n".join(lines)

    def _on_done(self, token: str, chat_id: str, result: Any) -> None:
        success = bool(getattr(result, "success", False))
        files = list(getattr(result, "generated_files", []) or [])
        cfg = self._cfg()

        if success and files:
            latest = files[-1]
            caption = f"✅ <b>Generation complete</b>\n<code>{_esc(os.path.basename(latest))}</code>"
            if cfg.get("send_file") and latest:
                # Edit panel to "done" first, then send the file as a separate message.
                self._show_panel(token, chat_id, caption, _main_keyboard(False))
                ok, err = tg.send_file(token, chat_id, latest, caption, reply_markup=_main_keyboard(False))
                if not ok:
                    self._show_panel(token, chat_id, f"⚠️ Upload failed: {_esc(err)}", _main_keyboard(False))
            else:
                self._show_panel(token, chat_id, caption, _main_keyboard(False))
        elif success:
            self._show_panel(token, chat_id, "✅ Done (no file returned).", _main_keyboard(False))
        else:
            errors = getattr(result, "errors", []) or []
            detail = _esc(str(errors[0])[:200]) if errors else "unknown error"
            self._show_panel(token, chat_id, f"❌ <b>Generation failed</b>\n{detail}", _main_keyboard(False))

    # ---------------------------------------------------------------- public: update panel from plugin hooks

    def update_panel_for_chat(self, token: str, chat_id: str, text: str, keyboard: dict) -> None:
        """Called by the plugin to push a notification into the panel."""
        self._show_panel(token, chat_id, text, keyboard)
