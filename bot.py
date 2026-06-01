"""Telegram bot for WanGP — group-chat edition.

One persistent "panel" message per chat that is always edited in-place.
User command messages are deleted immediately after processing.
Navigation is fully inline-keyboard driven; /cancel returns to the main menu
from any nested screen.

Supported commands (also available as inline buttons):
  /start, /help    – show main menu
  /status          – current state
  /regen           – re-run last generation with a new seed
  /generate        – run with current settings
  /prompt <text>   – set prompt (or opens text-input mode)
  /model           – model picker (paginated)
  /settings        – editable settings list
  /set key value   – change one setting
  /abort           – stop running generation
  /cancel          – cancel any pending input and return to main menu
"""

from __future__ import annotations

import html
import threading
import time
from typing import Any, Optional

from . import telegram_api as tg
from .engine import EDITABLE_SETTINGS, GenerationEngine, _coerce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _main_keyboard(busy: bool) -> dict:
    rows = [
        [("🔁 Regenerate", "act:regen"), ("▶️ Generate", "act:generate")],
        [("✏️ Prompt", "act:prompt"), ("🧩 Model", "act:model")],
        [("⚙️ Settings", "act:settings"), ("📊 Status", "act:status")],
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
        self._thread = threading.Thread(target=self._run, daemon=True, name="tg-bot-poll")
        self._thread.start()

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
            ok, _ = tg.edit_message_text(token, chat_id, mid, text, reply_markup=keyboard)
            if ok:
                return
            # Message was deleted externally — fall through to send a new one.
        ok, result = tg.send_message(token, chat_id, text, reply_markup=keyboard)
        if ok and isinstance(result, dict):
            with state.lock:
                state.panel_message_id = result.get("message_id")

    def _delete_message(self, token: str, chat_id: str, message_id: int) -> None:
        tg.delete_message(token, chat_id, message_id)

    # ---------------------------------------------------------------- main loop

    def _run(self) -> None:
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
        if not chat_id:
            return

        # Always delete the user's message to keep the chat clean.
        if msg_id:
            self._delete_message(token, chat_id, msg_id)

        if not self._is_authorized(chat_id):
            tg.send_message(token, chat_id, "⛔ This chat is not authorized.")
            return

        state = self._state(chat_id)

        # /cancel always wins.
        if text.lstrip("/").lower().split("@")[0] == "cancel":
            with state.lock:
                state.pending = None
            self._show_main_menu(token, chat_id)
            return

        # Free-text follow-up (prompt entry or /set value).
        pending = state.pending
        if pending and not text.startswith("/"):
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
            "status":     self._cmd_status,
            "regen":      self._cmd_regen,
            "regenerate": self._cmd_regen,
            "generate":   self._cmd_generate,
            "prompt":     self._cmd_prompt,
            "model":      lambda t, c, a: self._show_model_picker(t, c, 0),
            "settings":   self._cmd_settings,
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
        elif action == "prompt":
            self._cmd_prompt(token, chat_id, "")
        elif action == "model":
            self._show_model_picker(token, chat_id, 0)
        elif action == "settings":
            self._cmd_settings(token, chat_id, "")
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
        lines.append("🟡 Busy generating…" if busy else "🟢 Idle")
        self._show_panel(token, chat_id, "\n".join(lines), _main_keyboard(busy))

    # ---------------------------------------------------------------- commands

    def _cmd_help(self, token: str, chat_id: str, _argument: str) -> None:
        self._show_main_menu(token, chat_id)

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

    def _cmd_settings(self, token: str, chat_id: str, _argument: str) -> None:
        settings = self._engine.get_last_settings()
        if not settings:
            self._show_panel(
                token, chat_id,
                "⚠️ No settings captured yet. Generate once from the UI first.",
                _back_keyboard(),
            )
            return
        lines = ["<b>⚙️ Editable settings</b>"]
        rows = []
        for key, meta in EDITABLE_SETTINGS.items():
            value = settings.get(key, "—")
            shown = str(value)
            if len(shown) > 50:
                shown = shown[:47] + "…"
            lines.append(f"• <b>{_esc(meta['label'])}</b>: {_esc(shown)}")
            rows.append([(f"✏️ {meta['label']}", f"set:{key}")])
        rows.append(_back_cancel_row())
        self._show_panel(token, chat_id, "\n".join(lines), tg.inline_keyboard(rows))

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
        confirm = f"✅ <b>{_esc(meta['label'])}</b> set to <code>{_esc(value)}</code>."
        if offer_generate:
            keyboard = tg.inline_keyboard([[
                ("▶️ Generate", "act:generate"),
                ("🔁 New seed", "act:regen"),
                ("◀️ Back", "act:back"),
            ]])
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
        rows = [[(name[:60], f"model:{model_type}")] for model_type, name in chunk]
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

    def _select_model(self, token: str, chat_id: str, model_type: str) -> None:
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
            import os
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
