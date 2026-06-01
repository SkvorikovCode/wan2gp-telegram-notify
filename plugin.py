import os
import threading
import time
import json
import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import telegram_api as tg
from .engine import GenerationEngine
from .bot import TelegramBot

PLUGIN_ID = "wan2gp-telegram-notify"

_BOT_TOKEN_KEY = "tg_bot_token"
_CHAT_ID_KEY = "tg_chat_id"
_SEND_FILE_KEY = "tg_send_file"
_ENABLED_KEY = "tg_enabled"
_BOT_ENABLED_KEY = "tg_bot_enabled"
_ALLOWED_IDS_KEY = "tg_allowed_chat_ids"

_UPDATE_INTERVAL = 5  # seconds between progress message edits


def _load_config(server_config: dict) -> dict:
    return {
        "enabled": bool(server_config.get(_ENABLED_KEY, False)),
        "bot_enabled": bool(server_config.get(_BOT_ENABLED_KEY, False)),
        "bot_token": str(server_config.get(_BOT_TOKEN_KEY, "") or ""),
        "chat_id": str(server_config.get(_CHAT_ID_KEY, "") or ""),
        "allowed_chat_ids": str(server_config.get(_ALLOWED_IDS_KEY, "") or ""),
        "send_file": bool(server_config.get(_SEND_FILE_KEY, False)),
    }


def _save_config(server_config, server_config_filename, *, enabled, bot_enabled, bot_token, chat_id, allowed_chat_ids, send_file):
    server_config[_ENABLED_KEY] = enabled
    server_config[_BOT_ENABLED_KEY] = bot_enabled
    server_config[_BOT_TOKEN_KEY] = bot_token.strip()
    server_config[_CHAT_ID_KEY] = chat_id.strip()
    server_config[_ALLOWED_IDS_KEY] = allowed_chat_ids.strip()
    server_config[_SEND_FILE_KEY] = send_file
    if server_config_filename:
        try:
            with open(server_config_filename, "w", encoding="utf-8") as f:
                json.dump(server_config, f, indent=4)
        except Exception as e:
            print(f"[TelegramNotify] Failed to save config: {e}")


def _format_progress_text(gen: dict, task_count: int) -> str:
    """Build a progress status line from gen state (used for UI notifications)."""
    progress_args = gen.get("last_progress_args")
    status = gen.get("status", "")

    lines = ["⏳ <b>Generation in progress...</b>"]

    if task_count > 1:
        prompt_no = gen.get("prompt_no", 1)
        lines.append(f"Task: {prompt_no}/{task_count}")

    if progress_args and len(progress_args) >= 2:
        first = progress_args[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            step, total = first
            pct = int(step / total * 100) if total > 0 else 0
            filled = int(10 * pct / 100)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"[{bar}] {pct}% ({step}/{total} steps)")
        status_text = str(progress_args[1]).strip() if len(progress_args) > 1 else ""
        if status_text:
            lines.append(f"<i>{status_text}</i>")
    elif status:
        lines.append(f"<i>{status}</i>")

    return "\n".join(lines)


class _ProgressWatcher:
    """Background thread that edits a Telegram message every N seconds."""

    def __init__(self, bot_token, chat_id, message_id, gen, task_count):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._message_id = message_id
        self._gen = gen
        self._task_count = task_count
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.wait(_UPDATE_INTERVAL):
            if not self._gen.get("in_progress"):
                break
            text = _format_progress_text(self._gen, self._task_count)
            tg.edit_message_text(self._bot_token, self._chat_id, self._message_id, text)


class TelegramNotifyPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Telegram Notify"
        self.version = "2.0.0"
        self.description = "Telegram notifications plus a two-way control bot (regenerate, change prompt/model/settings, abort)."
        self._watcher = None
        self._progress_message_id = None
        self._engine = GenerationEngine()
        self._bot = TelegramBot(self._get_cfg, self._engine)

    def setup_ui(self):
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.add_tab(tab_id=PLUGIN_ID, label="Telegram", component_constructor=self._build_ui)
        self.register_data_hook("on_generation_start", self._on_generation_start_hook)
        self.register_data_hook("on_generation_complete", self._on_generation_complete_hook)

    def post_ui_setup(self, components):
        # server_config is injected before post_ui_setup, so the bot can start correctly.
        self._maybe_start_bot()
        return {}

    def _get_cfg(self):
        sc = getattr(self, "server_config", None) or {}
        return _load_config(sc)

    def _maybe_start_bot(self):
        cfg = self._get_cfg()
        if cfg["bot_enabled"] and cfg["bot_token"]:
            self._bot.start()
        else:
            self._bot.stop()

    # ------------------------------------------------------------- hooks
    def _on_generation_start_hook(self, configs, **kwargs):
        gen = kwargs.get("gen", {})
        state = kwargs.get("state")
        if isinstance(state, dict):
            self._engine.remember_ui_state(state)

        cfg = self._get_cfg()
        if not cfg["enabled"] or not cfg["bot_token"] or not cfg["chat_id"]:
            return configs

        queue = kwargs.get("queue", [])
        task_count = len(queue)

        start_text = "⏳ <b>Generation in progress...</b>"
        if task_count > 1:
            start_text += f"\nTasks in queue: {task_count}"

        ok, result = tg.send_message(cfg["bot_token"], cfg["chat_id"], start_text)
        msg_id = result.get("message_id") if ok and isinstance(result, dict) else None
        if msg_id:
            self._progress_message_id = msg_id
            self._watcher = _ProgressWatcher(cfg["bot_token"], cfg["chat_id"], msg_id, gen, task_count)
        elif not ok:
            print(f"[TelegramNotify] Failed to send start message: {result}")
        return configs

    def _on_generation_complete_hook(self, configs, **kwargs):
        # Always capture the produced settings so the bot can regenerate them,
        # even when plain notifications are disabled.
        state = kwargs.get("state")
        if isinstance(state, dict):
            self._engine.remember_ui_state(state)
            try:
                gen = self._engine.wgp.get_gen_info(state)
                self._engine.capture_last_settings_from_gen(gen)
            except Exception:
                pass

        cfg = self._get_cfg()
        if not cfg["enabled"]:
            return configs

        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

        success = kwargs.get("success", False)
        file_list = kwargs.get("file_list", [])

        threading.Thread(
            target=self._do_notify,
            args=(cfg, success, file_list),
            daemon=True,
        ).start()
        return configs

    def _do_notify(self, cfg, success, file_list):
        bot_token = cfg["bot_token"]
        chat_id = cfg["chat_id"]
        send_file = cfg["send_file"]
        bot_enabled = cfg["bot_enabled"]
        msg_id = self._progress_message_id
        self._progress_message_id = None

        if success and file_list:
            latest = file_list[-1]
            caption = f"✅ <b>Generation complete</b>\n<code>{os.path.basename(latest)}</code>"
        elif success:
            caption = "✅ <b>Generation complete</b>"
        else:
            caption = "❌ <b>Generation failed</b>"

        # Offer quick actions when the control bot is live.
        keyboard = None
        if bot_enabled:
            keyboard = tg.inline_keyboard([
                [("🔁 Regenerate", "act:regen"), ("✏️ Prompt", "act:prompt")],
                [("🧩 Model", "act:model"), ("⚙️ Settings", "act:settings")],
            ])

        if msg_id:
            tg.edit_message_text(bot_token, chat_id, msg_id, caption, reply_markup=keyboard)

        if success and file_list and send_file:
            latest = file_list[-1]
            if os.path.isfile(latest):
                ok, err = tg.send_file(bot_token, chat_id, latest, os.path.basename(latest), reply_markup=keyboard)
                if not ok:
                    tg.send_message(bot_token, chat_id, f"⚠️ File upload failed: {err}")

        if not msg_id:
            tg.send_message(bot_token, chat_id, caption, reply_markup=keyboard)

    # ------------------------------------------------------------- UI
    def _build_ui(self):
        cfg = self._get_cfg()

        with gr.Column():
            gr.Markdown(
                "## Telegram Integration\n"
                "Send notifications when generation finishes, and optionally run a **two-way control bot** "
                "so you can regenerate, change the prompt/model/settings, and abort right from Telegram."
            )

            with gr.Accordion("Connection", open=True):
                bot_token_tb = gr.Textbox(
                    label="Bot Token",
                    value=cfg["bot_token"],
                    placeholder="123456789:AABBccDDee...",
                    type="password",
                )
                chat_id_tb = gr.Textbox(
                    label="Chat ID (primary)",
                    value=cfg["chat_id"],
                    placeholder="-100123456789 or your user ID",
                )
                allowed_ids_tb = gr.Textbox(
                    label="Additional authorized chat IDs (comma-separated, optional)",
                    value=cfg["allowed_chat_ids"],
                    placeholder="111111111, 222222222",
                )

            with gr.Accordion("Notifications", open=True):
                enabled_cb = gr.Checkbox(label="Enable completion notifications", value=cfg["enabled"])
                send_file_cb = gr.Checkbox(
                    label="Send the generated file (video/image), not just a text message",
                    value=cfg["send_file"],
                )

            with gr.Accordion("Control bot (two-way)", open=True):
                gr.Markdown(
                    "When enabled, the bot long-polls Telegram and accepts commands: "
                    "`/regen`, `/generate`, `/prompt`, `/model`, `/set`, `/settings`, `/abort`, `/status`.\n"
                    "Only authorized chat IDs can issue commands. Generations launched from Telegram reuse "
                    "the currently loaded model and appear in the normal gallery."
                )
                bot_enabled_cb = gr.Checkbox(
                    label="Enable control bot (accept commands from Telegram)",
                    value=cfg["bot_enabled"],
                )

            with gr.Row():
                save_btn = gr.Button("Save settings", variant="primary")
                test_btn = gr.Button("Send test message")

            status_md = gr.Markdown("")
            bot_status_md = gr.Markdown(self._bot_status_line())

        def save(enabled, bot_enabled, bot_token, chat_id, allowed_chat_ids, send_file):
            sc = getattr(self, "server_config", None) or {}
            scf = getattr(self, "server_config_filename", "") or ""
            _save_config(
                sc, scf,
                enabled=enabled, bot_enabled=bot_enabled, bot_token=bot_token,
                chat_id=chat_id, allowed_chat_ids=allowed_chat_ids, send_file=send_file,
            )
            self._maybe_start_bot()
            return "✅ Settings saved.", self._bot_status_line()

        def test_notify(bot_token, chat_id):
            token = bot_token.strip()
            cid = chat_id.strip()
            if not token or not cid:
                return "⚠️ Please enter Bot Token and Chat ID first."
            ok, result = tg.send_message(token, cid, "✅ WAN2GP Telegram — test message!")
            return "✅ Test message sent!" if ok else f"❌ Failed: {result}"

        save_btn.click(
            fn=save,
            inputs=[enabled_cb, bot_enabled_cb, bot_token_tb, chat_id_tb, allowed_ids_tb, send_file_cb],
            outputs=[status_md, bot_status_md],
        )
        test_btn.click(fn=test_notify, inputs=[bot_token_tb, chat_id_tb], outputs=[status_md])

    def _bot_status_line(self):
        return "🟢 Control bot is running." if self._bot.running else "⚪ Control bot is stopped."
