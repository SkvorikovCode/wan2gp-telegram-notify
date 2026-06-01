"""Generation engine bridge for the Telegram bot.

This module is the only place that talks to the running WanGP instance. It:
  * captures the settings of the last UI / bot generation (so they can be
    re-run or tweaked),
  * enumerates the installed models,
  * launches a head-less generation through ``shared.api`` (which reuses the
    already-loaded model and writes into the normal output folder), and
  * aborts a running generation.

Everything here is designed to be called from the bot's background thread.
The single shared GPU/model means we serialise bot generations behind a lock
and refuse to start one while the web UI is busy.
"""

from __future__ import annotations

import copy
import importlib
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Optional


# Settings keys that the bot is allowed to tweak by name, with a coercer and a
# short human label. Anything not in here is left exactly as the source task
# had it, so we never silently break model-specific parameters.
EDITABLE_SETTINGS: dict[str, dict[str, Any]] = {
    "prompt": {"label": "Prompt", "type": "str"},
    "negative_prompt": {"label": "Negative prompt", "type": "str"},
    "seed": {"label": "Seed", "type": "int"},
    "num_inference_steps": {"label": "Steps", "type": "int"},
    "guidance_scale": {"label": "Guidance", "type": "float"},
    "flow_shift": {"label": "Flow shift", "type": "float"},
    "video_length": {"label": "Frames", "type": "int"},
    "resolution": {"label": "Resolution", "type": "str"},
}


def _coerce(kind: str, raw: str) -> Any:
    raw = raw.strip()
    if kind == "int":
        return int(float(raw))
    if kind == "float":
        return float(raw)
    return raw


_MAX_STORED_GENERATIONS = 50


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class GenerationEngine:
    """Thin, thread-safe facade over the live WanGP module + head-less API."""

    def __init__(self) -> None:
        self._wgp = None
        self._session = None
        self._session_lock = threading.Lock()
        # Guards a single bot-launched generation at a time.
        self._gen_lock = threading.Lock()
        # Per-generation settings store: gen_id -> settings (capped at _MAX_STORED_GENERATIONS).
        self._generations: OrderedDict[str, dict] = OrderedDict()
        self._generations_lock = threading.Lock()
        # Live per-session state captured from the generation hooks; lets us
        # abort a UI-launched generation and read its outputs.
        self._ui_state: Optional[dict] = None

    # ------------------------------------------------------------------ wgp
    @property
    def wgp(self):
        if self._wgp is None:
            # wgp registers itself as sys.modules["wgp"] at import time, so this
            # returns the already-running module rather than re-importing it.
            self._wgp = importlib.import_module("wgp")
        return self._wgp

    def _get_session(self):
        """Lazily create the head-less API session (shares the loaded model)."""
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                api = importlib.import_module("shared.api")
                # No output_dir override: inherit the UI's configured save
                # paths so bot results land in the same gallery.
                self._session = api.init(console_output=False)
            return self._session

    # -------------------------------------------------------------- capture
    def remember_ui_state(self, state: dict) -> None:
        if isinstance(state, dict):
            self._ui_state = state

    def store_generation(self, settings: dict) -> str:
        """Store settings for a generation and return its gen_id."""
        gen_id = _short_id()
        with self._generations_lock:
            self._generations[gen_id] = copy.deepcopy(settings)
            # Evict oldest entries beyond the cap.
            while len(self._generations) > _MAX_STORED_GENERATIONS:
                self._generations.popitem(last=False)
        return gen_id

    def get_generation(self, gen_id: str) -> Optional[dict]:
        with self._generations_lock:
            s = self._generations.get(gen_id)
            return copy.deepcopy(s) if s else None

    def get_last_settings(self) -> Optional[dict]:
        """Return the most recently stored settings (for backwards compat)."""
        with self._generations_lock:
            if not self._generations:
                return None
            return copy.deepcopy(next(reversed(self._generations.values())))

    def get_last_gen_id(self) -> Optional[str]:
        with self._generations_lock:
            if not self._generations:
                return None
            return next(reversed(self._generations))

    def set_last_settings(self, settings: dict) -> str:
        """Update the most recent generation's settings (or create a new entry). Returns gen_id."""
        with self._generations_lock:
            if self._generations:
                last_id = next(reversed(self._generations))
                self._generations[last_id] = copy.deepcopy(settings)
                return last_id
        return self.store_generation(settings)

    def has_last_settings(self) -> bool:
        with self._generations_lock:
            return bool(self._generations)

    def capture_last_settings_from_gen(self, gen: dict) -> Optional[str]:
        """Pull settings from the most recently produced output. Returns gen_id or None."""
        if not isinstance(gen, dict):
            return None
        settings_list = gen.get("file_settings_list") or []
        if not settings_list:
            return None
        latest = settings_list[-1]
        if not isinstance(latest, dict):
            return None
        return self.store_generation(latest)

    # --------------------------------------------------------------- models
    def list_models(self) -> list[tuple[str, str]]:
        """Return (model_type, display_name) for every visible model."""
        wgp = self.wgp
        result: list[tuple[str, str]] = []
        try:
            model_types = list(getattr(wgp, "displayed_model_types", []) or [])
        except Exception:
            model_types = []
        for model_type in model_types:
            try:
                name = wgp.get_model_name(model_type)
            except Exception:
                name = model_type
            result.append((model_type, name))
        result.sort(key=lambda item: item[1].lower())
        return result

    def model_display_name(self, model_type: str) -> str:
        try:
            return self.wgp.get_model_name(model_type)
        except Exception:
            return model_type

    # ----------------------------------------------------------- busy state
    def is_busy(self) -> bool:
        """True if the UI or a bot job currently holds the GPU/model."""
        if self._gen_lock.locked():
            return True
        try:
            return bool(getattr(self.wgp, "gen_in_progress", False))
        except Exception:
            return False

    # ----------------------------------------------------------- abort path
    def abort(self) -> tuple[bool, str]:
        """Signal the running generation (UI or bot) to stop."""
        wgp = self.wgp
        aborted = False
        # Head-less bot job: cancel through the session.
        if self._session is not None:
            try:
                self._session.cancel()
                aborted = True
            except Exception:
                pass
        # UI-launched job: flip the same flags the abort button uses.
        state = self._ui_state
        if isinstance(state, dict):
            try:
                gen = wgp.get_gen_info(state)
                gen["abort"] = True
                gen["resume"] = True
                if getattr(wgp, "wan_model", None) is not None:
                    wgp.wan_model._interrupt = True
                aborted = True
            except Exception:
                pass
        else:
            try:
                if getattr(wgp, "wan_model", None) is not None:
                    wgp.wan_model._interrupt = True
                    aborted = True
            except Exception:
                pass
        return (aborted, "Abort signal sent." if aborted else "Nothing is generating.")

    # --------------------------------------------------------- run a job
    def build_settings(self, base: dict, overrides: dict, new_seed: bool) -> dict:
        """Clone ``base`` settings, apply overrides, optionally randomise seed."""
        settings = copy.deepcopy(base)
        for key, value in (overrides or {}).items():
            settings[key] = value
        if new_seed:
            settings["seed"] = -1  # wgp interprets -1 as "pick a fresh seed"
        # Strip per-run identity so the API assigns a clean client id.
        settings.pop("client_id", None)
        return settings

    def run_generation(
        self,
        settings: dict,
        *,
        gen_id: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        on_done: Optional[Callable[[Any], None]] = None,
    ) -> tuple[bool, str]:
        """Launch a head-less generation in a worker thread.

        Returns (accepted, gen_id_or_error). ``accepted`` is False if something
        is already running. Progress/results are delivered through callbacks.
        """
        if self.is_busy():
            return False, "WanGP is busy with another generation."
        if not self._gen_lock.acquire(blocking=False):
            return False, "A bot generation is already running."

        if gen_id is None:
            gen_id = self.store_generation(settings)
        else:
            # Ensure the settings are stored under this gen_id.
            with self._generations_lock:
                self._generations[gen_id] = copy.deepcopy(settings)

        def worker() -> None:
            callbacks = _EngineCallbacks(on_progress)
            try:
                session = self._get_session()
                job = session.submit_task(settings, callbacks=callbacks)
                result = job.result()
                # Keep the stored settings up to date with what actually ran.
                with self._generations_lock:
                    self._generations[gen_id] = copy.deepcopy(settings)
                if on_done is not None:
                    on_done(result)
            except Exception as exc:
                if on_done is not None:
                    on_done(_FailureResult(str(exc)))
            finally:
                self._gen_lock.release()

        threading.Thread(target=worker, daemon=True, name="tg-bot-generation").start()
        return True, gen_id


class _FailureResult:
    """Stand-in result when the job never produced a GenerationResult."""

    def __init__(self, message: str) -> None:
        self.success = False
        self.generated_files: list[str] = []
        self.errors = [message]
        self.message = message


class _EngineCallbacks:
    """Adapts shared.api callback methods into a single throttled progress fn."""

    def __init__(self, on_progress: Optional[Callable[[dict], None]]) -> None:
        self._on_progress = on_progress
        self._last_emit = 0.0

    def _emit(self, payload: dict, *, force: bool = False) -> None:
        if self._on_progress is None:
            return
        now = time.time()
        if not force and now - self._last_emit < 1.0:
            return
        self._last_emit = now
        try:
            self._on_progress(payload)
        except Exception:
            pass

    def on_progress(self, update: Any) -> None:
        self._emit(
            {
                "phase": getattr(update, "phase", ""),
                "status": getattr(update, "status", ""),
                "progress": getattr(update, "progress", 0),
                "current_step": getattr(update, "current_step", None),
                "total_steps": getattr(update, "total_steps", None),
            }
        )

    def on_status(self, text: Any) -> None:
        self._emit({"status": str(text or ""), "progress": None}, force=False)
