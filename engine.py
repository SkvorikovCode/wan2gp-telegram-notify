"""Generation engine bridge for the Telegram bot.

This module is the only place that talks to the running WanGP instance. It:
  * captures the settings of the last UI / bot generation (so they can be
    re-run or tweaked),
  * enumerates the installed models and LoRAs,
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


# ---------------------------------------------------------------------------
# Settings catalogue
# ---------------------------------------------------------------------------

# Section → list of (key, meta). Keys match wgp.generate_video / models/_settings.json.
SETTINGS_SECTION_ORDER: tuple[str, ...] = (
    "General", "LoRAs", "Post Processing", "Quality", "Sliding Window", "Misc",
)
SETTINGS_SECTION_CODES: dict[str, str] = {
    "General": "gen",
    "LoRAs": "lora",
    "Post Processing": "post",
    "Quality": "qual",
    "Sliding Window": "slide",
    "Misc": "misc",
}
SETTINGS_CODE_TO_SECTION = {v: k for k, v in SETTINGS_SECTION_CODES.items()}

SETTINGS_SECTIONS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "General": [
        ("prompt",               {"label": "Prompt",            "type": "str"}),
        ("negative_prompt",      {"label": "Negative prompt",   "type": "str"}),
        ("seed",                 {"label": "Seed (-1=random)",  "type": "int",   "min": -1}),
        ("num_inference_steps",  {"label": "Steps",             "type": "int",   "min": 1, "max": 100}),
        ("guidance_scale",       {"label": "Guidance (CFG)",    "type": "float", "min": 1.0, "max": 20.0}),
        ("guidance2_scale",      {"label": "Guidance2 (CFG2)",  "type": "float", "min": 1.0, "max": 20.0}),
        ("guidance3_scale",      {"label": "Guidance3 (CFG3)",  "type": "float", "min": 1.0, "max": 20.0}),
        ("switch_threshold",     {"label": "Guidance switch threshold", "type": "int", "min": 0, "max": 1000}),
        ("switch_threshold2",    {"label": "Guidance switch threshold 2", "type": "int", "min": 0, "max": 1000}),
        ("guidance_phases",      {"label": "Guidance phases",   "type": "int",   "min": 0, "max": 3,
                                  "choices": ["0", "1", "2", "3"],
                                  "hint": "0=auto, 1=one phase, 2=two phases, 3=three phases"}),
        ("flow_shift",           {"label": "Flow shift",        "type": "float", "min": 1.0, "max": 25.0}),
        ("sample_solver",        {"label": "Sampler/Solver",    "type": "str",
                                  "choices": ["", "euler", "unipc", "dpm++"],
                                  "hint": "empty = default"}),
        ("NAG_scale",            {"label": "NAG Scale",         "type": "float", "min": 1.0, "max": 20.0}),
        ("NAG_tau",              {"label": "NAG Tau",           "type": "float", "min": 1.0, "max": 5.0}),
        ("NAG_alpha",            {"label": "NAG Alpha",         "type": "float", "min": 0.0, "max": 2.0}),
        ("video_length",         {"label": "Frames",            "type": "int",   "min": 5}),
        ("duration_seconds",     {"label": "Duration (sec)",    "type": "float", "min": 0.0,
                                  "hint": "0 = use frame count"}),
        ("resolution",           {"label": "Resolution",        "type": "str",
                                  "hint": "e.g. 832x480, 1280x720"}),
        ("batch_size",           {"label": "Batch size",        "type": "int",   "min": 1, "max": 16}),
        ("repeat_generation",    {"label": "Videos per prompt", "type": "int",   "min": 1, "max": 16}),
        ("denoising_strength",   {"label": "Denoising strength", "type": "float", "min": 0.0, "max": 1.0}),
        ("control_net_weight",   {"label": "ControlNet weight", "type": "float", "min": 0.0, "max": 2.0}),
        ("embedded_guidance_scale", {"label": "Embedded guidance", "type": "float", "min": 0.0, "max": 20.0}),
    ],
    "LoRAs": [],
    "Post Processing": [
        ("temporal_upsampling",  {"label": "Temporal upsampling", "type": "str",
                                  "choices": ["", "Rife x2 frames/s", "Rife x4 frames/s"],
                                  "hint": "empty = off"}),
        ("spatial_upsampling",   {"label": "Spatial upsampling",  "type": "str",
                                  "choices": ["", "vae1", "vae2"],
                                  "hint": "empty = off"}),
        ("film_grain_intensity", {"label": "Film grain intensity", "type": "float", "min": 0.0, "max": 1.0}),
        ("film_grain_saturation",{"label": "Film grain saturation","type": "float", "min": 0.0, "max": 1.0}),
    ],
    "Quality": [
        ("perturbation_switch",  {"label": "Perturbation",       "type": "switch",
                                  "hint": "requires guidance > 1"}),
        ("perturbation_start_perc", {"label": "Perturbation start %", "type": "int", "min": 0, "max": 100}),
        ("perturbation_end_perc",   {"label": "Perturbation end %",   "type": "int", "min": 0, "max": 100}),
        ("apg_switch",           {"label": "Adaptive projected guidance", "type": "switch",
                                  "hint": "requires guidance > 1"}),
        ("cfg_star_switch",      {"label": "CFG star",           "type": "switch"}),
    ],
    "Sliding Window": [
        ("sliding_window_size",  {"label": "Window size (frames)", "type": "int", "min": 5}),
        ("sliding_window_overlap",{"label": "Window overlap",      "type": "int", "min": 1}),
        ("sliding_window_discard_last_frames", {"label": "Discard last frames", "type": "int", "min": 0, "max": 20}),
        ("sliding_window_color_correction_strength", {"label": "Color correction strength", "type": "float", "min": 0.0, "max": 1.0}),
        ("sliding_window_overlap_noise", {"label": "Overlap noise", "type": "float", "min": 0.0, "max": 1.0}),
    ],
    "Misc": [
        ("RIFLEx_setting",       {"label": "RIFLEx setting",      "type": "int",
                                  "hint": "0 = auto"}),
        ("force_fps",            {"label": "FPS override",        "type": "str",
                                  "choices": ["", "auto", "control", "source", "8", "12", "16", "24"],
                                  "hint": "empty = model default"}),
        ("override_profile",     {"label": "Memory profile #",    "type": "int", "min": -1,
                                  "hint": "-1 = default profile"}),
        ("skip_steps_cache_type", {"label": "Skip-steps cache",   "type": "str",
                                  "choices": ["", "mag", "tea"],
                                  "hint": "empty = off"}),
        ("skip_steps_multiplier", {"label": "Skip-steps multiplier", "type": "float", "min": 1.0, "max": 3.0}),
        ("skip_steps_start_step_perc", {"label": "Skip-steps start %", "type": "int", "min": 0, "max": 100}),
        ("output_filename",      {"label": "Output filename",     "type": "str",
                                  "hint": "empty = auto naming"}),
    ],
}

# Flat dict for backwards-compat and quick lookup by key
EDITABLE_SETTINGS: dict[str, dict[str, Any]] = {
    key: meta
    for section_items in SETTINGS_SECTIONS.values()
    for key, meta in section_items
}


def _coerce(kind: str, raw: str) -> Any:
    raw = raw.strip()
    if kind == "switch":
        low = raw.lower()
        if low in ("1", "on", "true", "yes"):
            return 1
        if low in ("0", "off", "false", "no"):
            return 0
        raise ValueError(f"expected 0/1 or on/off, got {raw!r}")
    if kind == "int":
        return int(float(raw))
    if kind == "float":
        return float(raw)
    return raw


def format_setting_display(key: str, value: Any) -> str:
    """Human-readable value for Telegram panels."""
    meta = EDITABLE_SETTINGS.get(key, {})
    if meta.get("type") == "switch":
        return "ON" if int(value or 0) != 0 else "OFF"
    if value is None or value == "":
        return "—"
    shown = str(value)
    if len(shown) > 60:
        return shown[:57] + "…"
    return shown


def section_keys(section: str) -> list[str]:
    return [key for key, _ in SETTINGS_SECTIONS.get(section, [])]


_MAX_STORED_GENERATIONS = 50


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class GenerationEngine:
    """Thin, thread-safe facade over the live WanGP module + head-less API."""

    def __init__(self) -> None:
        self._wgp = None
        self._session = None
        self._session_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._generations: OrderedDict[str, dict] = OrderedDict()
        self._generations_lock = threading.Lock()
        self._ui_state: Optional[dict] = None
        self._requester_chat_id: Optional[str] = None

    def get_requester_chat_id(self) -> Optional[str]:
        return self._requester_chat_id

    # ------------------------------------------------------------------ wgp
    @property
    def wgp(self):
        if self._wgp is None:
            self._wgp = importlib.import_module("wgp")
        return self._wgp

    def _get_session(self):
        """Lazily create the API session, reusing the already-loaded model via the UI queue."""
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                api = importlib.import_module("shared.api")
                webui_state = self._ui_state if isinstance(self._ui_state, dict) else None
                self._session = api.init(console_output=False, webui_state=webui_state)
            return self._session

    # -------------------------------------------------------------- capture
    def remember_ui_state(self, state: dict) -> None:
        if isinstance(state, dict):
            self._ui_state = state
            with self._session_lock:
                if self._session is not None and not self._session._use_webui_queue:
                    self._session = None

    def store_generation(self, settings: dict) -> str:
        gen_id = _short_id()
        with self._generations_lock:
            self._generations[gen_id] = copy.deepcopy(settings)
            while len(self._generations) > _MAX_STORED_GENERATIONS:
                self._generations.popitem(last=False)
        return gen_id

    def get_generation(self, gen_id: str) -> Optional[dict]:
        with self._generations_lock:
            s = self._generations.get(gen_id)
            return copy.deepcopy(s) if s else None

    def get_last_settings(self) -> Optional[dict]:
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
        if not isinstance(gen, dict):
            return None
        settings_list = gen.get("file_settings_list") or []
        if settings_list:
            latest = settings_list[-1]
            if isinstance(latest, dict):
                return self.store_generation(latest)
        queue = gen.get("queue") or []
        if queue:
            task = queue[-1]
            if isinstance(task, dict):
                params = task.get("params") or task.get("settings")
                if isinstance(params, dict) and params:
                    return self.store_generation(params)
        return None

    # --------------------------------------------------------------- models
    def list_models(self) -> list[tuple[str, str]]:
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

    # --------------------------------------------------------------- LoRAs
    def list_loras(self, model_type: str) -> list[str]:
        """Return list of LoRA filenames available for the given model."""
        wgp = self.wgp
        try:
            lora_dir = wgp.get_lora_dir(model_type)
        except Exception:
            return []
        if not lora_dir or not os.path.isdir(lora_dir):
            return []
        result = []
        for fname in sorted(os.listdir(lora_dir)):
            if fname.lower().endswith((".safetensors", ".pt", ".bin", ".ckpt")):
                result.append(fname)
        return result

    def get_active_loras(self) -> tuple[list[str], str]:
        """Return (activated_loras, loras_multipliers) from last settings."""
        s = self.get_last_settings() or {}
        loras = s.get("activated_loras") or []
        mults = s.get("loras_multipliers") or ""
        # Normalise to basenames for display
        display = [os.path.basename(l) for l in loras]
        return display, str(mults)

    def set_active_loras(self, lora_basenames: list[str], multipliers: str) -> None:
        """Update activated_loras in last settings using basenames."""
        settings = self.get_last_settings()
        if settings is None:
            return
        model_type = settings.get("model_type", "")
        wgp = self.wgp
        try:
            lora_dir = wgp.get_lora_dir(model_type)
        except Exception:
            lora_dir = None

        full_paths: list[str] = []
        for name in lora_basenames:
            if lora_dir:
                full_paths.append(os.path.join(lora_dir, name))
            else:
                full_paths.append(name)

        settings["activated_loras"] = full_paths
        settings["loras_multipliers"] = multipliers
        self.set_last_settings(settings)

    # --------------------------------------------------------- i2v images
    @staticmethod
    def _add_unique_flags(value: str, flags: str) -> str:
        current = str(value or "")
        for ch in flags:
            if ch not in current:
                current += ch
        return current

    @staticmethod
    def _image_path_from_settings(settings: dict, key: str) -> Optional[str]:
        raw = settings.get(key)
        if isinstance(raw, list):
            for item in raw:
                path = str(item or "").strip()
                if path:
                    return path
            return None
        path = str(raw or "").strip()
        return path or None

    def get_image_paths_summary(self) -> tuple[Optional[str], Optional[str]]:
        settings = self.get_last_settings() or {}
        start = self._image_path_from_settings(settings, "image_start")
        end = self._image_path_from_settings(settings, "image_end")
        return (
            os.path.basename(start) if start else None,
            os.path.basename(end) if end else None,
        )

    def _tg_cache_dir(self) -> str:
        wgp = self.wgp
        save_path = str(getattr(wgp, "server_config", {}).get("save_path", "outputs") or "outputs")
        cache = os.path.join(save_path, "_tg_bot_cache")
        os.makedirs(cache, exist_ok=True)
        return cache

    def save_telegram_image(self, bot_token: str, file_id: str, chat_id: str, *, slot: str) -> tuple[bool, str]:
        from . import telegram_api as tg

        ok, result = tg.get_file(bot_token, file_id)
        if not ok or not isinstance(result, dict):
            return False, str(result or "getFile failed")
        tg_path = result.get("file_path")
        if not tg_path:
            return False, "Telegram returned no file path"

        ext = os.path.splitext(str(tg_path))[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            ext = ".jpg"
        local_name = f"tg_{chat_id}_{slot}_{int(time.time())}{ext}"
        dest = os.path.abspath(os.path.join(self._tg_cache_dir(), local_name))

        ok, err = tg.download_telegram_file(bot_token, tg_path, dest)
        if not ok:
            return False, err
        return True, dest

    def set_image_attachment(self, slot: str, path: str) -> tuple[bool, str]:
        """Attach a local image path as image_start or image_end."""
        slot = slot.strip().lower()
        if slot not in ("start", "end"):
            return False, "Invalid image slot."
        key = "image_start" if slot == "start" else "image_end"
        flag = "S" if slot == "start" else "E"

        settings = self.get_last_settings()
        if settings is None:
            return False, "No settings yet — pick a model or generate once in the UI."

        model_type = str(settings.get("model_type", "") or "")
        model_def = self.wgp.get_model_def(model_type) or {}
        allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
        if flag not in allowed:
            label = "Start image" if slot == "start" else "End image"
            return False, f"{label} is not supported for this model."

        settings[key] = os.path.abspath(path)
        ipt = str(settings.get("image_prompt_type", "") or "")
        settings["image_prompt_type"] = self._add_unique_flags(ipt, flag)
        self.set_last_settings(settings)
        return True, os.path.basename(path)

    def clear_image_attachment(self, slot: str) -> tuple[bool, str]:
        slot = slot.strip().lower()
        key = "image_start" if slot == "start" else "image_end"
        flag = "S" if slot == "start" else "E"
        settings = self.get_last_settings()
        if settings is None:
            return False, "No settings to update."
        settings[key] = None
        ipt = str(settings.get("image_prompt_type", "") or "").replace(flag, "")
        settings["image_prompt_type"] = ipt
        self.set_last_settings(settings)
        return True, f"Cleared {key}."

    # ----------------------------------------------------------- busy state
    def is_busy(self) -> bool:
        if self._gen_lock.locked():
            return True
        try:
            return bool(getattr(self.wgp, "gen_in_progress", False))
        except Exception:
            return False

    # ----------------------------------------------------------- abort path
    def abort(self) -> tuple[bool, str]:
        wgp = self.wgp
        aborted = False
        if self._session is not None:
            try:
                self._session.cancel()
                aborted = True
            except Exception:
                pass
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
        settings = copy.deepcopy(base)
        for key, value in (overrides or {}).items():
            settings[key] = value
        if new_seed:
            settings["seed"] = -1
        settings.pop("client_id", None)
        return settings

    def run_generation(
        self,
        settings: dict,
        *,
        gen_id: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        on_done: Optional[Callable[[Any], None]] = None,
        requester_chat_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        if self.is_busy():
            return False, "WanGP is busy with another generation."
        if not self._gen_lock.acquire(blocking=False):
            return False, "A bot generation is already running."

        if gen_id is None:
            gen_id = self.store_generation(settings)
        else:
            with self._generations_lock:
                self._generations[gen_id] = copy.deepcopy(settings)

        self._requester_chat_id = requester_chat_id

        def worker() -> None:
            callbacks = _EngineCallbacks(on_progress)
            try:
                session = self._get_session()
                job = session.submit_task(settings, callbacks=callbacks)
                result = job.result()
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
    def __init__(self, message: str) -> None:
        self.success = False
        self.generated_files: list[str] = []
        self.errors = [message]
        self.message = message


class _EngineCallbacks:
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
