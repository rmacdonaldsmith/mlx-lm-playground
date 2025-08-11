#!/usr/bin/env python3
"""
mlxchat: streaming CLI chat for local MLX-LM models with memory controls.

Features
- Streaming tokens (typing effect) via mlx_lm.stream_generate when available
- CLI flags: --model, --temp, --top-p, --max-tokens, --system
- Memory/KV controls: --max-kv-size, --kv-bits, --kv-group-size, --quantized-kv-start, --clear-cache-after
- Config files with model aliases:
    ~/.config/mlxchat/config.toml  (user defaults)
    ./mlxchat.toml                 (project overrides)
- REPL slash-commands:
    /help, /reset, /system, /temp, /top_p, /max, /model
    /mem, /clear, /kv, /max_kv

Quick start:
  python mlxchat.py --init-config
  python mlxchat.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# --- TOML loader (stdlib on 3.11+) ---
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

# --- MLX / MLX-LM imports with graceful fallbacks ---
import mlx.core as mx  # type: ignore

from mlx_lm import load, generate  # type: ignore
try:
    from mlx_lm import stream_generate  # type: ignore[attr-defined]
except Exception:
    stream_generate = None

# Prompt cache factory (handle older/newer paths)
try:
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore
except Exception:
    try:
        from mlx_lm.cache_utils import make_prompt_cache  # type: ignore
    except Exception:
        make_prompt_cache = None  # type: ignore

from mlx_lm.sample_utils import make_sampler  # type: ignore


# -------------------- Config & defaults --------------------

DEFAULT_ALIASES: Dict[str, str] = {
    "llama3-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit",
    "llama3-70b": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

DEFAULTS: Dict[str, object] = {
    "model": "llama3-8b",
    "temp": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "system": "",
    # Memory/KV management
    "max_kv_size": 4096,       # rotating KV cache (0 disables rotation)
    "kv_bits": None,           # e.g., 4 to quantize KV cache; None disables
    "kv_group_size": 64,
    "quantized_kv_start": 0,   # start step for quantized KV (0 = from start)
    "clear_cache_after": False # clear MLX caches after each reply
}

USER_CFG = Path.home() / ".config" / "mlxchat" / "config.toml"
PROJECT_CFG = Path.cwd() / "mlxchat.toml"


def read_toml(path: Path) -> Dict:
    if not path.exists():
        return {}
    if tomllib is None:
        print(f"Warning: tomllib not available; ignoring {path}", file=sys.stderr)
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def merged_config() -> Dict:
    cfg_user = read_toml(USER_CFG)
    cfg_proj = read_toml(PROJECT_CFG)

    # Environment overrides
    env: Dict[str, Optional[str]] = {
        "model": os.getenv("MLXCHAT_MODEL"),
        "temp": os.getenv("MLXCHAT_TEMP"),
        "top_p": os.getenv("MLXCHAT_TOP_P"),
        "max_tokens": os.getenv("MLXCHAT_MAX_TOKENS"),
        "system": os.getenv("MLXCHAT_SYSTEM"),
        "max_kv_size": os.getenv("MLXCHAT_MAX_KV_SIZE"),
        "kv_bits": os.getenv("MLXCHAT_KV_BITS"),
        "kv_group_size": os.getenv("MLXCHAT_KV_GROUP_SIZE"),
        "quantized_kv_start": os.getenv("MLXCHAT_QUANTIZED_KV_START"),
        "clear_cache_after": os.getenv("MLXCHAT_CLEAR_CACHE_AFTER"),
    }

    def coerce(name: str, v):
        if v is None:
            return None
        try:
            if name in {"temp", "top_p"}:
                return float(v)
            if name in {"max_tokens", "max_kv_size", "kv_group_size", "quantized_kv_start"}:
                return int(v)
            if name == "kv_bits":
                if str(v).strip().lower() in {"", "none"}:
                    return None
                return int(v)
            if name == "clear_cache_after":
                if isinstance(v, bool):
                    return v
                return str(v).lower() in {"1", "true", "t", "yes", "y"}
        except Exception:
            return None
        return v

    # Merge: defaults -> user -> project -> env
    settings = DEFAULTS.copy()

    def overlay(d: Dict):
        if not isinstance(d, dict):
            return
        for k in DEFAULTS.keys():
            if k in d and d[k] is not None:
                settings[k] = d[k]

    overlay(cfg_user)
    overlay(cfg_proj)
    overlay({k: coerce(k, v) for k, v in env.items()})

    aliases = DEFAULT_ALIASES.copy()
    aliases.update(cfg_user.get("aliases", {}))
    aliases.update(cfg_proj.get("aliases", {}))

    return {"aliases": aliases, **settings}


def init_default_config(force: bool = False) -> Path:
    target = USER_CFG
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target
    text = """# mlxchat config (user)
# You can override any of these in a project-local ./mlxchat.toml

# Defaults
model = "llama3-8b"
temp = 0.7
top_p = 0.9
max_tokens = 512
system = ""

# Memory / KV cache
max_kv_size = 4096
kv_bits = 4         # set to 4 for quantized KV, or comment/remove to disable
kv_group_size = 64
quantized_kv_start = 0
clear_cache_after = false

# Model aliases
[aliases]
llama3-8b  = "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit"
llama3-70b = "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"
qwen2.5-7b = "mlx-community/Qwen2.5-7B-Instruct-4bit"
mistral-7b = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
"""
    target.write_text(text)
    return target


def resolve_model(name_or_id: str, aliases: Dict[str, str]) -> str:
    return aliases.get(name_or_id, name_or_id)


# -------------------- Engine --------------------

@dataclass
class Params:
    temp: float
    top_p: float
    max_tokens: int
    system: str
    model_id: str
    max_kv_size: int
    kv_bits: Optional[int]
    kv_group_size: int
    quantized_kv_start: int
    clear_cache_after: bool


class Engine:
    def __init__(self, params: Params):
        self.params = params
        self._load_model()
        self.messages: List[Dict[str, str]] = []
        if self.params.system:
            self.messages.append({"role": "system", "content": self.params.system})

    def _load_model(self) -> None:
        print(f"Loading model: {self.params.model_id} …", flush=True)
        self.model, self.tokenizer = load(self.params.model_id)
        self._make_cache()

    def _make_cache(self) -> None:
        self.prompt_cache = None
        if make_prompt_cache is None:
            return
        try:
            # Newer API can accept max_kv_size
            if self.params.max_kv_size and self.params.max_kv_size > 0:
                self.prompt_cache = make_prompt_cache(self.model, max_kv_size=int(self.params.max_kv_size))  # type: ignore
            else:
                self.prompt_cache = make_prompt_cache(self.model)  # type: ignore
        except TypeError:
            # Older API: no max_kv_size arg
            self.prompt_cache = make_prompt_cache(self.model)  # type: ignore

    def set_system(self, text: str) -> None:
        self.params.system = text
        if self.messages and self.messages[0].get("role") == "system":
            if text:
                self.messages[0]["content"] = text
            else:
                self.messages.pop(0)
        elif text:
            self.messages.insert(0, {"role": "system", "content": text})

    def reset(self) -> None:
        self.messages.clear()
        if self.params.system:
            self.messages.append({"role": "system", "content": self.params.system})

    def change_model(self, model_id: str) -> None:
        self.params.model_id = model_id
        self._load_model()
        self.reset()

    def _sampler(self):
        return make_sampler(temp=float(self.params.temp), top_p=float(self.params.top_p))

    def _gen_kwargs(self, prompt: str) -> Dict:
        kw: Dict[str, object] = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": int(self.params.max_tokens),
            "sampler": self._sampler(),
        }
        if self.prompt_cache is not None:
            kw["prompt_cache"] = self.prompt_cache

        # KV rotation size (supported on newer mlx-lm)
        if self.params.max_kv_size and self.params.max_kv_size > 0:
            kw["max_kv_size"] = int(self.params.max_kv_size)

        # KV quantization knobs (supported on newer mlx-lm)
        if self.params.kv_bits is not None:
            kw["kv_bits"] = int(self.params.kv_bits)
            kw["kv_group_size"] = int(self.params.kv_group_size)
            kw["quantized_kv_start"] = int(self.params.quantized_kv_start)

        return kw

    def _strip_unknown_kv_args(self, kw: Dict) -> Dict:
        bad = {"kv_bits", "kv_group_size", "quantized_kv_start", "max_kv_size"}
        return {k: v for k, v in kw.items() if k not in bad}

    def _try_stream_tokens(self, kw: Dict) -> Iterator[str]:
        """
        Yield tokens using stream_generate if available, else one-shot generate.
        If the installed mlx-lm doesn't support KV kwargs, retry without them.
        """
        # First attempt
        try:
            if stream_generate is not None:
                for resp in stream_generate(**kw):  # type: ignore
                    yield resp.text
                return
            else:
                yield generate(**kw)  # type: ignore
                return
        except TypeError as e:
            # Retry after stripping KV-related keys if error mentions them
            msg = str(e)
            if any(k in msg for k in ("kv_bits", "kv_group_size", "quantized_kv_start", "max_kv_size")):
                kw2 = self._strip_unknown_kv_args(kw)
                if stream_generate is not None:
                    for resp in stream_generate(**kw2):  # type: ignore
                        yield resp.text
                else:
                    yield generate(**kw2)  # type: ignore
                return
            raise

    def _clear_cache(self) -> None:
        # Best-effort clear across MLX versions
        try:
            mx.clear_cache()  # recent MLX
        except Exception:
            try:
                mx.metal.clear_cache()  # older MLX
            except Exception:
                pass
        try:
            mx.reset_peak_memory()  # not always available
        except Exception:
            pass

    def stream_reply(self, user_text: str) -> Iterator[str]:
        """
        Stream assistant reply tokens. Appends the final assistant message
        to the conversation and optionally clears caches after.
        """
        if not user_text.strip():
            yield "Please enter something non-empty."
            return

        self.messages.append({"role": "user", "content": user_text})
        prompt = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True)
        kw = self._gen_kwargs(prompt)

        chunks: List[str] = []
        for tok in self._try_stream_tokens(kw):
            chunks.append(tok)
            yield tok

        full = "".join(chunks)
        self.messages.append({"role": "assistant", "content": full})

        if self.params.clear_cache_after:
            self._clear_cache()


# -------------------- CLI / REPL --------------------

HELP_TEXT = """\
Slash commands:
  /help                 Show this help
  /reset                Clear conversation context (keeps system message)
  /system <text>        Set/replace system prompt (use /system to clear)
  /temp <float>         Set temperature (e.g., 0.1–1.5)
  /top_p <float>        Set nucleus sampling top_p (0–1)
  /max <int>            Set max new tokens
  /model <alias|id>     Switch model by alias or full ID
  /mem                  Show active/cache/peak memory (if supported by MLX)
  /clear                Clear MLX caches and reset peak memory
  /kv <bits> <group> [start]   Quantize KV cache (e.g., /kv 4 64 0). Use '/kv none' to disable.
  /max_kv <int>         Set rotating KV cache size (0 disables)
  /config               Show current runtime config
  /aliases              List configured model aliases
  /system?              Show current system prompt


Notes:
- Model aliases come from ~/.config/mlxchat/config.toml and ./mlxchat.toml
- Ctrl-C or 'exit'/'quit' to leave
"""
def snapshot_config(eng: "Engine", aliases: Dict[str, str]) -> Dict[str, object]:
    sysmsg = eng.params.system or ""
    sys_preview = (sysmsg[:120] + "…") if len(sysmsg) > 120 else sysmsg
    return {
        "model_id": eng.params.model_id,
        "temp": eng.params.temp,
        "top_p": eng.params.top_p,
        "max_tokens": eng.params.max_tokens,
        "system_len": len(sysmsg),
        "system_preview": sys_preview,
        "max_kv_size": eng.params.max_kv_size,
        "kv_bits": eng.params.kv_bits,
        "kv_group_size": eng.params.kv_group_size,
        "quantized_kv_start": eng.params.quantized_kv_start,
        "clear_cache_after": eng.params.clear_cache_after,
        "streaming_available": (stream_generate is not None),
        "prompt_cache_enabled": (eng.prompt_cache is not None),
        "turns_in_history": sum(1 for m in eng.messages if m["role"] in {"user", "assistant"}) // 2,
        "aliases_count": len(aliases),
    }

def print_config_diagnostic(eng: "Engine", aliases: Dict[str, str]) -> None:
    snap = snapshot_config(eng, aliases)
    lines = [
        "— Current config —",
        f"model_id            : {snap['model_id']}",
        f"temp/top_p          : {snap['temp']}/{snap['top_p']}",
        f"max_tokens          : {snap['max_tokens']}",
        f"system(len)         : {snap['system_len']}",
        f"system(preview)     : {snap['system_preview']!r}",
        f"max_kv_size         : {snap['max_kv_size']}",
        f"kv_bits/group/start : {snap['kv_bits']}/{snap['kv_group_size']}/{snap['quantized_kv_start']}",
        f"clear_cache_after   : {snap['clear_cache_after']}",
        f"streaming_available : {snap['streaming_available']}",
        f"prompt_cache_enabled: {snap['prompt_cache_enabled']}",
        f"turns_in_history    : {snap['turns_in_history']}",
        f"aliases(configured) : {snap['aliases_count']}",
    ]
    print("\n".join(lines))

def print_aliases(aliases: Dict[str, str]) -> None:
    if not aliases:
        print("(no aliases configured)")
        return
    width = max(len(k) for k in aliases.keys())
    print("— Model aliases —")
    for k, v in sorted(aliases.items()):
        print(f"{k.ljust(width)} -> {v}")

def print_system_prompt(eng: "Engine") -> None:
    s = eng.params.system or ""
    if not s:
        print("(system prompt: NONE)")
        return
    bar = "-" * 60
    print("— System prompt —")
    print(bar)
    print(s)
    print(bar)
    print(f"(length: {len(s)} chars)")

def parse_args(cfg: Dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming local chat for MLX-LM models")
    p.add_argument("--init-config", action="store_true", help="Create ~/.config/mlxchat/config.toml with sensible defaults")
    p.add_argument("--force", action="store_true", help="Overwrite config when used with --init-config")
    p.add_argument("--model", default=cfg["model"], help="Model alias or full identifier")
    p.add_argument("--temp", type=float, default=cfg["temp"], help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=cfg["top_p"], help="Top-p (nucleus) sampling")
    p.add_argument("--max-tokens", type=int, default=cfg["max_tokens"], help="Maximum new tokens to generate")
    p.add_argument("--system", default=cfg["system"], help="System prompt")

    # Memory / KV flags
    p.add_argument("--max-kv-size", type=int, default=cfg["max_kv_size"],
                   help="Maximum KV cache size (rotating cache). 0 disables rotation")
    p.add_argument("--kv-bits", type=int, nargs="?", default=cfg["kv_bits"],
                   help="Quantize KV cache to N bits (e.g., 4). Omit/None to disable")
    p.add_argument("--kv-group-size", type=int, default=cfg["kv_group_size"], help="Group size for KV quantization")
    p.add_argument("--quantized-kv-start", type=int, default=cfg["quantized_kv_start"],
                   help="Start step for quantized KV (0 = from start)")
    p.add_argument("--clear-cache-after", action="store_true", default=cfg["clear_cache_after"],
                   help="Call MLX clear_cache() after each reply")

    return p.parse_args()


def main() -> int:
    cfg = merged_config()
    args = parse_args(cfg)

    if args.init_config:
        path = init_default_config(force=args.force)
        print(f"Config written: {path}")
        return 0

    aliases: Dict[str, str] = cfg["aliases"]
    model_id = resolve_model(str(args.model), aliases)

    params = Params(
        temp=float(args.temp),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        system=str(args.system),
        model_id=model_id,
        max_kv_size=int(args.max_kv_size),
        kv_bits=int(args.kv_bits) if args.kv_bits is not None else None,
        kv_group_size=int(args.kv_group_size),
        quantized_kv_start=int(args.quantized_kv_start),
        clear_cache_after=bool(args.clear_cache_after),
    )
    eng = Engine(params)

    print("\n>>> mlxchat ready. Type '/help' for commands. 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break

            # Slash-commands
            if user.startswith("/"):
                parts = user.split(maxsplit=1)
                cmd = parts[0][1:].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "help":
                    print(HELP_TEXT)
                elif cmd == "reset":
                    eng.reset()
                    print("(context cleared)")
                elif cmd == "system":
                    eng.set_system(arg)
                    print(f"(system set to: {repr(arg) if arg else 'CLEARED'})")
                elif cmd == "temp":
                    try:
                        v = float(arg); eng.params.temp = v; print(f"(temp = {v})")
                    except Exception:
                        print("Usage: /temp <float>")
                elif cmd in {"top_p", "topp", "top-p"}:
                    try:
                        v = float(arg); eng.params.top_p = v; print(f"(top_p = {v})")
                    except Exception:
                        print("Usage: /top_p <float>")
                elif cmd == "max":
                    try:
                        v = int(arg); eng.params.max_tokens = v; print(f"(max_tokens = {v})")
                    except Exception:
                        print("Usage: /max <int>")
                elif cmd == "model":
                    if not arg:
                        print("Usage: /model <alias|full-id>")
                    else:
                        new_id = resolve_model(arg, aliases)
                        if new_id == eng.params.model_id:
                            print(f"(model unchanged: {new_id})")
                        else:
                            eng.change_model(new_id)
                            print(f"(model = {new_id})")
                elif cmd == "mem":
                    try:
                        active = mx.get_active_memory()      # bytes
                        cache  = getattr(mx, "get_cache_memory", lambda: 0)()
                        peak   = getattr(mx, "get_peak_memory",  lambda: 0)()
                        print(f"(mem) active={active/1e6:.1f}MB cache={cache/1e6:.1f}MB peak={peak/1e6:.1f}MB")
                    except Exception:
                        print("(mem) memory APIs not available in this MLX version")
                elif cmd == "clear":
                    eng._clear_cache(); print("(cache cleared)")
                elif cmd == "config":
                    print_config_diagnostic(eng, aliases)
                elif cmd == "aliases":
                    print_aliases(aliases)
                elif cmd in {"system?", "show_system", "show-system", "sys"}:
                    print_system_prompt(eng)
                elif cmd in {"kv", "kv_bits"}:
                    if arg.strip().lower() == "none":
                        eng.params.kv_bits = None
                        print("(KV quantization disabled)")
                    else:
                        parts2 = arg.split()
                        if len(parts2) >= 2:
                            try:
                                bits = int(parts2[0])
                                group = int(parts2[1])
                                start = int(parts2[2]) if len(parts2) > 2 else eng.params.quantized_kv_start
                                eng.params.kv_bits = bits
                                eng.params.kv_group_size = group
                                eng.params.quantized_kv_start = start
                                print(f"(kv_bits={bits}, group={group}, start={start})")
                            except Exception:
                                print("Usage: /kv <bits> <group> [start]   or: /kv none")
                        else:
                            print("Usage: /kv <bits> <group> [start]   or: /kv none")
                elif cmd in {"max_kv", "maxkv"}:
                    try:
                        n = int(arg); eng.params.max_kv_size = n; eng._make_cache(); print(f"(max_kv_size={n})")
                    except Exception:
                        print("Usage: /max_kv <int>")
                else:
                    print("Unknown command. Try /help.")
                continue

            # Normal prompt -> stream reply
            print("Llama: ", end="", flush=True)
            for tok in eng.stream_reply(user):
                print(tok, end="", flush=True)
            print()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
