#!/usr/bin/env python3
"""
mlxchat: tiny streaming CLI chat for local MLX-LM models.

Features:
- Streaming tokens (typing effect) via mlx_lm.stream_generate when present
- Configurable temperature, top_p, max_tokens, system message
- Model aliases via ~/.config/mlxchat/config.toml (and optional ./mlxchat.toml per project)
- REPL slash-commands: /help, /reset, /system, /temp, /top_p, /max, /model

Run examples:
  python mlxchat.py
  python mlxchat.py --model llama3-8b --temp 0.6 --system "You are terse."

Initialize a default config:
  python mlxchat.py --init-config

Config precedence:
  CLI args > ENV vars > ./mlxchat.toml > ~/.config/mlxchat/config.toml > built-ins
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import argparse
import os
import sys
import time

# --- TOML loader (stdlib on 3.11+) ---
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # We won't write TOML; only read if available.

# --- MLX-LM imports with graceful fallbacks ---
from mlx_lm import load, generate  # type: ignore
try:
    from mlx_lm import stream_generate  # type: ignore[attr-defined]
except Exception:
    stream_generate = None

try:
    from mlx_lm.cache_utils import make_prompt_cache  # type: ignore
except Exception:
    make_prompt_cache = None

from mlx_lm.sample_utils import make_sampler  # type: ignore


# -------------------- Config --------------------

DEFAULT_ALIASES: Dict[str, str] = {
    # Feel free to edit in ~/.config/mlxchat/config.toml
    "llama3-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit",
    "llama3-70b": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mistral-7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

DEFAULTS = {
    "model": "llama3-8b",
    "temp": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "system": "",
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
    cfg = {}
    # global user config
    cfg_user = read_toml(USER_CFG)
    # project override
    cfg_proj = read_toml(PROJECT_CFG)

    # env overrides (simple names)
    env = {
        "model": os.getenv("MLXCHAT_MODEL"),
        "temp": os.getenv("MLXCHAT_TEMP"),
        "top_p": os.getenv("MLXCHAT_TOP_P"),
        "max_tokens": os.getenv("MLXCHAT_MAX_TOKENS"),
        "system": os.getenv("MLXCHAT_SYSTEM"),
    }
    # normalize env numeric values
    for k in ("temp", "top_p"):
        if env[k] is not None:
            try:
                env[k] = float(env[k])  # type: ignore
            except ValueError:
                env[k] = None
    if env["max_tokens"] is not None:
        try:
            env["max_tokens"] = int(env["max_tokens"])  # type: ignore
        except ValueError:
            env["max_tokens"] = None

    # merge on precedence: defaults -> user -> project -> env -> CLI later
    def deep_get(d: Dict, key: str, default=None):
        return d.get(key, default) if isinstance(d, dict) else default

    aliases = DEFAULT_ALIASES.copy()
    aliases.update(deep_get(cfg_user, "aliases", {}))
    aliases.update(deep_get(cfg_proj, "aliases", {}))

    settings = DEFAULTS.copy()
    for source in (cfg_user, cfg_proj, env):
        for k in ("model", "temp", "top_p", "max_tokens", "system"):
            v = deep_get(source, k, None)
            if v is not None:
                settings[k] = v

    return {"aliases": aliases, **settings}


def init_default_config(force: bool = False) -> Path:
    target = USER_CFG
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target

    # write a friendly TOML by hand (no external writers)
    text = """# mlxchat config (user)
# You can override any of these in a project-local ./mlxchat.toml

# Default settings
model = "llama3-8b"
temp = 0.7
top_p = 0.9
max_tokens = 512
system = ""

# Model aliases (edit to your taste)
[aliases]
llama3-8b = "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit"
llama3-70b = "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"
qwen2.5-7b = "mlx-community/Qwen2.5-7B-Instruct-4bit"
mistral-7b = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
"""
    target.write_text(text)
    return target


def resolve_model(name_or_id: str, aliases: Dict[str, str]) -> str:
    return aliases.get(name_or_id, name_or_id)


# -------------------- Chat engine --------------------

@dataclass
class Params:
    temp: float
    top_p: float
    max_tokens: int
    system: str
    model_id: str


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
        self.prompt_cache = make_prompt_cache(self.model) if make_prompt_cache else None

    def set_system(self, text: str) -> None:
        self.params.system = text
        # update/insert system in messages
        if self.messages and self.messages[0].get("role") == "system":
            if text:
                self.messages[0]["content"] = text
            else:
                # remove system if clearing
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
        # Recreate prompt cache tied to the new model
        self.reset()

    def _sampler(self):
        return make_sampler(temp=self.params.temp, top_p=self.params.top_p)

    def _gen_kwargs(self, prompt: str) -> Dict:
        kw = dict(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.params.max_tokens,
            sampler=self._sampler(),
        )
        if self.prompt_cache is not None:
            kw["prompt_cache"] = self.prompt_cache
        return kw

    def stream_reply(self, user_text: str) -> Iterator[str]:
        if not user_text.strip():
            yield "Please enter something non-empty."
            return

        self.messages.append({"role": "user", "content": user_text})
        prompt = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True)
        kw = self._gen_kwargs(prompt)

        if stream_generate is not None:
            chunks: List[str] = []
            for resp in stream_generate(**kw):
                token = resp.text
                chunks.append(token)
                yield token
            full = "".join(chunks)
        else:
            full = generate(**kw)
            yield full

        self.messages.append({"role": "assistant", "content": full})


# -------------------- CLI / REPL --------------------

HELP_TEXT = """\
Slash commands:
  /help                 Show this help
  /reset                Clear conversation context (keeps system message)
  /system <text>        Set/replace system prompt (use /system to clear)
  /temp <float>         Set temperature (e.g., 0.1–1.5). Current: {temp}
  /top_p <float>        Set nucleus sampling top_p (0–1). Current: {top_p}
  /max <int>            Set max new tokens. Current: {max_tokens}
  /model <alias|id>     Switch model by alias or full ID. Current: {model}

Notes:
- Model aliases come from ~/.config/mlxchat/config.toml and ./mlxchat.toml
- Press Ctrl-C to exit, or type 'exit'/'quit'
"""


def parse_args(cfg: Dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming local chat for MLX-LM models")
    p.add_argument("--init-config", action="store_true", help="Create ~/.config/mlxchat/config.toml with sensible defaults")
    p.add_argument("--force", action="store_true", help="Overwrite config when used with --init-config")
    p.add_argument("--model", default=cfg["model"], help="Model alias or full identifier (see config.toml)")
    p.add_argument("--temp", type=float, default=cfg["temp"], help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=cfg["top_p"], help="Top-p (nucleus) sampling")
    p.add_argument("--max-tokens", type=int, default=cfg["max_tokens"], help="Maximum new tokens to generate")
    p.add_argument("--system", default=cfg["system"], help="System prompt")
    return p.parse_args()


def main() -> int:
    cfg = merged_config()
    args = parse_args(cfg)

    if args.init_config:
        path = init_default_config(force=args.force)
        print(f"Config written: {path}")
        return 0

    aliases: Dict[str, str] = cfg["aliases"]
    model_id = resolve_model(args.model, aliases)

    params = Params(
        temp=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        system=args.system,
        model_id=model_id,
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
                    print(HELP_TEXT.format(
                        temp=eng.params.temp,
                        top_p=eng.params.top_p,
                        max_tokens=eng.params.max_tokens,
                        model=eng.params.model_id,
                    ))
                elif cmd == "reset":
                    eng.reset()
                    print("(context cleared)")
                elif cmd == "system":
                    eng.set_system(arg)
                    print(f"(system set to: {repr(arg) if arg else 'CLEARED'})")
                elif cmd == "temp":
                    try:
                        v = float(arg)
                        eng.params.temp = v
                        print(f"(temp = {v})")
                    except Exception:
                        print("Usage: /temp <float>")
                elif cmd in {"top_p", "topp", "top-p"}:
                    try:
                        v = float(arg)
                        eng.params.top_p = v
                        print(f"(top_p = {v})")
                    except Exception:
                        print("Usage: /top_p <float>")
                elif cmd == "max":
                    try:
                        v = int(arg)
                        eng.params.max_tokens = v
                        print(f"(max_tokens = {v})")
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
                else:
                    print("Unknown command. Try /help.")
                continue

            # Normal prompt -> stream reply
            print("Llama: ", end="", flush=True)
            for tok in eng.stream_reply(user):
                print(tok, end="", flush=True)
            print()  # newline
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
