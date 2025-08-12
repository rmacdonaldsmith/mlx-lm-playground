from __future__ import annotations
from typing import Callable, Dict, Tuple

from .engine import Engine
from .memory import snapshot_bytes

Command = Callable[[str, Engine, Dict[str, str]], None]
REGISTRY: Dict[str, Command] = {}

def command(*names: str):
    def deco(fn: Command):
        for n in names:
            REGISTRY[n] = fn
        return fn
    return deco

@command("help")
def cmd_help(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    print("""\
Slash commands:
  /help                 Show this help
  /reset                Clear conversation context (keeps system message)
  /system <text>        Set/replace system prompt (use /system to clear)
  /system?              Show current system prompt
  /temp <float>         Set temperature (e.g., 0.1–1.5)
  /top_p <float>        Set nucleus sampling top_p (0–1)
  /max <int>            Set max new tokens
  /model <alias|id>     Switch model by alias or full ID
  /config               Show current runtime config
  /aliases              List configured model aliases
  /mem                  Show active/cache/peak memory (if supported by MLX)
  /clear                Clear MLX caches and reset peak memory
  /kv <bits> <group> [start]   Quantize KV cache (e.g., /kv 4 64 0). '/kv none' disables.
  /max_kv <int>         Set rotating KV cache size (0 disables)
""")

@command("reset")
def cmd_reset(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    eng.reset()
    print("(context cleared)")

@command("system")
def cmd_system(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    eng.set_system(arg)
    print(f"(system set to: {repr(arg) if arg else 'CLEARED'})")

@command("system?", "show_system", "sys", "show-system")
def cmd_system_show(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
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

@command("temp")
def cmd_temp(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    try:
        v = float(arg); eng.params.temp = v; print(f"(temp = {v})")
    except Exception:
        print("Usage: /temp <float>")

@command("top_p", "topp", "top-p")
def cmd_top_p(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    try:
        v = float(arg); eng.params.top_p = v; print(f"(top_p = {v})")
    except Exception:
        print("Usage: /top_p <float>")

@command("max")
def cmd_max(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    try:
        v = int(arg); eng.params.max_tokens = v; print(f"(max_tokens = {v})")
    except Exception:
        print("Usage: /max <int>")

@command("model")
def cmd_model(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    if not arg:
        print("Usage: /model <alias|full-id>")
        return
    new_id = aliases.get(arg, arg)
    if new_id == eng.params.model_id:
        print(f"(model unchanged: {new_id})"); return
    eng.change_model(new_id)
    print(f"(model = {new_id})")

@command("config")
def cmd_config(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    snap = eng.snapshot_config()
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
        f"aliases(configured) : {len(aliases)}",
    ]
    print("\n".join(lines))

@command("aliases")
def cmd_aliases(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    if not aliases:
        print("(no aliases configured)"); return
    width = max(len(k) for k in aliases.keys())
    print("— Model aliases —")
    for k, v in sorted(aliases.items()):
        print(f"{k.ljust(width)} -> {v}")

@command("mem")
def cmd_mem(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    info = snapshot_bytes()
    fmt = lambda b: f"{b/1e6:.1f}MB"
    print(f"(mem) active={fmt(info['active'])} cache={fmt(info['cache'])} peak={fmt(info['peak'])}")

@command("clear")
def cmd_clear(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    eng.clear_cache()
    print("(cache cleared)")

@command("kv", "kv_bits")
def cmd_kv(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    s = arg.strip().lower()
    if s == "none":
        eng.params.kv_bits = None
        print("(KV quantization disabled)")
        return
    parts = arg.split()
    if len(parts) < 2:
        print("Usage: /kv <bits> <group> [start]   or: /kv none")
        return
    try:
        bits = int(parts[0])
        group = int(parts[1])
        start = int(parts[2]) if len(parts) > 2 else eng.params.quantized_kv_start
        eng.params.kv_bits = bits
        eng.params.kv_group_size = group
        eng.params.quantized_kv_start = start
        print(f"(kv_bits={bits}, group={group}, start={start})")
    except Exception:
        print("Usage: /kv <bits> <group> [start]   or: /kv none")

@command("max_kv", "maxkv")
def cmd_max_kv(arg: str, eng: Engine, aliases: Dict[str, str]) -> None:
    try:
        n = int(arg)
        eng.set_max_kv_size(n)
        print(f"(max_kv_size = {n})")
    except Exception:
        print("Usage: /max_kv <int>")
