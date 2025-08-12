from __future__ import annotations
import argparse

from .config import merged_config, init_default_config, resolve_model
from .engine import Engine
from .types import Params
from .commands import REGISTRY as CMD

HELP_BANNER = "\n>>> mlxchat ready. Type '/help' for commands. 'exit' or Ctrl-C to quit.\n"

def parse_args(cfg: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming local chat for MLX-LM models")
    p.add_argument("--init-config", action="store_true", help="Create ~/.config/mlxchat/config.toml with sensible defaults")
    p.add_argument("--force", action="store_true", help="Overwrite config when used with --init-config")
    p.add_argument("--model", default=cfg["model"], help="Model alias or full identifier")
    p.add_argument("--temp", type=float, default=cfg["temp"], help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=cfg["top_p"], help="Top-p (nucleus) sampling")
    p.add_argument("--max-tokens", type=int, default=cfg["max_tokens"], help="Maximum new tokens to generate")
    p.add_argument("--system", default=cfg["system"], help="System prompt")
    # Memory / KV
    p.add_argument("--max-kv-size", type=int, default=cfg["max_kv_size"], help="Maximum KV cache size (rotating cache). 0 disables rotation")
    p.add_argument("--kv-bits", type=int, nargs="?", default=cfg["kv_bits"], help="Quantize KV cache to N bits (e.g., 4). Omit/None to disable")
    p.add_argument("--kv-group-size", type=int, default=cfg["kv_group_size"], help="Group size for KV quantization")
    p.add_argument("--quantized-kv-start", type=int, default=cfg["quantized_kv_start"], help="Start step for quantized KV (0 = from start)")
    p.add_argument("--clear-cache-after", action="store_true", default=cfg["clear_cache_after"], help="Call MLX clear_cache() after each reply")
    return p.parse_args()

def main() -> int:
    cfg = merged_config()
    args = parse_args(cfg)

    if args.init_config:
        path = init_default_config(force=args.force)
        print(f"Config written: {path}")
        return 0

    aliases = cfg["aliases"]
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

    print(HELP_BANNER)
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
                fn = CMD.get(cmd)
                if fn is None:
                    print("Unknown command. Try /help.")
                else:
                    fn(arg, eng, aliases)
                continue

            # Normal prompt -> stream reply
            print("Llama: ", end="", flush=True)
            for tok in eng.stream_reply(user):
                print(tok, end="", flush=True)
            print()
    except KeyboardInterrupt:
        pass
    return 0
