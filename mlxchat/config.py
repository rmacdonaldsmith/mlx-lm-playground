from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# stdlib TOML in 3.11+
try:
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    tomllib = None  # we'll just skip reading if unavailable

USER_CFG = Path.home() / ".config" / "mlxchat" / "config.toml"
PROJECT_CFG = Path.cwd() / "mlxchat.toml"

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
    "max_kv_size": 4096,
    "kv_bits": None,
    "kv_group_size": 64,
    "quantized_kv_start": 0,
    "clear_cache_after": False,
}

def _read_toml(path: Path) -> Dict:
    if not path.exists():
        return {}
    if tomllib is None:
        print(f"Warning: tomllib not available; ignoring {path}", file=sys.stderr)
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)

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
kv_bits = 4
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

def _coerce(name: str, v):
    if v is None:
        return None
    try:
        if name in {"temp", "top_p"}:
            return float(v)
        if name in {"max_tokens", "max_kv_size", "kv_group_size", "quantized_kv_start"}:
            return int(v)
        if name == "kv_bits":
            s = str(v).strip().lower()
            return None if s in {"", "none"} else int(v)
        if name == "clear_cache_after":
            if isinstance(v, bool): return v
            return str(v).lower() in {"1","true","t","yes","y"}
    except Exception:
        return None
    return v

def merged_config() -> Dict:
    cfg_user = _read_toml(USER_CFG)
    cfg_proj = _read_toml(PROJECT_CFG)

    # environment overrides
    env = {
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
    env = {k: _coerce(k, v) for k, v in env.items()}

    # merge: defaults -> user -> project -> env
    settings = DEFAULTS.copy()
    def overlay(d: Dict):
        if not isinstance(d, dict): return
        for k in settings.keys():
            if k in d and d[k] is not None:
                settings[k] = d[k]
    overlay(cfg_user)
    overlay(cfg_proj)
    overlay(env)

    aliases = DEFAULT_ALIASES.copy()
    aliases.update(cfg_user.get("aliases", {}))
    aliases.update(cfg_proj.get("aliases", {}))

    return {"aliases": aliases, **settings}

def resolve_model(name_or_id: str, aliases: Dict[str, str]) -> str:
    return aliases.get(name_or_id, name_or_id)
