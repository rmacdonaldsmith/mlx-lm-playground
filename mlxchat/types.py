from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


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
