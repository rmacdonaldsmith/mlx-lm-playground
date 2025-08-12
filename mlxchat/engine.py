from __future__ import annotations
from typing import Dict, Iterator, List, Optional

from .types import Params
from .memory import clear_cache as _clear_cache_impl

# MLX / MLX-LM imports
from mlx_lm import load, generate  # type: ignore
try:
    from mlx_lm import stream_generate  # type: ignore[attr-defined]
except Exception:
    stream_generate = None

# cache factory (newer and older paths)
try:
    from mlx_lm.models.cache import make_prompt_cache  # type: ignore
except Exception:
    try:
        from mlx_lm.cache_utils import make_prompt_cache  # type: ignore
    except Exception:
        make_prompt_cache = None  # type: ignore

from mlx_lm.sample_utils import make_sampler  # type: ignore


class Engine:
    """Owns model, tokenizer, chat history, and streaming generation."""

    def __init__(self, params: Params):
        self.params = params
        self.messages: List[Dict[str, str]] = []
        self._load_model()
        if self.params.system:
            self.messages.append({"role": "system", "content": self.params.system})

    # ---------- lifecycle ----------
    def _load_model(self) -> None:
        print(f"Loading model: {self.params.model_id} …", flush=True)
        self.model, self.tokenizer = load(self.params.model_id)
        self._make_cache()

    def _make_cache(self) -> None:
        self.prompt_cache = None
        if make_prompt_cache is None:
            return
        try:
            if self.params.max_kv_size and self.params.max_kv_size > 0:
                self.prompt_cache = make_prompt_cache(self.model, max_kv_size=int(self.params.max_kv_size))
            else:
                self.prompt_cache = make_prompt_cache(self.model)
        except TypeError:
            self.prompt_cache = make_prompt_cache(self.model)

    # ---------- public controls ----------
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

    def clear_cache(self) -> None:
        _clear_cache_impl()

    def set_max_kv_size(self, n: int) -> None:
        self.params.max_kv_size = n
        self._make_cache()

    # ---------- generation ----------
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
        if self.params.max_kv_size and self.params.max_kv_size > 0:
            kw["max_kv_size"] = int(self.params.max_kv_size)
        if self.params.kv_bits is not None:
            kw["kv_bits"] = int(self.params.kv_bits)
            kw["kv_group_size"] = int(self.params.kv_group_size)
            kw["quantized_kv_start"] = int(self.params.quantized_kv_start)
        return kw

    @staticmethod
    def _strip_unknown_kv_args(kw: Dict) -> Dict:
        bad = {"kv_bits", "kv_group_size", "quantized_kv_start", "max_kv_size"}
        return {k: v for k, v in kw.items() if k not in bad}

    def _try_stream_tokens(self, kw: Dict) -> Iterator[str]:
        """Yield tokens with stream_generate; fallback to generate; strip unknown kv args on older mlx-lm."""
        try:
            if stream_generate is not None:
                for resp in stream_generate(**kw):  # type: ignore
                    yield resp.text
            else:
                yield generate(**kw)  # type: ignore
        except TypeError as e:
            msg = str(e)
            if any(k in msg for k in ("kv_bits", "kv_group_size", "quantized_kv_start", "max_kv_size")):
                kw2 = self._strip_unknown_kv_args(kw)
                if stream_generate is not None:
                    for resp in stream_generate(**kw2):  # type: ignore
                        yield resp.text
                else:
                    yield generate(**kw2)  # type: ignore
            else:
                raise

    def stream_reply(self, user_text: str) -> Iterator[str]:
        """Append user msg, stream assistant tokens, append full reply, and optionally clear caches."""
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
            self.clear_cache()

    # ---------- diagnostics ----------
    def snapshot_config(self) -> Dict[str, object]:
        sysmsg = self.params.system or ""
        preview = (sysmsg[:120] + "…") if len(sysmsg) > 120 else sysmsg
        turns = sum(1 for m in self.messages if m["role"] in {"user", "assistant"}) // 2
        return {
            "model_id": self.params.model_id,
            "temp": self.params.temp,
            "top_p": self.params.top_p,
            "max_tokens": self.params.max_tokens,
            "system_len": len(sysmsg),
            "system_preview": preview,
            "max_kv_size": self.params.max_kv_size,
            "kv_bits": self.params.kv_bits,
            "kv_group_size": self.params.kv_group_size,
            "quantized_kv_start": self.params.quantized_kv_start,
            "clear_cache_after": self.params.clear_cache_after,
            "prompt_cache_enabled": (self.prompt_cache is not None),
            "streaming_available": (stream_generate is not None),
            "turns_in_history": turns,
        }
