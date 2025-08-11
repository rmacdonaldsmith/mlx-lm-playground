"""
Streaming chat for a local MLX-LM model (typing effect).
Run:  python simple-chat.py
"""

from typing import Iterator

from mlx_lm import load, generate
try:
    # Streaming is available in recent mlx-lm
    from mlx_lm import stream_generate  # type: ignore[attr-defined]
except Exception:
    stream_generate = None  # fallback later

try:
    from mlx_lm.cache_utils import make_prompt_cache  # optional API
except Exception:
    make_prompt_cache = None

from mlx_lm.sample_utils import make_sampler


# -------------------- Model & tokenizer --------------------
MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-3bit"

print("Loading model … this may take a few seconds")
model, tokenizer = load(MODEL_ID)

# -------------------- Prompt-cache for faster multi-turn --------------------
prompt_cache = make_prompt_cache(model) if make_prompt_cache else None

# -------------------- Conversation state --------------------
messages: list[dict[str, str]] = []


def chat_turn_stream(
    user_input: str,
    *,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> Iterator[str]:
    """
    Yield assistant tokens as they stream in (typing effect).
    Also appends the full assistant reply to `messages` when done.
    """
    if not user_input.strip():
        yield "Please enter something non-empty."
        return

    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=temp, top_p=top_p)

    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    if prompt_cache is not None:
        kwargs["prompt_cache"] = prompt_cache  # supported on recent mlx-lm

    # Stream if supported; otherwise generate once.
    if stream_generate is not None:
        chunks: list[str] = []
        for resp in stream_generate(**kwargs):
            # resp.text is the newly generated chunk
            token = resp.text
            chunks.append(token)
            yield token
        full = "".join(chunks)
    else:
        full = generate(**kwargs)
        yield full

    messages.append({"role": "assistant", "content": full})


# -------------------- Simple REPL --------------------
if __name__ == "__main__":
    print(">>> Local Llama-3 chat.  Type 'exit' or Ctrl-C to quit. Type '/reset' to clear context.\n")
    try:
        while True:
            user = input("You: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            if user == "/reset":
                messages.clear()
                print("(context cleared)\n")
                continue

            print("Llama-3: ", end="", flush=True)
            for tok in chat_turn_stream(user):
                print(tok, end="", flush=True)
            print("\n")
    except KeyboardInterrupt:
        pass
