from mlx_lm import load, generate
try:
    from mlx_lm.cache_utils import make_prompt_cache  # new API
except Exception:
    make_prompt_cache = None

from mlx_lm.sample_utils import make_sampler

print("Loading model … this may take a few seconds")
model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-3bit")

prompt_cache = make_prompt_cache(model) if make_prompt_cache else None
messages: list[dict[str, str]] = []

def chat_turn(user_input: str, *, temp: float = 0.7, top_p: float = 0.9, max_tokens: int = 512) -> str:
    if not user_input.strip():
        return "Please enter something non-empty."

    messages.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=temp, top_p=top_p)

    kwargs = dict(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
    if prompt_cache is not None:
        kwargs["prompt_cache"] = prompt_cache  # works on newer mlx-lm only

    response = generate(**kwargs)
    messages.append({"role": "assistant", "content": response})
    return response

if __name__ == "__main__":
    print(">>> Local Llama-3 chat.  Type 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            user = input("You: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            print(f"Llama-3: {chat_turn(user)}\n")
    except KeyboardInterrupt:
        pass
