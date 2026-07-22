import torch

from engine.engine import Engine
from engine.request import Request

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 20

PROMPTS = [
    "Once upon a time, there",
    "The capital of France is",
    "def fibonacci(n):",
]


def compare_prompt(engine: Engine, prompt: str) -> None:
    print(f"\n--- prompt: {prompt!r} ---")

    request = Request(prompt=prompt, max_new_tokens=MAX_NEW_TOKENS)
    our_ids = list(engine.generate(request))
    print("ours:", our_ids)

    input_ids = engine.tokenizer.encode(
        prompt,
        return_tensors="pt",
    ).to(engine.device)

    with torch.inference_mode():
        reference = engine.model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    reference_ids = reference[0, input_ids.shape[1] :].tolist()
    print("reference:", reference_ids)
    print("match:", our_ids == reference_ids)
    print(engine.tokenizer.decode(our_ids))
    print(engine.tokenizer.decode(reference_ids))


def main():
    engine = Engine(MODEL_NAME)
    for prompt in PROMPTS:
        compare_prompt(engine, prompt)


if __name__ == "__main__":
    main()
