import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from engine.request import Request
from engine.sampling import sample_next_token

from collections.abc import Iterator


class Engine:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()


    @torch.inference_mode()
    def prefill(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(input_ids, use_cache=True)
        token_id = sample_next_token(outputs.logits[0, -1, :])
        return token_id, input_ids, outputs.past_key_values

    @torch.inference_mode()
    def decode(self, token_id: int, past_key_values):
        input_ids = torch.tensor([[token_id]], device=self.device)
        outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        token_id = sample_next_token(outputs.logits[0, -1, :])
        return token_id, outputs.past_key_values

    def generate(self, request: Request) -> Iterator[int]:

        if request.max_new_tokens <= 0:
            request.finished = True
            request.finished_reason = "length"
            return
        
        token_id, input_ids, past_key_values = self.prefill(request.prompt)
        request.input_ids = input_ids[0].tolist()
        request.generated_token_ids.append(token_id)

        yield token_id

        if token_id == self.tokenizer.eos_token_id:
            request.finished = True
            request.finished_reason = "eos"
            return

        while len(request.generated_token_ids) < request.max_new_tokens:
            token_id, past_key_values = self.decode(token_id, past_key_values)
            request.generated_token_ids.append(token_id)
            yield token_id

            if token_id == self.tokenizer.eos_token_id:
                request.finished = True
                request.finished_reason = "eos"
                return
        
        request.finished = True
        request.finished_reason = "length"
        return


if __name__ == "__main__":
    engine = Engine("Qwen/Qwen3-0.6B")
    request1 = Request(prompt="Once upon a time, there", max_new_tokens=10)
    request2 = Request(prompt="Once upon a time, there", max_new_tokens=0)
    for token in engine.generate(request1):
        text = engine.tokenizer.decode([token])
        print(text, end="", flush=True)

    print()
    print(f"Generated token ids: {request1.generated_token_ids}")
    print(f"Finished: {request1.finished}")
    print(f"Finished reason: {request1.finished_reason}")

    for token in engine.generate(request2):
        text = engine.tokenizer.decode([token])
        print(text, end="", flush=True)

    print()
    print(f"Generated token ids: {request2.generated_token_ids}")
    print(f"Finished: {request2.finished}")
    print(f"Finished reason: {request2.finished_reason}")