import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from engine.sampling import sample_next_token

def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device=device)
    model.eval()
    return tokenizer, model

def decode(tokenizer: AutoTokenizer, outputs: torch.Tensor):
    next_token_logits = outputs.logits[0, -1, :]
    next_token = sample_next_token(next_token_logits)
    output = tokenizer.decode(next_token)
    print(f"Output: {output}")
    print(f"Next token {next_token}")
    return torch.tensor([next_token]).reshape(1, 1).to(device)


def print_metadata(prompt: str, input_ids: torch.Tensor, outputs: torch.Tensor, tokenizer: AutoTokenizer):
    print(f"Prompt: {prompt}")
    print(f"Input IDs: {input_ids}")
    print(f"Input IDs shape: {input_ids.shape}")

    print(f"Outputs: {outputs}")
    print(f"Outputs shape: {outputs.logits.shape}")
    print(f"Cache keys: {outputs.past_key_values}")
    print(f"Cache keys sequence length: {outputs.past_key_values.get_seq_length()}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "Hello, how are you?"
    with torch.inference_mode():
        tokenizer, model = load_model_and_tokenizer("Qwen/Qwen3-0.6B")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model(input_ids, use_cache=True)
        print_metadata(prompt, input_ids, outputs, tokenizer)
        next_token = decode(tokenizer, outputs)
        print(f"Next token shape: {next_token.shape}")

        outputs = model(next_token, use_cache=True, past_key_values=outputs.past_key_values)
        print_metadata(prompt, next_token, outputs, tokenizer)
        decode(tokenizer, outputs)
    
