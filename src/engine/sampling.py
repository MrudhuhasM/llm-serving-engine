import torch

def sample_next_token(logits: torch.Tensor) -> int:
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor")
    return torch.argmax(logits).item() # Return a Python int, not a tensor.

if __name__ == "__main__":
    logits = torch.tensor([-0.5,1.8,5.7,3.2,0.1,-0.8,-2.3,-4.5,-6.7,-8.9])
    print(sample_next_token(logits))