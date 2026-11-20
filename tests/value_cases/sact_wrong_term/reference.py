import torch
from silu_mul import term

def reference(gate, up, n):
    return term(gate, up, n).to(torch.bfloat16)
