import torch
import torch_musa
import torch.nn as nn
from opensora.utils.train_utils import set_seed

SEED=1024

def test_dropout():
    set_seed(SEED)
    dropout = nn.Dropout(0.5)
    a = torch.rand(10, dtype=torch.bfloat16, device='musa')
    res = dropout(a)
    print(f"a {a}\n")
    print(f"res {res}")
    

if __name__ == "__main__":
    test_dropout()
