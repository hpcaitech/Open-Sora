import copy
import torch
import torch_musa
from torch.testing import assert_close
from opensora.models.layers.blocks import LabelEmbedder

def test_label_emb(device):
    label = torch.randint(low=0, high=9, size=(1024,), device=device)
    label_embedder = LabelEmbedder(
        num_classes=10,
        hidden_size=256,
        dropout_prob=0.5
    ).to(device=device)
    output = label_embedder(labels=label, train=None)
    output.sum().backward()
    print(f"Shape {output.shape}\n {output}\n")

# TODO: distributed test; may not;
# TODO: correctness test 
def test_label_emb_correctness():
    dtype = torch.bfloat16 # torch.float, torch.float16, torch.bfloat16
    device="musa"
    torch.manual_seed(1024)
    
    label_cpu = torch.randint(low=0, high=9, size=(1024,))
    label_muas = copy.deepcopy(label_cpu).to(device=device)
    
    label_embedder_cpu = LabelEmbedder(
        num_classes=10,
        hidden_size=256,
        dropout_prob=0.5
    )
    label_embedder_musa = copy.deepcopy(label_embedder_cpu).to(device=device)
    
    output_cpu = label_embedder_cpu(labels=label_cpu, train=None)
    output_musa = label_embedder_musa(labels=label_muas, train=None)
    
    assert_close(output_cpu, output_musa, check_device=False)

if __name__ == "__main__":
    print("Test label Embedding")
    test_label_emb("musa")
    
    print("Test Label Embedding Correctness")
    test_label_emb_correctness()