import torch

from opensora.registry import MODELS


@MODELS.register_module("classes")
class ClassEncoder:
    def __init__(self, num_classes, model_max_length=None, device="cuda", dtype=torch.float):
        self.num_classes = num_classes
        self.y_embedder = None

        self.model_max_length = model_max_length
        self.output_dim = None
        self.device = device
        self.tokenize_fn = None

    def encode(self, input_ids, attention_mask=None):
        return dict(y=input_ids.to(self.device))

    def null(self, n):
        return torch.tensor([self.num_classes] * n).to(self.device)
