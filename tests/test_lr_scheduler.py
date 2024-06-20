import torch
from torch.optim import Adam
from torchvision.models import resnet50
from tqdm import tqdm

from opensora.utils.lr_scheduler import LinearWarmupLR


def test_lr_scheduler():
    warmup_steps = 200
    model = resnet50().cuda()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = LinearWarmupLR(optimizer, warmup_steps=warmup_steps)
    current_lr = scheduler.get_lr()[0]
    data = torch.rand(1, 3, 224, 224).cuda()

    for i in tqdm(range(warmup_steps * 2)):
        out = model(data)
        out.mean().backward()
        optimizer.step()
        scheduler.step()

        if i >= warmup_steps:
            assert scheduler.get_lr()[0] == 0.01
        else:
            assert scheduler.get_lr()[0] > current_lr, f"{scheduler.get_lr()[0]} <= {current_lr}"
            current_lr = scheduler.get_lr()[0]


if __name__ == "__main__":
    test_lr_scheduler()
