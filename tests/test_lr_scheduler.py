import torch
from torch.optim import Adam
from torchvision.models import resnet50

from opensora.utils.lr_scheduler import LinearWarmupLR


def test_lr_scheduler():
    model = resnet50().cuda()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = LinearWarmupLR(optimizer, warmup_steps=10)
    current_lr = scheduler.get_lr()[0]
    data = torch.rand(128, 3, 224, 224).cuda()

    for i in range(100):
        out = model(data)
        out.mean().backward()

        optimizer.step()
        scheduler.step()

        if i >= 10:
            assert scheduler.get_lr()[0] == 0.01
        else:
            assert scheduler.get_lr()[0] > current_lr, f"{scheduler.get_lr()[0]} <= {current_lr}"
            current_lr = scheduler.get_lr()[0]


if __name__ == "__main__":
    test_lr_scheduler()
