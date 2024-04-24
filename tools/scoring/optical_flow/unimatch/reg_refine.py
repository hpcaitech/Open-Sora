import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=256,
        out_dim=2,
    ):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))

        return out


class SepConvGRU(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        input_dim=192 + 128,
        kernel_size=5,
    ):
        padding = (kernel_size - 1) // 2

        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(
        self,
        corr_channels=324,
        flow_channels=2,
    ):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(corr_channels, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(flow_channels, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - flow_channels, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        corr_channels=324,
        hidden_dim=128,
        context_dim=128,
        downsample_factor=8,
        flow_dim=2,
        bilinear_up=False,
    ):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(
            corr_channels=corr_channels,
            flow_channels=flow_dim,
        )

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + hidden_dim)

        self.flow_head = FlowHead(
            hidden_dim,
            hidden_dim=256,
            out_dim=flow_dim,
        )

        if bilinear_up:
            self.mask = None
        else:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, downsample_factor**2 * 9, 1, padding=0),
            )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)

        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        if self.mask is not None:
            mask = self.mask(net)
        else:
            mask = None

        return net, mask, delta_flow
