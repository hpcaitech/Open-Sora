import os

import torch.nn as nn

from opensora.registry import MODELS
from opensora.utils.ckpt import load_checkpoint


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_conv(m):
    if hasattr(m, "conv"):
        m = m.conv
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""

    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=5,
        norm_layer=nn.BatchNorm3d,
        conv_cls="conv3d",
        dropout=0.30,
    ):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminator3D, self).__init__()
        assert conv_cls == "conv3d"
        use_bias = False

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=2 if n == 1 else (1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=(kw, kw, kw),
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.main(x)


@MODELS.register_module("N_Layer_discriminator_3D")
def N_LAYER_DISCRIMINATOR_3D(from_pretrained=None, force_huggingface=None, **kwargs):
    model = NLayerDiscriminator3D(**kwargs).apply(weights_init)
    if from_pretrained is not None:
        if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
            raise NotImplementedError
        else:
            load_checkpoint(model, from_pretrained)
        print(f"loaded model from: {from_pretrained}")
    return model
