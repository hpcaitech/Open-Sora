# modified from
# https://github.com/bornfly-detachment/asymmetric_magvitv2/blob/main/models/vae/vae.py


import os
from typing import Tuple, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel

from opensora.models.vae.utils import DiagonalGaussianDistribution
from opensora.registry import MODELS, build_module
from opensora.utils.ckpt_utils import load_checkpoint

from .constants import get_vae_stats
from .utils import to_torch_dtype


class OpenSoraVAE_V1_3_PiplineConfig(PretrainedConfig):
    model_type = "OpenSoraVAE_V1_3_Pipline"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class OpenSoraVAE_V1_3_Pipline(PreTrainedModel):
    config_class = OpenSoraVAE_V1_3_PiplineConfig

    def __init__(
        self,
        config: OpenSoraVAE_V1_3_PiplineConfig,
        micro_batch_size=None,
        micro_batch_size_2d=None,
        micro_frame_size=None,
        use_tiled_conv3d=False,
        tile_size=16,
        tiled_dim=None,
        num_tiles=None,
        temporal_overlap=False,
        normalization=None,
    ):
        super().__init__(config=config)
        print("OpenSoraVAE_V1_3_Pipline config", config)
        self.micro_batch_size = micro_batch_size
        self.micro_batch_size_2d = micro_batch_size_2d
        self.micro_frame_size = micro_frame_size
        self.encoder = build_module(config.encoder, MODELS)
        self.decoder = build_module(config.decoder, MODELS)

        self.out_channels = config.encoder["z_channels"]  # for dit training
        if use_tiled_conv3d:
            self._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

        self.temporal_overlap = temporal_overlap
        self.normalization = normalization

    def _enable_tiled_conv3d(self, tile_size=16, tiled_dim=None, num_tiles=None):
        self.encoder._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        self.decoder._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)

    # for dit training
    def get_latent_size(self, input_size, patch_size=[4, 8, 8]):
        latent_size = []
        # temporal
        if input_size[0] is None:
            latent_size.append(None)
        elif input_size[0] == 1:
            latent_size.append(1)
        elif self.micro_frame_size is None:
            latent_size.append((input_size[0] - 1) // patch_size[0] + 1)
        elif not self.temporal_overlap:
            micro_z_frame_size = int((self.micro_frame_size - 1) / 4 + 1)
            latent_size.append(input_size[0] // self.micro_frame_size * micro_z_frame_size)
        else:
            micro_z_frame_size = int((self.micro_frame_size - 1) / 4 + 1)
            latent_size.append((input_size[0] - 1) // (self.micro_frame_size - 1) * micro_z_frame_size)
        # spatial
        for i in range(1, 3):
            latent_size.append((input_size[i] - 1) // patch_size[i] + 1 if input_size[i] is not None else None)
        return latent_size

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    @property
    def micro_z_frame_size(self):
        return int((self.micro_frame_size - 1) / 4 + 1) if self.micro_frame_size else None

    def encode_micro_frame(self, x):
        z_list = []
        # x.shape[2] = n * self.micro_frame_size
        if not self.temporal_overlap:
            for i in range(0, x.shape[2], self.micro_frame_size):
                x_bs = x[:, :, i : i + self.micro_frame_size]
                z_list.append(self.encoder(x_bs))
        else:
            for i in range(0, x.shape[2] - 1, self.micro_frame_size - 1):
                x_bs = x[:, :, i : i + self.micro_frame_size]
                z_list.append(self.encoder(x_bs))
        z = torch.cat(z_list, dim=2)
        return z

    def encode_micro_batch(self, x):
        z_list = []
        # x.shape[0] = n * self.micro_batch_size
        for i in range(0, x.shape[0], self.micro_batch_size):
            x_bs = x[i : i + self.micro_batch_size]
            if self.micro_frame_size is None or x_bs.size(2) == 1:
                z_list.append(self.encoder(x_bs))
            else:
                z_list.append(self.encode_micro_frame(x_bs))
        z = torch.cat(z_list, dim=0)
        return z

    def normalize_z(self, z):
        if self.normalization is None:
            return z
        elif self.normalization == "video":
            mean, std = get_vae_stats("video", z)
            mean = mean[None, :, None, None, None]
            std = std[None, :, None, None, None]
            z = ((z - mean) / std).to(z.dtype)
        elif self.normalization == "video_5frames":
            mean, std = get_vae_stats("video_5frames", z)
            if z.size(2) == 1:
                mean = mean[None, :, :1, None, None]
                std = std[None, :, :1, None, None]
            else:
                num_replica = z.size(2) // mean.size(1)
                mean = mean.repeat(1, num_replica)[None, :, :, None, None]
                std = std.repeat(1, num_replica)[None, :, :, None, None]
            z = ((z - mean) / std).to(z.dtype)
        else:
            raise NotImplementedError
        return z

    def encode(
        self,
        x: torch.Tensor,
        is_training=False,
        sample_posterior=True,
        return_posterior=False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if is_training:  # only use micro_batch_size during inference
            self.micro_batch_size = None
            self.micro_batch_size_2d = None

        if self.micro_batch_size is None:
            z = self.encoder(x, is_training=is_training)
        else:
            z = self.encode_micro_batch(x)

        posterior = DiagonalGaussianDistribution(z)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # scale for dit training
        z = self.normalize_z(z)

        if return_posterior:
            return z, posterior
        else:
            return z

    def decode_micro_frame(self, z):
        x_list = []
        # z.shape[2] =  n * micro_z_frame_size
        micro_z_frame_size = int((self.micro_frame_size - 1) / 4 + 1) if self.micro_frame_size else None
        for i in range(0, z.size(2), micro_z_frame_size):
            z_bs = z[:, :, i : i + micro_z_frame_size]
            x_bs = self.decoder(z_bs)
            if self.temporal_overlap and i > 0:
                x_list[-1][:, :, -1:, :, :] = 0.5 * (x_list[-1][:, :, -1:, :, :] + x_bs[:, :, :1, :, :])
                x_bs = x_bs[:, :, 1:, :, :]
            x_list.append(x_bs)
        x = torch.cat(x_list, dim=2)
        return x

    def decode_micro_batch(self, z):
        x_list = []
        # x.shape[0] = n * self.micro_batch_size
        for i in range(0, z.shape[0], self.micro_batch_size):
            z_bs = z[i : i + self.micro_batch_size]
            if self.micro_frame_size is None:
                x_list.append(self.decoder(z_bs))
            else:
                x_list.append(self.decode_micro_frame(z_bs))
        x = torch.cat(x_list, dim=0)
        return x

    def un_normalize_z(self, z):
        if self.normalization is None:
            return z
        elif self.normalization == "video":
            mean, std = get_vae_stats("video", z)
            mean = mean[None, :, None, None, None]
            std = std[None, :, None, None, None]
            z = (z * std + mean).to(z.dtype)
        elif self.normalization == "video_5frames":
            mean, std = get_vae_stats("video_5frames", z)
            if z.size(2) == 1:
                mean = mean[None, :, :1, None, None]
                std = std[None, :, :1, None, None]
            else:
                num_replica = z.size(2) // mean.size(1)
                mean = mean.repeat(1, num_replica)[None, :, :, None, None]
                std = std.repeat(1, num_replica)[None, :, :, None, None]
            z = (z * std + mean).to(z.dtype)
        else:
            raise NotImplementedError
        return z

    def decode(self, z: torch.Tensor, is_training=False, **kwargs) -> torch.Tensor:
        # revert back for vae inference
        z = self.un_normalize_z(z)

        if is_training:
            self.micro_batch_size = None
            self.micro_batch_size_2d = None

        if self.micro_batch_size is None:
            x = self.decoder(z, is_training=is_training)
        else:
            x = self.decode_micro_batch(z)
        return x

    def forward(self, x: torch.Tensor, is_training=False) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, posterior = self.encode(x, is_training=is_training, return_posterior=True)
        dec = self.decode(z.to(x.dtype), is_training=is_training)
        return z, dec, posterior


@MODELS.register_module()
def OpenSoraVAE_V1_3(
    from_pretrained=None,
    force_huggingface=False,
    dtype="bf16",
    z_channels=16,
    micro_batch_size=None,
    micro_batch_size_2d=None,
    micro_frame_size=None,
    use_tiled_conv3d=False,
    tile_size=16,
    tiled_dim=None,
    num_tiles=None,
    temporal_overlap=False,
    normalization=None,
):
    encoder_config = dict(
        type="VideoEncoder",
        double_z=True,
        z_channels=z_channels,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        down_sampling_layer=[1, 2],
        micro_batch_size_2d=micro_batch_size_2d,
    )

    decoder_config = dict(
        type="VideoDecoder",
        z_channels=z_channels,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temporal_up_layers=[2, 3],
        micro_batch_size_2d=micro_batch_size_2d,
    )

    dtype = to_torch_dtype(dtype)

    kwargs = dict(
        encoder=encoder_config,
        decoder=decoder_config,
    )

    if force_huggingface or (from_pretrained is not None and not os.path.exists(from_pretrained)):
        model = OpenSoraVAE_V1_3_Pipline.from_pretrained(from_pretrained, **kwargs)
        model.micro_batch_size = micro_batch_size
        model.micro_batch_size_2d = micro_batch_size_2d
        model.micro_frame_size = micro_frame_size
        model.encoder.micro_batch_size = micro_batch_size
        model.decoder.micro_batch_size = micro_batch_size
        model.normalization = normalization
        model.temporal_overlap = temporal_overlap

        if use_tiled_conv3d:
            model._enable_tiled_conv3d(tile_size=tile_size, tiled_dim=tiled_dim, num_tiles=num_tiles)
        print(f"huggingface model loaded: {from_pretrained}")
    else:
        config = OpenSoraVAE_V1_3_PiplineConfig(**kwargs)
        model = OpenSoraVAE_V1_3_Pipline(
            config,
            use_tiled_conv3d=use_tiled_conv3d,
            micro_batch_size=micro_batch_size,
            micro_batch_size_2d=micro_batch_size_2d,
            micro_frame_size=micro_frame_size,
            tile_size=16,
            tiled_dim=None,
            num_tiles=None,
            temporal_overlap=temporal_overlap,
            normalization=normalization,
        )
        if from_pretrained:
            load_checkpoint(model, from_pretrained)
            print(f"local model loaded: {from_pretrained}")
    model.encoder.to(dtype)
    model.decoder.to(dtype)

    return model
