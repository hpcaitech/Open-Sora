import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

# from .builder import MODELS
# from .pixart import PixArt, PixArtBlock
# TODO:
from nanosora.utils.communications import gather_forward_split_backward, split_forward_gather_backward
from nanosora.utils.parallel_states import get_sequence_parallel_group
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from opensora.registry import MODELS

from .blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)

approx_gelu = lambda: nn.GELU(approximate="tanh")


class OpenSoraBlock(nn.Module):  # Inherit PixArtBlock
    def __init__(
        self,
        # hidden_size,
        # num_heads,
        # *args,
        # d_s=None,
        # d_t=None,
        # **kwargs,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self._enable_gradient_checkpoint = False
        self._enable_sequence_parallelism = enable_sequence_parallelism

        if enable_sequence_parallelism:
            self.attn_cls = SeqParallelAttention
            self.mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            self.attn_cls = Attention
            self.mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # new code
        self.d_s = d_s
        self.d_t = d_t

        if self._enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            # make sure d_t is divisible by sp_size
            assert d_t % sp_size == 0
            self.d_t = d_t // sp_size

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
        )

    def forward(self, x, y, t, mask=None, tpe=None):
        B, N, C = x.shape

        def _custom_spatial_attn_forward(*inputs):
            if self._enable_gradient_checkpoint:
                return torch.utils.checkpoint.checkpoint(self.attn, *inputs)
            else:
                return self.attn(*inputs)

        def _custom_temporal_attn_forward(*inputs):
            if self._enable_gradient_checkpoint:
                return torch.utils.checkpoint.checkpoint(self.attn_temp, *inputs)
            else:
                return self.attn_temp(*inputs)

        def _custom_cross_attn_forward(*inputs):
            if self._enable_gradient_checkpoint:
                return torch.utils.checkpoint.checkpoint(self.cross_attn, *inputs)
            else:
                return self.cross_attn(*inputs)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = rearrange(x_m, "b (t s) d -> (b t) s d", t=self.d_t, s=self.d_s)
        x_s = _custom_spatial_attn_forward(x_s)
        x_s = rearrange(x_s, "(b t) s d -> b (t s) d", t=self.d_t, s=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        x_t = rearrange(x, "b (t s) d -> (b s) t d", t=self.d_t, s=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = _custom_temporal_attn_forward(x_t)
        x_t = rearrange(x_t, "(b s) t d -> b (t s) d", t=self.d_t, s=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        x = x + _custom_cross_attn_forward(x, y, mask)

        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

    def enable_gradient_checkpointing(self):
        self._enable_gradient_checkpoint = True


@MODELS.register_module()
class OpenSora(nn.Module):  # inherit PixArt
    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        # condition
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        dtype=torch.float32,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_sequence_parallelism=False,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.space_scale = space_scale
        self.time_scale = time_scale

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        # TODO: check init logic
        # drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        # self.blocks = nn.ModuleList(
        #     [
        #         PixArtBlock(
        #             hidden_size,
        #             num_heads,
        #             mlp_ratio=mlp_ratio,
        #             drop_path=drop_path[i],
        #             enable_flashattn=enable_flashattn,
        #             enable_layernorm_kernel=enable_layernorm_kernel,
        #         )
        #         for i in range(depth)
        #     ]
        # )
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        # self.initialize_weights()
        if freeze is not None:
            assert freeze in ["text"]
            if freeze == "text":
                self.freeze_text()

        # super().__init__(*args, **kwargs)
        # replace the PixArtBlock with STPixArtBlock
        # TODO: bug??? initialize_weights() above has no effect on self.blocks
        self.blocks = nn.ModuleList(
            [
                OpenSoraBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                )
                for _ in range(self.depth)
            ]
        )

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = enable_sequence_parallelism

        if enable_sequence_parallelism:
            self.sp_rank = dist.get_rank(get_sequence_parallel_group())
        else:
            self.sp_rank = None

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def forward(self, x, timestep, y, mask=None):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "b t s d -> b (t s) d")

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="down")

        t = self.t_embedder(timestep, dtype=x.dtype)  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for i, block in enumerate(self.blocks):
            additional_args = {}
            if i == 0:
                if self.enable_sequence_parallelism:
                    additional_args["tpe"] = torch.chunk(
                        self.pos_embed_temporal, dist.get_world_size(get_sequence_parallel_group()), dim=1
                    )[self.sp_rank].contiguous()
                else:
                    additional_args["tpe"] = self.pos_embed_temporal
            x = block(x, y, t0, y_lens, **additional_args)

        if self.enable_sequence_parallelism:
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="up")

        # final process
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def enable_gradient_checkpointing(self):
        for blk in self.blocks:
            blk.enable_gradient_checkpointing()

    def forward_with_dpmsolver(self, x, timestep, y, mask=None):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out["x"] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@MODELS.register_module("OpenSora-XL/2")
def OpenSora_XL_2(**kwargs):
    return OpenSora(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
