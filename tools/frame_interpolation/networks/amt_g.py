import torch
import torch.nn as nn

from .blocks.feat_enc import LargeEncoder
from .blocks.ifrnet import Encoder, InitDecoder, IntermediateDecoder, resize
from .blocks.multi_flow import MultiFlowDecoder, multi_flow_combine
from .blocks.raft import BasicUpdateBlock, BidirCorrBlock, coords_grid


class Model(nn.Module):
    def __init__(self, corr_radius=3, corr_lvls=4, num_flows=5, channels=[84, 96, 112, 128], skip_channels=84):
        super(Model, self).__init__()
        self.radius = corr_radius
        self.corr_levels = corr_lvls
        self.num_flows = num_flows

        self.feat_encoder = LargeEncoder(output_dim=128, norm_fn="instance", dropout=0.0)
        self.encoder = Encoder(channels, large=True)
        self.decoder4 = InitDecoder(channels[3], channels[2], skip_channels)
        self.decoder3 = IntermediateDecoder(channels[2], channels[1], skip_channels)
        self.decoder2 = IntermediateDecoder(channels[1], channels[0], skip_channels)
        self.decoder1 = MultiFlowDecoder(channels[0], skip_channels, num_flows)

        self.update4 = self._get_updateblock(112, None)
        self.update3_low = self._get_updateblock(96, 2.0)
        self.update2_low = self._get_updateblock(84, 4.0)

        self.update3_high = self._get_updateblock(96, None)
        self.update2_high = self._get_updateblock(84, None)

        self.comb_block = nn.Sequential(
            nn.Conv2d(3 * self.num_flows, 6 * self.num_flows, 7, 1, 3),
            nn.PReLU(6 * self.num_flows),
            nn.Conv2d(6 * self.num_flows, 3, 7, 1, 3),
        )

    def _get_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(
            cdim=cdim,
            hidden_dim=192,
            flow_dim=64,
            corr_dim=256,
            corr_dim2=192,
            fc_dim=188,
            scale_factor=scale_factor,
            corr_levels=self.corr_levels,
            radius=self.radius,
        )

    def _corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        # convert t -> 0 to 0 -> 1 | convert t -> 1 to 1 -> 0
        # based on linear assumption
        t1_scale = 1.0 / embt
        t0_scale = 1.0 / (1.0 - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)

        corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale)
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow

    def forward(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)

        fmap0, fmap1 = self.feat_encoder([img0_, img1_])  # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord, up_flow0_4, up_flow1_4, embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_

        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_ = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn, coord, up_flow0_3, up_flow1_3, embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3_low(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_

        # residue update with lookup corr (hr)
        corr_3 = resize(corr_3, scale_factor=2.0)
        up_flow_3 = torch.cat([up_flow0_3, up_flow1_3], dim=1)
        delta_ft_2_, delta_up_flow_3 = self.update3_high(ft_2_, up_flow_3, corr_3)
        ft_2_ += delta_ft_2_
        up_flow0_3 += delta_up_flow_3[:, 0:2]
        up_flow1_3 += delta_up_flow_3[:, 2:4]

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_ = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn, coord, up_flow0_2, up_flow1_2, embt, downsample=4)

        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2_low(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_

        # residue update with lookup corr (hr)
        corr_2 = resize(corr_2, scale_factor=4.0)
        up_flow_2 = torch.cat([up_flow0_2, up_flow1_2], dim=1)
        delta_ft_1_, delta_up_flow_2 = self.update2_high(ft_1_, up_flow_2, corr_2)
        ft_1_ += delta_ft_1_
        up_flow0_2 += delta_up_flow_2[:, 0:2]
        up_flow1_2 += delta_up_flow_2[:, 2:4]

        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)

        if scale_factor != 1.0:
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            mask = resize(mask, scale_factor=(1.0 / scale_factor))
            img_res = resize(img_res, scale_factor=(1.0 / scale_factor))

        # Merge multiple predictions
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1, mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if eval:
            return {
                "imgt_pred": imgt_pred,
            }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                "imgt_pred": imgt_pred,
                "flow0_pred": [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                "flow1_pred": [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                "ft_pred": [ft_1_, ft_2_, ft_3_],
            }
