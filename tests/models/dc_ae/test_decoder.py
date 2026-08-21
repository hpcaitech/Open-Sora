from opensora.models.dc_ae.models.dc_ae import Decoder, DecoderConfig


def test_decoder_defaults_empty_temporal_upsample_to_false():
    decoder = Decoder(
        DecoderConfig(
            in_channels=3,
            latent_channels=4,
            width_list=(4, 4),
            depth_list=(1, 1),
            norm="rms3d",
            is_video=True,
            upsample_block_type="InterpolateConv",
        )
    )

    upsample = decoder.stages[0].op_list[0]
    assert upsample.main.temporal_upsample is False
    assert upsample.shortcut.temporal_upsample is False
