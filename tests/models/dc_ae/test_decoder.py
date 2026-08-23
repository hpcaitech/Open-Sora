from opensora.models.dc_ae.models.dc_ae import Decoder, DecoderConfig, Encoder, EncoderConfig


def test_encoder_defaults_empty_temporal_downsample_to_false():
    encoder = Encoder(
        EncoderConfig(
            in_channels=3,
            latent_channels=4,
            width_list=(4, 4),
            depth_list=(1, 1),
            norm="rms3d",
            is_video=True,
            downsample_block_type="Conv",
        )
    )

    downsample = encoder.stages[0].op_list[-1]
    assert downsample.main.conv.stride == (1, 2, 2)
    assert downsample.shortcut.temporal_downsample is False


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
