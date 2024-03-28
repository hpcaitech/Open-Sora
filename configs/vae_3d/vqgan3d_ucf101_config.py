# Copyright 2023 The videogvt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""
Configs for the VQGAN-3D on the UCF101.

usage: `config = vqgan3d_ucf101_config.get_config()`

"""

import ml_collections

from opensora.configs.vae_3d import vqgan2d_ucf101_config

VARIANT = 'VQGAN/3D'


def get_config(config_str='B'):
  """Returns the base experiment configuration."""
  version, *options = config_str.split('-')

  config = vqgan2d_ucf101_config.get_config(config_str)
  config.experiment_name = f'UCF101_{VARIANT}'
  model_class, model_type = VARIANT.split('/')

  # Overall
  config.batch_size = {'B': 128, 'L': 256, 'H': 256}[version]
  config.eval_batch_size = config.get_ref('batch_size') // 4
  config.num_training_epochs = {'B': 500, 'L': 2000, 'H': 2000}[version]

  # Dataset.
  del config.num_train_sampled_frames

  # Model: vqvae
  config.model_class = model_class

  config.pretrained_image_model = True  # TODO(Lijun-Yu): 3d perceptual loss

  config.vqgan.model_type = model_type

  config.vqvae.in_channels = 3  # NOTE: SCH: added
  config.vqvae.num_groups = 32 # NOTE: SCH: added, for nn.GroupNorm
  config.vqvae.architecture = '3dcnn'
  config.vqvae.codebook_size = {'B': 1024, 'L': 1024, 'H': 4096}[version]
  config.vqvae.filters = {'B': 64, 'L': 128, 'H': 128}[version]
  config.vqvae.num_enc_res_blocks = {'B': 2, 'L': 2, 'H': 4}[version]
  config.vqvae.num_dec_res_blocks = {'B': 2, 'L': 2, 'H': 4}[version]
  config.vqvae.channel_multipliers = (1, 2, 2, 4)
  config.vqvae.temporal_downsample = (True, True, False)
  config.vqvae.embedding_dim = {'B': 256, 'L': 256, 'H': 512}[version] # z embedding dimension
  config.vqvae.conv_downsample = False
  config.vqvae.deconv_upsample = False

  # Save memory
  config.vqvae.num_enc_remat_blocks = {'B': 0, 'L': 1, 'H': 3}[version]
  config.vqvae.num_dec_remat_blocks = config.vqvae.get_ref(
      'num_enc_remat_blocks')
  config.discriminator.num_remat_blocks = config.vqvae.get_ref(
      'num_enc_remat_blocks')

  # Pretrained models on ImageNet.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.inflation = '2d->3d'

  # Standalone evaluation.
  if 'eval' in options:
    config.eval_only = True
    config.eval_from.checkpoint_path = {
        'B': 'gs://magvit/models/ucf_3d_base',
        'L': 'gs://magvit/models/ucf_3d_large',
    }[version]
    config.eval_from.step = -1
    config.eval_from.legacy_checkpoint = True

  if 'runlocal' in options:
    config.batch_size = 16
    config.num_training_epochs = 10
    # gets a small model for debugging
    config.vqvae.filters = 32
    config.vqvae.embedding_dim = 16
    config.vqvae.num_enc_res_blocks = 1
    config.vqvae.num_dec_res_blocks = 1
    config.discriminator.filters = 1
    config.discriminator.channel_multipliers = (1,)
    config.vqvae.channel_multipliers = (1,)
    config.vqvae.codebook_size = 128

  return config


