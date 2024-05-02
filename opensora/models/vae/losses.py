import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


# from MAGVIT, used in place hof hinge_d_loss
def sigmoid_cross_entropy_with_logits(labels, logits):
    # The final formulation is: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
    lecam_loss = torch.mean(torch.pow(nn.ReLU()(real_pred - ema_fake_pred), 2))
    lecam_loss += torch.mean(torch.pow(nn.ReLU()(ema_real_pred - fake_pred), 2))
    return lecam_loss


def gradient_penalty_fn(images, output):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class VAELoss(nn.Module):
    def __init__(
        self,
        logvar_init=0.0,
        perceptual_loss_weight=0.1,
        kl_loss_weight=0.000001,
        device="cpu",
        dtype="bf16",
    ):
        super().__init__()

        if type(dtype) == str:
            if dtype == "bf16":
                dtype = torch.bfloat16
            elif dtype == "fp16":
                dtype = torch.float16
            else:
                raise NotImplementedError(f"dtype: {dtype}")

        # KL Loss
        self.kl_loss_weight = kl_loss_weight
        # Perceptual Loss
        self.perceptual_loss_fn = LPIPS().eval().to(device, dtype)
        self.perceptual_loss_weight = perceptual_loss_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(
        self,
        video,
        recon_video,
        posterior,
        nll_weights=None,
        no_perceptual=False,
    ):
        video = rearrange(video, "b c t h w -> (b t) c h w").contiguous()
        recon_video = rearrange(recon_video, "b c t h w -> (b t) c h w").contiguous()

        # reconstruction loss
        recon_loss = torch.abs(video - recon_video)

        # perceptual loss
        if self.perceptual_loss_weight is not None and self.perceptual_loss_weight > 0.0 and not no_perceptual:
            # handle channels
            channels = video.shape[1]
            assert channels in {1, 3}
            if channels == 1:
                input_vgg_input = repeat(video, "b 1 h w -> b c h w", c=3)
                recon_vgg_input = repeat(recon_video, "b 1 h w -> b c h w", c=3)
            else:
                input_vgg_input = video
                recon_vgg_input = recon_video

            perceptual_loss = self.perceptual_loss_fn(input_vgg_input, recon_vgg_input)
            recon_loss = recon_loss + self.perceptual_loss_weight * perceptual_loss

        nll_loss = recon_loss / torch.exp(self.logvar) + self.logvar

        weighted_nll_loss = nll_loss
        if nll_weights is not None:
            weighted_nll_loss = nll_weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # KL Loss
        weighted_kl_loss = 0
        if self.kl_loss_weight is not None and self.kl_loss_weight > 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            weighted_kl_loss = kl_loss * self.kl_loss_weight

        return nll_loss, weighted_nll_loss, weighted_kl_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


class AdversarialLoss(nn.Module):
    def __init__(
        self,
        discriminator_factor=1.0,
        discriminator_start=50001,
        generator_factor=0.5,
        generator_loss_type="non-saturating",
    ):
        super().__init__()
        self.discriminator_factor = discriminator_factor
        self.discriminator_start = discriminator_start
        self.generator_factor = generator_factor
        self.generator_loss_type = generator_loss_type

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.generator_factor
        return d_weight

    def forward(
        self,
        fake_logits,
        nll_loss,
        last_layer,
        global_step,
        is_training=True,
    ):
        # NOTE: following MAGVIT to allow non_saturating
        assert self.generator_loss_type in ["hinge", "vanilla", "non-saturating"]

        if self.generator_loss_type == "hinge":
            gen_loss = -torch.mean(fake_logits)
        elif self.generator_loss_type == "non-saturating":
            gen_loss = torch.mean(
                sigmoid_cross_entropy_with_logits(labels=torch.ones_like(fake_logits), logits=fake_logits)
            )
        else:
            raise ValueError("Generator loss {} not supported".format(self.generator_loss_type))

        if self.discriminator_factor is not None and self.discriminator_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, gen_loss, last_layer)
            except RuntimeError:
                assert not is_training
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_start)
        weighted_gen_loss = d_weight * disc_factor * gen_loss

        return weighted_gen_loss


class LeCamEMA:
    def __init__(self, ema_real=0.0, ema_fake=0.0, decay=0.999, dtype=torch.bfloat16, device="cpu"):
        self.decay = decay
        self.ema_real = torch.tensor(ema_real).to(device, dtype)
        self.ema_fake = torch.tensor(ema_fake).to(device, dtype)

    def update(self, ema_real, ema_fake):
        self.ema_real = self.ema_real * self.decay + ema_real * (1 - self.decay)
        self.ema_fake = self.ema_fake * self.decay + ema_fake * (1 - self.decay)

    def get(self):
        return self.ema_real, self.ema_fake


class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        discriminator_factor=1.0,
        discriminator_start=50001,
        discriminator_loss_type="non-saturating",
        lecam_loss_weight=None,
        gradient_penalty_loss_weight=None,  # SCH: following MAGVIT config.vqgan.grad_penalty_cost
    ):
        super().__init__()

        assert discriminator_loss_type in ["hinge", "vanilla", "non-saturating"]
        self.discriminator_factor = discriminator_factor
        self.discriminator_start = discriminator_start
        self.lecam_loss_weight = lecam_loss_weight
        self.gradient_penalty_loss_weight = gradient_penalty_loss_weight
        self.discriminator_loss_type = discriminator_loss_type

    def forward(
        self,
        real_logits,
        fake_logits,
        global_step,
        lecam_ema_real=None,
        lecam_ema_fake=None,
        real_video=None,
        split="train",
    ):
        if self.discriminator_factor is not None and self.discriminator_factor > 0.0:
            disc_factor = adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_start)

            if self.discriminator_loss_type == "hinge":
                disc_loss = hinge_d_loss(real_logits, fake_logits)
            elif self.discriminator_loss_type == "non-saturating":
                if real_logits is not None:
                    real_loss = sigmoid_cross_entropy_with_logits(
                        labels=torch.ones_like(real_logits), logits=real_logits
                    )
                else:
                    real_loss = 0.0
                if fake_logits is not None:
                    fake_loss = sigmoid_cross_entropy_with_logits(
                        labels=torch.zeros_like(fake_logits), logits=fake_logits
                    )
                else:
                    fake_loss = 0.0
                disc_loss = 0.5 * (torch.mean(real_loss) + torch.mean(fake_loss))
            elif self.discriminator_loss_type == "vanilla":
                disc_loss = vanilla_d_loss(real_logits, fake_logits)
            else:
                raise ValueError(f"Unknown GAN loss '{self.discriminator_loss_type}'.")

            weighted_d_adversarial_loss = disc_factor * disc_loss

        else:
            weighted_d_adversarial_loss = 0

        lecam_loss = torch.tensor(0.0)
        if self.lecam_loss_weight is not None and self.lecam_loss_weight > 0.0:
            real_pred = torch.mean(real_logits)
            fake_pred = torch.mean(fake_logits)
            lecam_loss = lecam_reg(real_pred, fake_pred, lecam_ema_real, lecam_ema_fake)
            lecam_loss = lecam_loss * self.lecam_loss_weight

        gradient_penalty = torch.tensor(0.0)
        if self.gradient_penalty_loss_weight is not None and self.gradient_penalty_loss_weight > 0.0:
            assert real_video is not None
            gradient_penalty = gradient_penalty_fn(real_video, real_logits)
            gradient_penalty *= self.gradient_penalty_loss_weight

        return (weighted_d_adversarial_loss, lecam_loss, gradient_penalty)
