import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from opensora.models.vae.lpips import LPIPS


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


def wgan_gp_loss(logits_real, logits_fake):
    d_loss = 0.5 * (-logits_real.mean() + logits_fake.mean())
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


def batch_mean(x):
    return torch.sum(x) / x.shape[0]


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
        perceptual_loss_weight=1.0,
        kl_loss_weight=5e-4,
        device="cpu",
        dtype="bf16",
    ):
        super().__init__()

        if type(dtype) == str:
            if dtype == "bf16":
                dtype = torch.bfloat16
            elif dtype == "fp16":
                dtype = torch.float16
            elif dtype == "fp32":
                dtype = torch.float32
            else:
                raise NotImplementedError(f"dtype: {dtype}")

        # KL Loss
        self.kl_weight = kl_loss_weight
        # Perceptual Loss
        self.perceptual_loss_fn = LPIPS().eval().to(device, dtype)
        self.perceptual_loss_fn.requires_grad_(False)
        self.perceptual_loss_weight = perceptual_loss_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(
        self,
        video,
        recon_video,
        posterior,
    ) -> dict:
        video.size(0)
        video = rearrange(video, "b c t h w -> (b t) c h w").contiguous()
        recon_video = rearrange(recon_video, "b c t h w -> (b t) c h w").contiguous()

        # reconstruction loss
        recon_loss = l1(video, recon_video)

        # perceptual loss
        perceptual_loss = self.perceptual_loss_fn(video, recon_video)
        # nll loss (from reconstruction loss and perceptual loss)
        nll_loss = recon_loss + perceptual_loss * self.perceptual_loss_weight
        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar

        # Batch Mean
        nll_loss = batch_mean(nll_loss)
        recon_loss = batch_mean(recon_loss)
        numel_elements = video.numel() // video.size(0)
        perceptual_loss = batch_mean(perceptual_loss) * numel_elements

        # KL Loss
        if posterior is None:
            kl_loss = torch.tensor(0.0).to(video.device, video.dtype)
        else:
            kl_loss = posterior.kl()
            kl_loss = batch_mean(kl_loss)
        weighted_kl_loss = kl_loss * self.kl_weight

        return {
            "nll_loss": nll_loss,
            "kl_loss": weighted_kl_loss,
            "recon_loss": recon_loss,
            "perceptual_loss": perceptual_loss,
        }


class GeneratorLoss(nn.Module):
    def __init__(self, gen_start=2001, disc_factor=1.0, disc_weight=0.5):
        super().__init__()
        self.disc_factor = disc_factor
        self.gen_start = gen_start
        self.disc_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(
        self,
        logits_fake,
        nll_loss,
        last_layer,
        global_step,
        is_training=True,
    ):
        g_loss = -torch.mean(logits_fake)

        if self.disc_factor is not None and self.disc_factor > 0.0:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
        else:
            d_weight = torch.tensor(1.0)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.gen_start)
        weighted_gen_loss = d_weight * disc_factor * g_loss

        return weighted_gen_loss, g_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_start=2001, disc_factor=1.0, disc_loss_type="hinge"):
        super().__init__()

        assert disc_loss_type in ["hinge", "vanilla", "wgan-gp"]
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.disc_loss_type = disc_loss_type

        if self.disc_loss_type == "hinge":
            self.loss_fn = hinge_d_loss
        elif self.disc_loss_type == "vanilla":
            self.loss_fn = vanilla_d_loss
        elif self.disc_loss_type == "wgan-gp":
            self.loss_fn = wgan_gp_loss
        else:
            raise ValueError(f"Unknown GAN loss '{self.disc_loss_type}'.")

    def forward(
        self,
        real_logits,
        fake_logits,
        global_step,
    ):
        if self.disc_factor is not None and self.disc_factor > 0.0:
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
            disc_loss = self.loss_fn(real_logits, fake_logits)
            weighted_discriminator_loss = disc_factor * disc_loss
        else:
            weighted_discriminator_loss = 0

        return weighted_discriminator_loss
