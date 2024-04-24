from typing import Callable

import numpy as np
import torch


# ==================================
# Warm Up Beta
# ==================================
def _warmup_beta_numpy(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def _warmup_beta_torch(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=torch.float64)
    return betas


def test_warmup_beta():
    beta_start = 1e-6
    beta_end = 0.99
    num_diffusion_timesteps = 1000
    warmup_frac = 0.1
    betas_np = _warmup_beta_numpy(beta_start, beta_end, num_diffusion_timesteps, warmup_frac)
    betas_torch = _warmup_beta_torch(beta_start, beta_end, num_diffusion_timesteps, warmup_frac)
    assert np.allclose(betas_np, betas_torch.numpy())
    print("Test passed for warmup_beta()")


# ==================================
# Beta Schedule
# ==================================


def get_beta_schedule_numpy(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta_numpy(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta_numpy(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_beta_schedule_torch(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta_torch(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta_torch(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=torch.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def test_get_beta_Schedule():
    beta_start = 1e-6
    beta_end = 0.99
    num_diffusion_timesteps = 1000
    beta_schedule = "linear"
    betas_np = get_beta_schedule_numpy(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
    )
    betas_torch = get_beta_schedule_torch(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
    )
    assert np.allclose(betas_np, betas_torch.numpy())
    print("Test passed for get_beta_schedule()")


# ====================
# Replace alpha
# ====================
def betas_for_alpha_bar_numpy(num_diffusion_timesteps: int, alpha_bar: Callable, max_beta: float = 0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_torch(num_diffusion_timesteps: int, alpha_bar: Callable, max_beta: float = 0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.DoubleTensor(betas)


def test_betas_for_alpha_bar():
    num_diffusion_timesteps = 1000
    alpha_bar = lambda t: 1 - t
    max_beta = 0.999
    betas_np = betas_for_alpha_bar_numpy(num_diffusion_timesteps, alpha_bar, max_beta)
    betas_torch = betas_for_alpha_bar_torch(num_diffusion_timesteps, alpha_bar, max_beta)
    assert np.allclose(betas_np, betas_torch.numpy())
    print("Test passed for betas_for_alpha_bar()")


# =======================
# Gaussian init
# =======================
def init_numpy(betas):
    # Use float64 for accuracy.
    betas = torch.DoubleTensor(betas)
    assert len(betas.shape) == 1, "betas must be 1-D"
    assert (betas > 0).all() and (betas <= 1).all()

    num_timesteps = int(betas.shape[0])

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
    assert alphas_cumprod_prev.shape == (num_timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    np.sqrt(alphas_cumprod)
    np.sqrt(1.0 - alphas_cumprod)
    np.log(1.0 - alphas_cumprod)
    np.sqrt(1.0 / alphas_cumprod)
    np.sqrt(1.0 / alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    posterior_log_variance_clipped = (
        np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        if len(posterior_variance) > 1
        else np.array([])
    )

    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    return alphas_cumprod_prev, alphas_cumprod_next, posterior_mean_coef1, posterior_mean_coef2


def gaussian_init_numpy(betas):
    # Use float64 for accuracy.
    betas = np.array(betas, dtype=np.float64)
    assert len(betas.shape) == 1, "betas must be 1-D"
    assert (betas > 0).all() and (betas <= 1).all()

    num_timesteps = int(betas.shape[0])

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
    assert alphas_cumprod_prev.shape == (num_timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    posterior_log_variance_clipped = (
        np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        if len(posterior_variance) > 1
        else np.array([])
    )

    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    return (
        alphas_cumprod_prev,
        alphas_cumprod_next,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        log_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
        posterior_log_variance_clipped,
        posterior_mean_coef1,
        posterior_mean_coef2,
    )


def gaussian_init_torch(betas):
    # Use float64 for accuracy.
    betas = torch.DoubleTensor(betas)
    assert len(betas.shape) == 1, "betas must be 1-D"
    assert (betas > 0).all() and (betas <= 1).all()

    num_timesteps = int(betas.shape[0])

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.tensor([0.0])])
    assert alphas_cumprod_prev.shape == (num_timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

    posterior_log_variance_clipped = (
        torch.log(torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]]))
        if len(posterior_variance) > 1
        else torch.array([])
    )

    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    return (
        alphas_cumprod_prev,
        alphas_cumprod_next,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        log_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
        posterior_log_variance_clipped,
        posterior_mean_coef1,
        posterior_mean_coef2,
    )


def test_gaussian_init():
    betas = np.linspace(1e-6, 0.99, 1000)
    (
        alphas_cumprod_prev,
        alphas_cumprod_next,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        log_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod,
        posterior_log_variance_clipped,
        posterior_mean_coef1,
        posterior_mean_coef2,
    ) = gaussian_init_numpy(betas)
    (
        alphas_cumprod_prev_t,
        alphas_cumprod_next_t,
        sqrt_alphas_cumprod_t,
        sqrt_one_minus_alphas_cumprod_t,
        log_one_minus_alphas_cumprod_t,
        sqrt_recip_alphas_cumprod_t,
        sqrt_recipm1_alphas_cumprod_t,
        posterior_log_variance_clipped_t,
        posterior_mean_coef1_t,
        posterior_mean_coef2_t,
    ) = gaussian_init_torch(betas)

    assert np.allclose(alphas_cumprod_prev, alphas_cumprod_prev_t.numpy())
    assert np.allclose(alphas_cumprod_next, alphas_cumprod_next_t.numpy())
    assert np.allclose(sqrt_alphas_cumprod, sqrt_alphas_cumprod_t.numpy())
    assert np.allclose(sqrt_one_minus_alphas_cumprod, sqrt_one_minus_alphas_cumprod_t.numpy())
    assert np.allclose(log_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod_t.numpy())
    assert np.allclose(sqrt_recip_alphas_cumprod, sqrt_recip_alphas_cumprod_t.numpy())
    assert np.allclose(sqrt_recipm1_alphas_cumprod, sqrt_recipm1_alphas_cumprod_t.numpy())
    assert np.allclose(posterior_log_variance_clipped, posterior_log_variance_clipped_t.numpy())
    assert np.allclose(posterior_mean_coef1, posterior_mean_coef1_t.numpy())
    assert np.allclose(posterior_mean_coef2, posterior_mean_coef2_t.numpy())
    print("Test passed for gaussian_init()")


if __name__ == "__main__":
    test_warmup_beta()
    test_get_beta_Schedule()
    test_betas_for_alpha_bar()
    test_gaussian_init()
