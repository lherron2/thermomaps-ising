import torch
from torch import vmap
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def polynomial_noise(t, alpha_max, alpha_min, s=1e-5):
    """
    Generate polynomial noise schedule used in Hoogeboom et. al.

    Args:
        t (torch.Tensor): Time steps.
        alpha_max (float): Maximum alpha value.
        alpha_min (float): Minimum alpha value.
        s (float): Smoothing factor.

    Returns:
        torch.Tensor: Alpha schedule.
    """
    T = t[-1]
    alphas = (1 - 2 * s) * (1 - (t / T) ** 2) + s
    a = alphas[1:] / alphas[:-1]
    a[a**2 < 0.001] = 0.001
    alpha_schedule = torch.cumprod(a, 0)
    return alpha_schedule


NOISE_FUNCS = {
    "polynomial": polynomial_noise,
}


class DiffusionProcess:
    """
    Instantiates the noise parameterization, rescaling of noise distribution,
    and timesteps for a diffusion process.
    """

    def __init__(
        self,
        num_diffusion_timesteps,
        noise_schedule,
        alpha_max,
        alpha_min,
        NOISE_FUNCS,
    ):
        """
        Initialize a DiffusionProcess.

        Args:
            num_diffusion_timesteps (int): Number of diffusion timesteps.
            noise_schedule (str): Noise schedule type.
            alpha_max (float): Maximum alpha value.
            alpha_min (float): Minimum alpha value.
            NOISE_FUNCS (dict): Dictionary of noise functions.
        """
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.times = torch.arange(num_diffusion_timesteps)
        self.alphas = NOISE_FUNCS[noise_schedule](
            torch.arange(num_diffusion_timesteps + 1), alpha_max, alpha_min
        )


class VPDiffusion(DiffusionProcess):
    """
    Subclass of a DiffusionProcess: Performs a diffusion according to the VP-SDE.
    """

    def __init__(
        self,
        num_diffusion_timesteps,
        noise_schedule="polynomial",
        alpha_max=20.0,
        alpha_min=0.01,
        NOISE_FUNCS=NOISE_FUNCS,
    ):
        """
        Initialize a VPDiffusion process.

        Args:
            num_diffusion_timesteps (int): Number of diffusion timesteps.
            noise_schedule (str): Noise schedule type.
            alpha_max (float): Maximum alpha value.
            alpha_min (float): Minimum alpha value.
            NOISE_FUNCS (dict): Dictionary of noise functions.
        """
        super().__init__(
            num_diffusion_timesteps, noise_schedule, alpha_max, alpha_min, NOISE_FUNCS
        )

        self.bmul = vmap(torch.mul)

    def get_alphas(self):
        """
        Get alpha values.

        Returns:
            torch.Tensor: Alpha values.
        """
        return self.alphas

    def forward_kernel(self, x0, t, prior, **prior_kwargs):
        """
        Marginal transition kernels of the forward process. q(x_t|x_0).

        Args:
            x0 (torch.Tensor): Initial data.
            t (int): Time step.
            prior: Prior distribution.
            **prior_kwargs: Additional keyword arguments.

        Returns:
            tuple: Tuple containing x_t and noise.
        """
        alphas_t = self.alphas[t]
        noise = prior.sample(**prior_kwargs)
        x_t = self.bmul(x0, alphas_t.sqrt()) + self.bmul(noise, (1 - alphas_t).sqrt())
        return x_t, noise

    def reverse_kernel(self, x_t, t, backbone, pred_type):
        """
        Marginal transition kernels of the reverse process. p(x_0|x_t).

        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Time step.
            backbone: Backbone model.
            pred_type (str): Type of prediction.

        Returns:
            tuple: Tuple containing x0_t and noise.
        """
        alphas_t = self.alphas[t]
        if pred_type == "noise":
            noise = backbone(x_t, alphas_t)
            noise_interp = self.bmul(noise, (1 - alphas_t).sqrt())
            x0_t = self.bmul((x_t - noise_interp), 1 / alphas_t.sqrt())

        elif pred_type == "x0":
            x0_t = backbone(x_t, alphas_t)
            x0_interp = self.bmul(x0_t, alphas_t.sqrt())
            noise = self.bmul((x_t - x0_interp), 1 / (1 - alphas_t).sqrt())

        else:
            raise Exception("Please provide a valid prediction type: 'noise' or 'x0'")

        return x0_t, noise

    def reverse_step(self, x_t, t, t_next, backbone, pred_type, eta, prior, **prior_kwargs):
        """
        Stepwise transition kernel of the reverse process p(x_t-1|x_t).

        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Time step t.
            t_next (int): Time step t_next.
            backbone: Backbone model.
            pred_type (str): Type of prediction.

        Returns:
            torch.Tensor: Data at time t_next.
        """
        alphas_t_next = self.alphas[t_next]
        alphas_t = self.alphas[t]
        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type)
        c1 = eta * ((1 - alphas_t / alphas_t_next) * (1 - alphas_t_next) / (1 - alphas_t)).sqrt()
        c2 = ((1 - alphas_t_next) - c1**2).sqrt()
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul(c1, noise) + self.bmul(c2, prior.sample(**prior_kwargs))
        return xt_next

    def compute_SNR(self, t):
        """
        Compute Signal-to-Noise Ratio (SNR) at a given time step.

        Args:
            t (int): Time step.

        Returns:
            torch.Tensor: SNR value.
        """
        alpha_sq = self.alphas[t.long()].pow(2)
        sigma_sq = 1 - alpha_sq
        gamma_t = -(torch.log(alpha_sq) - torch.log(sigma_sq))
        return torch.exp(-gamma_t)


class SteeredVPDiffusion(VPDiffusion):
    """
    Subclass of VPDiffusion: VPDiffusion with steering control.
    """

    def __init__(
        self,
        num_diffusion_timesteps,
        noise_schedule="polynomial",
        alpha_max=20.0,
        alpha_min=0.01,
        NOISE_FUNCS=NOISE_FUNCS,
    ):
        """
        Initialize a SteeredVPDiffusion process.

        Args:
            num_diffusion_timesteps (int): Number of diffusion timesteps.
            noise_schedule (str): Noise schedule type.
            alpha_max (float): Maximum alpha value.
            alpha_min (float): Minimum alpha value.
            NOISE_FUNCS (dict): Dictionary of noise functions.
        """
        super().__init__(
            num_diffusion_timesteps, noise_schedule, alpha_max, alpha_min, NOISE_FUNCS
        )

    def reverse_step(self, x_t, t, t_next, backbone, pred_type):
        """
        Stepwise transition kernel of the reverse process p(x_t-1|x_t) with steering control.

        Args:
            x_t (torch.Tensor): Data at time t.
            t (int): Time step t.
            t_next (int): Time step t_next.
            backbone: Backbone model.
            pred_type (str): Type of prediction.

        Returns:
            torch.Tensor: Data at time t_next.
        """
        alphas_t_next = self.alphas[t_next]
        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type)
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul(
            (1 - alphas_t_next).sqrt(), noise
        )
        return xt_next
