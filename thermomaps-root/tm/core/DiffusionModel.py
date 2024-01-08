import torch
import os
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

def temperature_density_rescaling(std_temp, ref_temp):
    """
    Calculate temperature density rescaling factor.

    Args:
        std_temp (float): The standard temperature.
        ref_temp (float): The reference temperature.

    Returns:
        float: The temperature density rescaling factor.
    """
    return (std_temp / ref_temp).pow(0.5)


def identity(t, *args, **kwargs):
    """
    Identity function.

    Args:
        t: Input tensor.

    Returns:
        t: Input tensor.
    """
    return t


RESCALE_FUNCS = {
    "density": temperature_density_rescaling,
    "no_rescale": identity,
}


class DiffusionModel:
    """
    Base class for diffusion models.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        pred_type,
        prior,
        rescale_func_name="no_rescale",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs
    ):
        """
        Initialize a DiffusionModel.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            control_ref (float): Control reference temperature.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        self.loader = loader
        self.BB = backbone
        self.DP = diffusion_process
        self.pred_type = pred_type
        self.rescale_func = RESCALE_FUNCS[rescale_func_name]
        self.prior = prior

    def noise_batch(self, b_t, t, prior, **prior_kwargs):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the forward noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        return self.DP.forward_kernel(b_t, t, prior, **prior_kwargs)

    def denoise_batch(self, b_t, t):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        return self.DP.reverse_kernel(b_t, t, self.BB, self.pred_type)

    def denoise_step(self, b_t, t, t_next):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        b_t_next = self.DP.reverse_step(b_t, t, t_next, self.BB, self.pred_type)
        return b_t_next

    def sample_times(self, num_times):
        """
        Randomly sample times from the time-discretization of the
        diffusion process.
        """
        return torch.randint(
            low=0, high=self.DP.num_diffusion_timesteps, size=(num_times,)
        ).long()

    @staticmethod
    def get_adjacent_times(times):
        """
        Pairs t with t+1 for all times in the time-discretization
        of the diffusion process.
        """
        times_next = torch.cat((torch.Tensor([0]).long(), times[:-1]))
        return list(zip(reversed(times), reversed(times_next)))


class DiffusionTrainer(DiffusionModel):
    """
    Subclass of a DiffusionModel: A trainer defines a loss function and
    performs backprop + optimizes model outputs.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        model_dir,
        pred_type,
        prior,
        optim=None,
        scheduler=None,
        rescale_func_name="density",
        RESCALE_FUNCS=RESCALE_FUNCS,
        device=0,
        identifier="model"
    ):
        """
        Initialize a DiffusionTrainer.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            optim: Optimizer.
            scheduler: Learning rate scheduler.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        super().__init__(
            diffusion_process,
            backbone,
            loader,
            pred_type,
            prior,
            rescale_func_name,
            RESCALE_FUNCS,
        )

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.identifier = identifier
        self.train_losses = []
        self.test_losses = []

    def loss_function(self, e, e_pred, weight, loss_type="l2"):
        """
        Loss function can be the l1-norm, l2-norm, or the VLB (weighted l2-norm).

        Args:
            e: Actual data.
            e_pred: Predicted data.
            weight: Weight factor.
            loss_type (str): Type of loss function.

        Returns:
            float: The loss value.
        """
        sum_indices = tuple(list(range(1, self.loader.num_dims)))

        def l1_loss(e, e_pred, weight):
            return (e - e_pred).abs().sum(sum_indices)
        
        def smooth_l1_loss(e, e_pred, weight):
            return torch.nn.functional.smooth_l1_loss(e, e_pred, reduction='mean')

        def l2_loss(e, e_pred, weight):
            return (e - e_pred).pow(2).sum((1, 2, 3)).pow(0.5).mean()

        def VLB_loss(e, e_pred, weight):
            return (weight * ((e - e_pred).pow(2).sum(sum_indices)).pow(0.5)).mean()

        loss_dict = {"l1": l1_loss, "l2": l2_loss, "VLB": VLB_loss}

        return loss_dict[loss_type](e, e_pred, weight)

    def train(
        self,
        num_epochs,
        grad_accumulation_steps=1,
        print_freq=10,
        batch_size=128,
        loss_type="l2",
    ):
        """
        Trains a diffusion model.

        Args:
            num_epochs (int): Number of training epochs.
            grad_accumulation_steps (int): Number of gradient accumulation steps.
            print_freq (int): Frequency of printing training progress.
            batch_size (int): Batch size.
            loss_type (str): Type of loss function.
        """
        train_loader = torch.utils.data.DataLoader(
            self.loader,
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(num_epochs):
            epoch += self.BB.start_epoch
            for i, (unstd_control, b) in enumerate(train_loader, 0):
                t = self.sample_times(b.size(0))
                t_prev = t - 1
                t_prev[t_prev == -1] = 0
                weight = self.DP.compute_SNR(t_prev) - self.DP.compute_SNR(t)

                logging.debug(f"{unstd_control}")


                noise, noise_pred = self.train_step(b, t, self.prior, batch_size=int(b.shape[0]),
                    temp=unstd_control, sample_type="from_data", n_dims=int(b.shape[1]) - 1,)

                loss = (self.loss_function(noise, noise_pred, weight, loss_type=loss_type) / grad_accumulation_steps)

                if i % grad_accumulation_steps == 0:
                    self.BB.optim.zero_grad()
                    # append loss to loss list
                    self.train_losses.append(loss.detach().cpu().numpy())
                    loss.backward()
                    self.BB.optim.step()

                # generate samples to test loss against.


                if i % print_freq == 0:
                    print(f"step: {i}, loss {loss.detach():.3f}")
            print(f"epoch: {epoch}")

            if self.BB.scheduler:
                self.BB.scheduler.step()

            self.BB.save_state(self.model_dir, epoch, identifier=self.identifier)

    def train_step(self, b, t, prior, **kwargs):
        """
        Training step.

        Args:
            b: Input batch.
            t: Sampled times.
            prior: Prior distribution.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple: (noise, noise_pred)
        """
        b_t, noise = self.noise_batch(b, t, prior, **kwargs)
        b_0, noise_pred = self.denoise_batch(b_t, t)
        return noise, noise_pred


class DiffusionSampler(DiffusionModel):
    """
    Subclass of a DiffusionModel: A sampler generates samples from random noise.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        sample_dir,
        pred_type,
        prior,
        rescale_func_name="density",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs
    ):
        """
        Initialize a DiffusionSampler.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        super().__init__(
            diffusion_process,
            backbone,
            loader,
            pred_type,
            prior,
            rescale_func_name,
            RESCALE_FUNCS,
            **kwargs
        )

        self.sample_dir = sample_dir
        os.makedirs(self.sample_dir, exist_ok=True)

    def sample_batch(self, **prior_kwargs):
        """
        Sample a batch of data.

        Args:
            **prior_kwargs: Keyword arguments for sampling.

        Returns:
            Tensor: Sampled batch.
        """
        batch_size = prior_kwargs["batch_size"]
        xt = self.prior.sample_prior(**prior_kwargs)
        stds = self.prior.fit_prior(**prior_kwargs)
        time_pairs = self.get_adjacent_times(self.DP.times)

        for t, t_next in time_pairs:
            t = torch.Tensor.repeat(t, batch_size)
            t_next = torch.Tensor.repeat(t_next, batch_size)
            xt_next = self.denoise_step(xt, t, t_next, control=stds)
            xt = xt_next
        return xt

    def save_batch(self, batch, save_prefix, temperature, save_idx):
        """
        Save a batch of samples.

        Args:
            batch: Batch of samples.
            save_prefix (str): Prefix for saving.
            temperature: Temperature for saving.
            save_idx (int): Index for saving.
        """
        save_path = os.path.join(self.sample_dir, f"{temperature}K")
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_path, f"{save_prefix}_idx={save_idx}.npz"), traj=batch
        )

    def sample_loop(self, num_samples, batch_size, save_prefix, temperature, n_ch):
        """
        Sampling loop.

        Args:
            num_samples (int): Number of samples to generate.
            batch_size (int): Batch size.
            save_prefix (str): Prefix for saving.
            temperature: Temperature for saving.
            n_ch: Number of channels.
        """
        n_runs = max(num_samples // batch_size, 1)
        if num_samples <= batch_size:
            batch_size = num_samples
        with torch.no_grad():
            for save_idx in range(n_runs):
                x0 = self.sample_batch(
                    batch_size=batch_size,
                    temp=temperature,
                    sample_type="from_fit",
                    n_dims=n_ch - 1,
                )
                x0 = x0[:, :-1, :, :]
                self.save_batch(x0, save_prefix, temperature, save_idx)


class SteeredDiffusionSampler(DiffusionSampler):
    """
    A DiffusionModel consists of instances of a DiffusionProcess, Backbone,
    and Loader objects.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        sample_dir,
        pred_type,
        prior,
        rescale_func_name="no_rescale",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs,
    ):
        """
        Initialize a SteeredDiffusionSampler.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            diffusion_process,
            backbone,
            loader,
            sample_dir,
            pred_type,
            prior,
            rescale_func_name,
            RESCALE_FUNCS,
            **kwargs,
        )

        self.kwargs = kwargs

    def denoise_step(self, b_t, t, t_next, control=None):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        gamma = self.kwargs["gamma"]
        b_t_next = self.DP.reverse_step(b_t, t, t_next, self.BB, self.pred_type)
        b_t_next[:, -1, :, :] = (1 - gamma) * b_t_next[:, -1, :, :] + gamma * control
        return b_t_next

    def sample_batch(self, **prior_kwargs):
        """
        Sample a batch of data.

        Args:
            **prior_kwargs: Keyword arguments for sampling.

        Returns:
            Tensor: Sampled batch.
        """
        batch_size = prior_kwargs["batch_size"]
        xt = self.prior.sample_prior(**prior_kwargs)
        stds = self.prior.fit_prior(**prior_kwargs)
        time_pairs = self.get_adjacent_times(self.DP.times)

        for t, t_next in time_pairs:
            t = torch.Tensor.repeat(t, batch_size)
            t_next = torch.Tensor.repeat(t_next, batch_size)
            xt_next = self.denoise_step(xt, t, t_next, control=stds[:, -1, :, :] ** 2)
            xt = xt_next
        return xt
