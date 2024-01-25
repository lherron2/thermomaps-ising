import torch.nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
import os
from tm.core.utils import Interpolater, default

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Backbone(nn.Module):
    """
    Diffusion wrapper for instances of deep learning architectures.

    Args:
        model: The deep learning model to wrap.
        data_shape: The shape of the input data.
        target_shape: The shape of the target data.
        num_dims: Number of dimensions.
        lr: Learning rate for optimization.
        optim: Optimization method (default is Adam).
        scheduler: Learning rate scheduler (default is MultiStepLR).

    Attributes:
        device (str): The device used for computation (CPU or CUDA).
        model: The deep learning model.
        interp: Interpolater for data shapes.
        expand_batch_to_dims: Tuple for expanding batch dimensions.
        state: Internal state of the backbone.
        start_epoch: Starting epoch for training.
        optim: Optimization method.
        scheduler: Learning rate scheduler.

    Methods:
        get_model_path(directory, epoch): Get the model path for saving.
        save_state(directory, epoch): Save internal state of the backbone.
        load_state(directory, epoch): Load internal state of the backbone.
        load_model(directory, epoch): Load model, optimizer, and starting epoch from state dict.
    """

    def __init__(
        self,
        model,
        data_shape,
        target_shape,
        num_dims=3,
        lr=1e-3,
        optim=None,
        scheduler=None,
        interpolate=True,
    ):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        data_shape = tuple(
            [data_shape] * (num_dims - 2)
        )  # Ignore batch and channel dims
        target_shape = tuple([target_shape] * (num_dims - 2))

        self.interpolate = interpolate
        if self.interpolate:
            self.interp = Interpolater(data_shape, target_shape)
        dim_vec = torch.ones(num_dims)
        dim_vec[0] = -1
        self.expand_batch_to_dims = tuple(dim_vec)
        self.state = None
        self.start_epoch = 0

        optim_dict = {
            "Adam": torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=False,
                betas=(0.9, 0.999),
                amsgrad=True,
                eps=1e-9,
            )
        }

        self.optim = default(optim_dict["Adam"], optim)

        scheduler_dict = {
            "multistep": MultiStepLR(self.optim, milestones=[300], gamma=0.1),
        }

        self.scheduler = default(scheduler_dict["multistep"], scheduler)

    @staticmethod
    def get_model_path(model_dir, epoch, identifier='model'):
        """Get the model path for saving."""
        return os.path.join(model_dir, f"{identifier}_{epoch}.pt")

    def save_state(self, model_dir, epoch, identifier='model'):
        """
        Save internal state of the backbone model.

        Args:
            directory: Directory where the state is saved.
            epoch: Current training epoch.

        Returns:
            None
        """
        states = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": epoch,
        }
        os.makedirs(model_dir, exist_ok=True)
        save_path = self.get_model_path(model_dir, epoch, identifier=identifier)
        torch.save(states, save_path)

    def load_state(self, model_dir, epoch, identifier='model', device=0):
        """
        Load internal state of the backbone model.

        Args:
            directory: Directory where the state is saved.
            epoch: Current training epoch.

        Returns:
            state_dict: Loaded state dictionary.
        """
        state_dict = torch.load(
            self.get_model_path(model_dir, epoch, identifier=identifier),
            map_location=torch.device(device),
        )
        return state_dict

    def load_model(self, model_dir, epoch, identifier='model', device=0):
        """
        Load model, optimizer, and starting epoch from state dict.

        Args:
            directory: Directory where the state is saved.
            epoch: Current training epoch.

        Returns:
            None
        """
        state_dict = self.load_state(model_dir, epoch, identifier=identifier, device=device)
        self.model.load_state_dict(state_dict["model"])
        self.optim.load_state_dict(state_dict["optim"])
        self.start_epoch = int(state_dict["epoch"]) + 1


class ConvBackbone(Backbone):
    """
    Backbone with a forward method for Convolutional Networks.

    Args:
        model: The deep learning model to wrap.
        data_shape: The shape of the input data.
        target_shape: The shape of the target data.
        num_dims: Number of dimensions.
        lr: Learning rate for optimization.
        optim: Optimization method (default is Adam).
        eval_mode: Evaluation mode ('train' or 'sample').
        self_condition: Whether to condition on itself during training (True or False).
        scheduler: Learning rate scheduler (default is MultiStepLR).

    Attributes:
        eval_mode (str): Evaluation mode ('train' or 'sample').
        self_condition (bool): Whether to condition on itself during training (True or False).

    Methods:
        get_self_condition(data, t): Get self-conditioning for training or sampling.
        forward(batch, t): Forward pass of the ConvBackbone.
    """

    def __init__(
        self,
        model,
        data_shape,
        target_shape,
        num_dims=4,
        lr=1e-3,
        optim=None,
        eval_mode="train",
        self_condition=False,
        scheduler=None,
        interpolate=True
    ):
        super().__init__(
            model, data_shape, target_shape, num_dims, lr, optim, scheduler
        )

        self.eval_mode = eval_mode
        self.self_condition = self_condition

    def get_self_condition(self, data, t):
        """Get self-conditioning for training or sampling."""
        if self.eval_mode == "train" and self.self_condition == True:
            if torch.rand(1) < 0.5:
                with torch.no_grad():
                    return self.model(data.to(self.device), t.to(self.device))
            else:
                return None
        elif self.eval_mode == "sample" and self.self_condition == True:
            return self.model(data.to(self.device), t.to(self.device))
        else:
            return None

    def forward(self, batch, t):
        """Forward pass of the ConvBackbone."""
        if self.interpolate:
            upsampled = self.interp.to_target(batch)
        else:
            upsampled = batch

        self_condition = self.get_self_condition(upsampled, t)
        upsampled_out = self.model(
            upsampled.to(self.device), t.to(self.device), x_self_cond=self_condition
        )
        if self.interpolate:
            batch_out = self.interp.from_target(upsampled_out.to("cpu"))
        else:
            batch_out = upsampled_out.to("cpu")

        return batch_out


class GraphBackbone(Backbone):
    """
    Backbone with a forward method for Convolutional Networks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def forward(self, batch, t):
        """Forward pass of the GraphBackbone."""
        pass
