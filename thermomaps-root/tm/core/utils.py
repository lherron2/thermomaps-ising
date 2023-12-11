import yaml
import torch.nn.functional as F
import numpy as np
import os
import re


def exists(x):
    """
    Check if a variable exists (is not None).

    Args:
        x: Any variable.

    Returns:
        bool: True if the variable is not None, False otherwise.
    """
    return x is not None


def default(val, d):
    """
    Return the value if it exists, otherwise return a default value or result.

    Args:
        val: Any variable.
        d: Default value or function to call if val does not exist.

    Returns:
        Any: val if it exists, otherwise d.
    """
    if exists(val):
        return val
    # Uncomment the following line if d is a function (is_lambda(d))
    # return d() if is_lambda(d) else d


def compute_model_dim(data_dim, groups):
    """
    Compute the model dimension based on data dimension and groups.

    Args:
        data_dim (int): Dimension of the data.
        groups (int): Number of groups.

    Returns:
        int: Model dimension.
    """
    return int(np.ceil(data_dim / groups) * groups)


class Interpolater:
    """
    Reshapes irregularly (or unconventionally) shaped data to be compatible with a model.
    """

    def __init__(self, data_shape: tuple, target_shape: tuple):
        """
        Initialize the Interpolater with data and target shapes.

        Args:
            data_shape (tuple): Shape of the original data.
            target_shape (tuple): Target shape for interpolation.
        """
        self.data_shape, self.target_shape = data_shape, target_shape

    def to_target(self, x):
        """
        Interpolate data to the target shape.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Interpolated data.
        """
        return F.interpolate(x, size=self.target_shape, mode="nearest-exact")

    def from_target(self, x):
        """
        Interpolate data from the target shape to the original shape.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Interpolated data.
        """
        return F.interpolate(x, size=self.data_shape, mode="nearest-exact")
