import torch
import numpy as np
from typing import Any, Callable, Dict, List
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UnitNormalPrior:
    def __init__(self, shape):
        """Initialize the Unit Normal Prior with the shape of the samples."""
        self.shape = list(shape)[1:]
        logging.debug(f"Initialized a UnitNormalPrior with shape {self.shape}.")
        logging.debug(f"The first dimension of the supplied {shape=} must be the batch size.")

    def sample(self, batch_size, *args, **kwargs):
        """Sample from a unit normal distribution."""
        shape = [batch_size] + self.shape
        logging.debug(f"Sampling from a UnitNormalPrior with shape {shape}")
        return torch.normal(mean=0, std=1, size=shape)

class GlobalEquilibriumHarmonicPrior(UnitNormalPrior):
    def __init__(self, shape, channels_info):
        """Initialize GEHP with shape and channels information."""
        super().__init__(shape)
        self.channels_info = channels_info  # Dictionary to define channel types

    def sample(self, batch_size, temperatures, *args, **kwargs):
        """Sample from a distribution where variance is defined by temperatures."""
        shape = [batch_size] + self.shape
        logging.debug(f"Sampling from a GlobalEquilibriumHarmonicPrior with shape {shape}")
        samples = torch.empty(shape)

        # Check if temperatures is a vector
        if isinstance(temperatures, torch.Tensor):
            if temperatures.shape[0] != shape[0]:
                raise ValueError("Length of temperatures vector must be equal to self.shape[0]")
            
        # The length of the first dimension of temperatures must be equal to the batch size
        # The length of the second dimension of temperatures must be equal to the number of coordinate channels
        # If temperatures is a scalar, then reshape it to a matrix of shape (batch_size, number of coordinate channels)
        # If temperatures is a vector and the length of the first dimension is one and the length of the second dimension is equal to the number of coordinate channels, then reshape it to a matrix of shape (batch_size, number of coordinate channels)
        
        # Sampling for different channel types
        for channel_type, channels in self.channels_info.items():
            if channel_type == "coordinate":
                if isinstance(temperatures, torch.Tensor): 
                    variance = temperatures 
                else:
                    variance = torch.full((shape[0],), temperatures)
                logging.debug(f"Channels {tuple(channels)} are coordinate channels with variance {variance}")
            elif channel_type == "fluctuation":
                variance = torch.full((shape[0],), 1)  # Unit variance for fluctuation channels
                logging.debug(f"Channels {tuple(channels)} are fluctuation channels with variance {variance}")
            else:
                raise ValueError("Unknown channel type")

            for channel in channels:
                for i in range(shape[0]):
                    samples[i, channel] = torch.normal(mean=0, std=np.sqrt(variance[i]), size=(1,))

        return samples
