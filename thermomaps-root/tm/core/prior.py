import torch
import numpy as np
from typing import Any, Callable, Dict, List
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class UnitNormalPrior:
    def __init__(self, shape, channels_info):
        self.channels_info = channels_info  # Dictionary to define channel types
        self.num_fluct_ch = len(self.channels_info['fluctuation'])
        self.num_coord_ch = len(self.channels_info['coordinate'])
        """Initialize the Unit Normal Prior with the shape of the samples."""
        self.shape = list(shape)[1:]
        logger.debug(f"Initialized a Prior with shape {self.shape}.")
        logger.debug(f"The first dimension of the supplied {shape=} must be the batch size.")

    def sample(self, batch_size, *args, **kwargs):
        """Sample from a unit normal distribution."""
        shape = [batch_size] + self.shape
        # print(f"Sampling from a UnitNormalPrior with shape {shape}")
        return torch.normal(mean=0, std=1, size=shape)

class GlobalEquilibriumHarmonicPrior(UnitNormalPrior):
    def __init__(self, shape, channels_info):
        """Initialize GEHP with shape and channels information."""
        super().__init__(shape, channels_info)
        # self.channels_info = channels_info  # Dictionary to define channel types
        # self.num_fluct_ch = len(self.channels_info['fluctuation'])
        # self.num_coord_ch = len(self.channels_info['coordinate'])

    def sample(self, batch_size, temperatures, *args, **kwargs):
        """Sample from a distribution where variance is defined by temperatures."""
        logger.debug(f"{temperatures=}")
        temperatures = torch.Tensor(np.array(temperatures))
        full_shape = [batch_size] + self.shape
        coord_shape = [batch_size] + [self.num_coord_ch] + self.shape[1:]
        fluct_shape = [batch_size] + [self.num_fluct_ch] + self.shape[1:]
        logger.debug(f"{full_shape=}")
        logger.debug(f"{coord_shape=}")
        logger.debug(f"{fluct_shape=}")
        logger.debug(f"{temperatures.shape=}")
        samples = torch.empty(full_shape)

        assert (temperatures.shape[1] == self.num_coord_ch and 
                (temperatures.shape[0] == 1 or temperatures.shape[0] == batch_size)), \
        f"{temperatures.shape=}. Expected (1,{self.num_coord_ch}) or ({batch_size}, {self.num_coord_ch})"
        
        coord_variances = temperatures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape) # expand along batch and coordinate dims
        fluct_variances = torch.full((fluct_shape), 1)

        variances = torch.cat((coord_variances, fluct_variances), dim=1)

        for sample_idx, ch_variances in enumerate(variances):
            logging.debug(f"{ch_variances.shape}")
            samples[sample_idx] = torch.normal(mean=0., std=np.sqrt(ch_variances))
        logger.debug(f"{samples.shape}")
        return samples
