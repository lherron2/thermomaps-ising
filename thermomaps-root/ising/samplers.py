import numpy as np
import random
from typing import Callable, Dict, List, Type
from abc import ABC, abstractmethod
from ising.base import IsingModel

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Sampler(ABC):
    """
    Abstract base class for samplers in the Ising model.

    A sampler is a method for updating the state of the system.

    Attributes:
        name (str): The name of the sampler.
    """

    def __init__(self, ising_model: Type[IsingModel]):
        """
        Initialize a Sampler.
        """
        self.IM = ising_model

    @abstractmethod
    def update(self):
        """
        Perform one update step.

        This is an abstract method that must be overridden by subclasses.
        """
        pass

class SwendsenWangSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SwendsenWang"

    def update(self):
        """
        Perform one update step using the Swendsen-Wang algorithm.
        """
        # Initialize the clusters from the current lattice configuration
        labels = self.initialize_clusters()

        # Build the clusters of connected spins
        self.build_clusters(labels)

        # Flip the clusters
        self.flip_clusters(labels)

        return self.IM.lattice

    def initialize_clusters(self) -> Dict[int, int]:
        """
        Initialize the clusters for the Swendsen-Wang algorithm.

        Returns:
            Dict[int, int]: A dictionary mapping each site to its cluster label.
        """
        return {i: i for i in range(self.IM.size * self.IM.size)}

    def find_root(self, site: int, labels: Dict[int, int]) -> int:
        """
        Find the root of the cluster that a site belongs to.

        Args:
            site (int): The linear index of the site.
            labels (Dict[int, int]): A dictionary mapping each site to its cluster label.

        Returns:
            int: The root of the cluster that the site belongs to.
        """
        while site != labels[site]:
            site = labels[site]
        return site

    def union(self, site1: int, site2: int, labels: Dict[int, int]):
        """
        Merge the clusters of two sites.

        Args:
            site1 (int): The linear index of the first site.
            site2 (int): The linear index of the second site.
            labels (Dict[int, int]): A dictionary mapping each site to its cluster label.
        """
        root1, root2 = self.find_root(site1, labels), self.find_root(site2, labels)
        if root1 != root2:
            labels[root2] = root1

    def build_clusters(self, labels: Dict[int, int]):
        """
        Build clusters of connected spins in the same state.
        """
        px = 1 - np.exp(-2 * self.IM.Jx / self.IM.temp)
        py = 1 - np.exp(-2 * self.IM.Jy / self.IM.temp)

        for x in range(self.IM.size):
            for y in range(self.IM.size):
                if x + 1 < self.IM.size and self.IM.lattice[x, y] == self.IM.lattice[x + 1, y] and random.random() < px:
                    self.union(x * self.IM.size + y, (x + 1) * self.IM.size + y, labels)
                if y + 1 < self.IM.size and self.IM.lattice[x, y] == self.IM.lattice[x, y + 1] and random.random() < py:
                    self.union(x * self.IM.size + y, x * self.IM.size + (y + 1), labels)

    def flip_clusters(self, labels: Dict[int, int]):
        """
        Flip clusters of connected spins.
        """
        should_flip = {root: random.choice([True, False]) for root in set(labels.values())}

        for x in range(self.IM.size):
            for y in range(self.IM.size):
                root = self.find_root(x * self.IM.size + y, labels)
                if should_flip[root]:
                    self.IM.lattice[x, y] *= -1

class SingleSpinFlipSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SingleSpin"

    def update(self):
        """
        Perform one update step using the Metropolis-Hastings algorithm.
        """
        for _ in range(self.IM.size * self.IM.size):
            self.metropolis_hastings_step()

        return self.IM.lattice

    def metropolis_hastings_step(self):
        """
        Perform a single Monte-Carlo sampling step using the Metropolis-Hastings algorithm.
        """
        x, y = np.random.randint(self.IM.size), np.random.randint(self.IM.size)
        delta_E = self.calculate_energy_change(x, y)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / self.IM.temp):
            self.IM.lattice[x, y] *= -1

    def calculate_energy_change(self, x: int, y: int) -> float:
        """
        Calculate the energy change for flipping a spin at a given position in an anisotropic Ising model.

        Args:
            x (int): The x-coordinate of the spin.
            y (int): The y-coordinate of the spin.

        Returns:
            float: The energy change for flipping the spin.
        """
        S = self.IM.lattice[x, y]
        
        # Neighbors along x direction
        neighbors_x = self.IM.lattice[(x + 1) % self.IM.size, y] + self.IM.lattice[(x - 1) % self.IM.size, y]
        
        # Neighbors along y direction
        neighbors_y = self.IM.lattice[x, (y + 1) % self.IM.size] + self.IM.lattice[x, (y - 1) % self.IM.size]

        # Calculate energy change with different interaction energies along x and y directions
        delta_E = 2 * S * (self.IM.Jx * neighbors_x + self.IM.Jy * neighbors_y)
        
        return delta_E

