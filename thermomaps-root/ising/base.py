import numpy as np
import random
from typing import Callable, Dict, List, Type
from abc import ABC, abstractmethod

class IsingModel:
    def __init__(self, sampler: Type['Sampler'], size: int, warm_up: int, temp: float, Jx: float = 1.0, Jy: float = 1.0):
        """
        Initialize the base Ising model.

        Args:
            size (int): The size of the lattice.
            warm_up (int): The number of warm-up steps.
            temp (float): The temperature of the system.
            Jx (float, optional): The interaction energy along the x direction. Defaults to 1.0.
            Jy (float, optional): The interaction energy along the y direction. Defaults to 1.0.
        """
        self.size = size
        self.lattice = np.random.choice([-1, 1], (size, size))
        self.warm_up = warm_up
        self.temp = temp
        self.Jx = Jx
        self.Jy = Jy
        self.snapshots = []
        sampler = sampler(self)
        self.update = sampler.update

    def simulate(self, steps: int, observables: List[Callable[[np.ndarray], float]], sampling_frequency: int):
        """
        Simulate the Ising model.

        Args:
            steps (int): The number of steps to perform.
            observables (List[Callable[[np.ndarray], float]]): A list of observables to measure.
            sampling_frequency (int): The frequency at which to sample the state of the system and the observables.
        """
        # Perform the warm-up steps
        for _ in range(self.warm_up):
            self.lattice = self.update()

        # Perform the simulation steps
        for i in range(steps):
            self.lattice = self.update()

            # Sample the state of the system and the observables
            if i % sampling_frequency == 0:
                self.sample_state(observables)

        return self.snapshots

    def sample_state(self, observables: List[Callable[[np.ndarray], float]]):
        """
        Sample the state of the system and the observables.

        Args:
            observables (List[Callable[[np.ndarray], float]]): A list of observables to measure.
        """
        if not self.snapshots:
            # Initialize the dictionary with lists
            self.snapshots = {'lattice': [self.lattice.copy()]}
            for obs in observables:
                self.snapshots[obs.name] = [obs.evaluate(self.lattice)]
        else:
            # Append to the existing lists
            self.snapshots['lattice'].append(self.lattice.copy())
            for obs in observables:
                self.snapshots[obs.name].append(obs.evaluate(self.lattice))

    def save_snapshots(self, filename: str, metadata: dict = None):
        """
        Save the snapshots of the system state and the observables to a file.

        Args:
            filename (str): The name of the file to save the snapshots to.
            metadata (dict, optional): A dictionary of metadata to include in the file.
        """
        if metadata:
            np.savez_compressed(f"{filename}.npz", **self.snapshots, **metadata)
        else:
            np.savez_compressed(f"{filename}.npz", **self.snapshots)

