import numpy as np
import random
import sys
from typing import Callable, Dict, List, Type
from abc import ABC, abstractmethod
from data.trajectory import EnsembleIsingTrajectory
from data.generic import Summary
from ising.observables import Energy, Magnetization
from slurmflow.serializer import ObjectSerializer


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class IsingModel:
    def __init__(self, sampler: Type['Sampler'], size: int, warmup: int, temp: float, Jx: float = 1.0, Jy: float = 1.0):
        """
        Initialize the base Ising model.

        Args:
            size (int): The size of the lattice.
            warmup (int): The number of warm-up steps.
            temp (float): The temperature of the system.
            Jx (float, optional): The interaction energy along the x direction. Defaults to 1.0.
            Jy (float, optional): The interaction energy along the y direction. Defaults to 1.0.
        """
        self.size = size
        self.lattice = np.random.choice([-1, 1], (size, size))
        self.warmup = warmup
        self.temp = temp
        self.Jx = Jx
        self.Jy = Jy
        self.snapshots = []
        sampler = sampler(self)
        self.update = sampler.update
        self.trajectory = EnsembleIsingTrajectory(
            summary = Summary(name="IsingModel", size=size, temperature=temp, Jx=Jx, Jy=Jy, sampler=sampler.name),
            state_variables = Summary(temperature=temp)
            )

    def simulate(self, steps: int, observables: List[Callable[[np.ndarray], float]], sampling_frequency: int):
        """
        Simulate the Ising model.

        Args:
            steps (int): The number of steps to perform.
            observables (List[Callable[[np.ndarray], float]]): A list of observables to measure.
            sampling_frequency (int): The frequency at which to sample the state of the system and the observables.
        """
        # Perform the warm-up steps
        for _ in range(self.warmup):
            self.lattice = self.update()

        logger.debug(f"Finished warm-up steps.")

        # Perform the simulation steps
        for i in range(steps):
            self.lattice = self.update()

            # Sample the state of the system and the observables
            if i % sampling_frequency == 0:
                logger.debug(f"Adding frame {i} (shape = {self.lattice.shape}) to the trajectory.")
                self.trajectory.add_frame(self.lattice)

        # Compute the observables and add them to the trajectory
        for obs in observables:
            logger.debug(f"Computing observable {obs.name}.")
            obs.evaluate(self.trajectory.coordinates)
            self.trajectory.add_observable(obs)

        return self.trajectory

    def save(self, filename: str, overwrite: bool = False):
        """
        Save the trajectory to a file.

        Args:
            filename (str): The name of the file to save the trajectory to.
        """
        OS = ObjectSerializer(filename)
        OS.serialize(self.trajectory, overwrite=overwrite)