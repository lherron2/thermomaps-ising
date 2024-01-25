import numpy as np
from abc import ABC, abstractmethod
from data.observables import Observable

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Energy(Observable):
    """
    Class for calculating the energy of a lattice.

    Energy is defined as the sum of the interaction energies between each spin and its neighbors.
    This class inherits from the Observable class.

    Attributes:
        name (str): The name of the observable, "energy".
        Jx (float): The interaction energy along the x direction.
        Jy (float): The interaction energy along the y direction.
    """

    def __init__(self, Jx: float = 1.0, Jy: float = 1.0, name: str = 'energy'):
        """
        Initialize an Energy observable.

        Args:
            Jx (float, optional): The interaction energy along the x direction. Defaults to 1.0.
            Jy (float, optional): The interaction energy along the y direction. Defaults to 1.0.
        """
        super().__init__("energy")
        self.Jx = Jx  # Interaction energy along the x direction
        self.Jy = Jy  # Interaction energy along the y direction

    def evaluate_frame(self, frame: np.ndarray) -> float:
        energy = 0.0
        size = len(frame)
        # Iterate over each spin in the lattice
        for i in range(size):
            for j in range(size):
                S = frame[i, j]
                # Calculate the sum of the spins of the neighbors
                neighbors_x = frame[(i+1)%size, j] + frame[(i-1)%size, j]
                neighbors_y = frame[i, (j+1)%size] + frame[i, (j-1)%size]
                # Add the interaction energy with the neighbors to the total energy
                energy += -self.Jx * S * neighbors_x - self.Jy * S * neighbors_y
        # Return the total energy, divided by 2 because each pair is counted twice
        return energy / 2

    def evaluate(self, time_series: np.ndarray) -> float:
        """
        Calculate the energy of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The energy of the lattice.
        """
        energies = []
        for frame in time_series:
            energies.append(self.evaluate_frame(frame))
        self.quantity = np.array(energies)

        return self.quantity

class Magnetization(Observable):
    """
    Class for calculating the magnetization of a lattice.

    Magnetization is defined as the mean value of the spins in the lattice.
    This class inherits from the Observable class.

    Attributes:
        name (str): The name of the observable, "magnetization".
    """

    def __init__(self):
        """
        Initialize a Magnetization observable.
        """
        super().__init__("magnetization")

    def evaluate_frame(self, frame: np.ndarray) -> float:
        """
        Calculate the magnetization of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The magnetization of the lattice.
        """
        # Magnetization is the mean value of the spins
        return abs(np.mean(frame))

    def evaluate(self, time_series: np.ndarray) -> float:
        """
        Calculate the magnetization of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The magnetization of the lattice.
        """
        magnetizations = []
        for frame in time_series:
            magnetizations.append(self.evaluate_frame(frame))
        self.quantity = np.array(magnetizations)

        return self.quantity
        