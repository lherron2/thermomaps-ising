import numpy as np
from abc import ABC, abstractmethod

class Observable(ABC):
    """
    Abstract base class for observables in the Ising model.

    An observable is a physical quantity that can be measured in a simulation.

    Attributes:
        name (str): The name of the observable.
    """

    def __init__(self, name: str):
        """
        Initialize an Observable.

        Args:
            name (str): The name of the observable.
        """
        # Name of the observable
        self.name = name

    @abstractmethod
    def evaluate(self, lattice: np.ndarray) -> float:
        """
        Calculate the value of the observable for a given lattice.

        This is an abstract method that must be overridden by subclasses.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The value of the observable.
        """
        pass

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

    def __init__(self, Jx: float = 1.0, Jy: float = 1.0):
        """
        Initialize an Energy observable.

        Args:
            Jx (float, optional): The interaction energy along the x direction. Defaults to 1.0.
            Jy (float, optional): The interaction energy along the y direction. Defaults to 1.0.
        """
        super().__init__("energy")
        self.Jx = Jx  # Interaction energy along the x direction
        self.Jy = Jy  # Interaction energy along the y direction

    def evaluate(self, lattice: np.ndarray) -> float:
        """
        Calculate the energy of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The energy of the lattice.
        """
        energy = 0.0
        size = len(lattice)
        # Iterate over each spin in the lattice
        for i in range(size):
            for j in range(size):
                S = lattice[i, j]
                # Calculate the sum of the spins of the neighbors
                neighbors_x = lattice[(i+1)%size, j] + lattice[(i-1)%size, j]
                neighbors_y = lattice[i, (j+1)%size] + lattice[i, (j-1)%size]
                # Add the interaction energy with the neighbors to the total energy
                energy += -self.Jx * S * neighbors_x - self.Jy * S * neighbors_y
        # Return the total energy, divided by 2 because each pair is counted twice
        return energy / 2

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

    def evaluate(self, lattice: np.ndarray) -> float:
        """
        Calculate the magnetization of a lattice.

        Args:
            lattice (np.ndarray): The lattice of spins.

        Returns:
            float: The magnetization of the lattice.
        """
        # Magnetization is the mean value of the spins
        return np.mean(lattice.ravel())