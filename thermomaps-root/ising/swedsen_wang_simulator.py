import numpy as np
import random
from typing import Callable, Dict, List
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


class IsingSwendsenWang:
    """
    Class for simulating the Ising model using the Swendsen-Wang algorithm.

    Attributes:
        size (int): The size of the lattice.
        lattice (np.ndarray): The lattice of spins.
        warm_up (int): The number of warm-up steps.
        temp (float): The temperature of the system.
        snapshots (list): A list to store snapshots of the system state.
    """

    def __init__(self, size: int, warm_up: int, temp: float, Jx: float = 1.0, Jy: float = 1.0):
        """
        Initialize an IsingSwendsenWang simulator.

        Args:
            size (int): The size of the lattice.
            warm_up (int): The number of warm-up steps.
            temp (float): The temperature of the system.
            Jx (float, optional): The interaction energy along the x direction. Defaults to 1.0.
            Jy (float, optional): The interaction energy along the y direction. Defaults to 1.0.
        """
        # Size of the lattice
        self.size = size

        # Initialize the lattice with random spins
        self.lattice = np.random.choice([-1, 1], (size, size))

        # Number of warm-up steps
        self.warm_up = warm_up

        # Temperature of the system
        self.temp = temp

        # Interaction energy along the x direction
        self.Jx = Jx

        # Interaction energy along the y direction
        self.Jy = Jy

        # List to store snapshots of the system state
        self.snapshots = []

    def initialize_clusters(self) -> Dict[int, int]:
        """
        Initialize the clusters for the Swendsen-Wang algorithm.

        This function creates a new cluster for each site in the lattice. Each site is its own root.

        Returns:
            Dict[int, int]: A dictionary mapping each site to its cluster label.
        """
        # Create a new cluster for each site in the lattice
        # Each site is its own root
        return {i: i for i in range(self.size * self.size)}

    def find_root(self, site: int, labels: Dict[int, int]) -> int:
        """
        Find the root of the cluster that a site belongs to.

        This function is part of the Swendsen-Wang algorithm for the Ising model.
        It uses path compression to speed up future lookups.

        Args:
            site (int): The linear index of the site.
            labels (Dict[int, int]): A dictionary mapping each site to its cluster label.

        Returns:
            int: The root of the cluster that the site belongs to.
        """
        # Find the root of the cluster by following the parent pointers
        root = site
        while root != labels[root]:
            root = labels[root]

        # Compress the path from the site to the root
        while site != root:
            parent = labels[site]
            labels[site] = root
            site = parent

        # Return the root of the cluster
        return root

    def union(self, site1: int, site2: int, labels: Dict[int, int]):
        """
        Merge the clusters of two sites.

        This function is part of the Swendsen-Wang algorithm for the Ising model.
        It merges the clusters of site1 and site2 by setting the root of site2's cluster
        to be the root of site1's cluster.

        Args:
            site1 (int): The linear index of the first site.
            site2 (int): The linear index of the second site.
            labels (Dict[int, int]): A dictionary mapping each site to its cluster label.
        """
        # Find the roots of the clusters of site1 and site2
        root1, root2 = self.find_root(site1, labels), self.find_root(site2, labels)

        # If the sites are in different clusters, merge the clusters
        if root1 != root2:
            # Set the root of site2's cluster to be the root of site1's cluster
            labels[root2] = root1

    def build_clusters(self, labels: Dict[int, int]):
        """
        Build clusters of connected spins in the same state.

        This function implements part of the Swendsen-Wang algorithm for the Ising model.
        It iterates over each spin in the lattice and, with a certain probability, creates bonds
        between neighboring spins that are in the same state.

        Args:
            labels (Dict[int, int]): A dictionary mapping each spin to its cluster label.
        """
        # Calculate the probability of creating a bond between two neighboring spins
        px = 1 - np.exp(-2 * self.Jx / self.temp)
        py = 1 - np.exp(-2 * self.Jy / self.temp)

        # Iterate over each spin in the lattice
        for x in range(self.size):
            for y in range(self.size):
                # With probability px, create bonds between neighboring spins in the same state along x direction
                if random.random() < px:
                    # Check the spin to the right
                    if x + 1 < self.size and self.lattice[x, y] == self.lattice[x + 1, y]:
                        # If the spins are in the same state, create a bond between them
                        self.union(x * self.size + y, (x + 1) * self.size + y, labels)

                # With probability py, create bonds between neighboring spins in the same state along y direction
                if random.random() < py:
                    # Check the spin below
                    if y + 1 < self.size and self.lattice[x, y] == self.lattice[x, y + 1]:
                        # If the spins are in the same state, create a bond between them
                        self.union(x * self.size + y, x * self.size + (y + 1), labels)

    def flip_clusters(self, labels: Dict[int, int]):
        """
        Flip clusters of connected spins.

        This function implements part of the Swendsen-Wang algorithm for the Ising model.
        It decides randomly whether to flip each cluster of connected spins or not, and then
        flips the spins in the selected clusters.

        Args:
            labels (Dict[int, int]): A dictionary mapping each spin to its cluster label.
        """
        # Decide randomly whether to flip each cluster or not
        should_flip = {root: random.choice([True, False]) for root in set(labels.values())}

        # Iterate over each spin in the lattice
        for x in range(self.size):
            for y in range(self.size):
                # Find the root of the cluster that the spin belongs to
                root = self.find_root(x * self.size + y, labels)
                # If the cluster should be flipped, flip the spin
                if should_flip[root]:
                    self.lattice[x, y] *= -1

    def swendsen_wang_step(self):
        """
        Perform one step of the Swendsen-Wang algorithm.

        This function implements one step of the Swendsen-Wang algorithm for the Ising model.
        It first initializes the clusters, then builds the clusters of connected spins, and finally
        flips the clusters.
        """
        # Initialize the clusters from current lattice configuration
        labels = self.initialize_clusters()

        # Build the clusters of connected spins
        self.build_clusters(labels)

        # Flip the clusters
        self.flip_clusters(labels)

    def simulate(self, steps: int, observables: List[Callable[[np.ndarray], float]], sampling_frequency: int):
        """
        Simulate the Ising model using the Swendsen-Wang algorithm.

        This function performs a number of steps of the Swendsen-Wang algorithm, and periodically
        samples the state of the system and the values of the observables.

        Args:
            steps (int): The number of steps to perform.
            observables (List[Callable[[np.ndarray], float]]): A list of observables to measure.
            sampling_frequency (int): The frequency at which to sample the state of the system and the observables.

        Returns:
            dict: A dictionary containing the sampled states of the system and the values of the observables.
        """
        # Perform the warm-up steps
        for _ in range(self.warm_up):
            self.swendsen_wang_step()

        # Perform the simulation steps
        for i in range(steps):
            self.swendsen_wang_step()

            # Sample the state of the system and the observables
            if i % sampling_frequency == 0:
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

        # Convert lists to numpy arrays
        for key in self.snapshots:
            self.snapshots[key] = np.array(self.snapshots[key])

        # Return the sampled states and observables
        return self.snapshots

    def save_snapshots(self, filename: str, metadata: dict = None):
        """
        Save the snapshots of the system state and the observables to a file.

        This function saves the snapshots of the system state and the observables to a compressed
        numpy (.npz) file. Additional metadata can also be included in the file.

        Args:
            filename (str): The name of the file to save the snapshots to.
            metadata (dict, optional): A dictionary of metadata to include in the file.
        """
        # If metadata is provided, include it in the file
        if metadata:
            np.savez_compressed(f"{filename}.npz", **self.snapshots, **metadata)
        else:
            # Otherwise, just save the snapshots
            np.savez_compressed(f"{filename}.npz", **self.snapshots)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=int, default=16)
    argparser.add_argument("--warm-up", type=int, default=1000)
    argparser.add_argument("--steps", type=int, default=10000)
    argparser.add_argument("--temp", type=float, default=2.0)
    argparser.add_argument("--Jx", type=float, default=1.0)
    argparser.add_argument("--Jy", type=float, default=1.0)
    argparser.add_argument("--sampling-frequency", type=int, default=100)
    argparser.add_argument("--filename", type=str, default="ising")
    args = argparser.parse_args()

    ising = IsingSwendsenWang(args.size, args.warm_up, args.temp, args.Jx, args.Jy)
    results = ising.simulate(args.steps, [Energy(args.Jx, args.Jy), Magnetization()], args.sampling_frequency)
    ising.save_snapshots(args.filename)