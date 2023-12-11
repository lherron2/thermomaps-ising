import numpy as np
import random
from typing import Callable, Dict, List

class Energy:
    def __init__(self, J: float = 1.0):
        self.J = J  # Interaction energy

    def evaluate(self, lattice: np.ndarray) -> float:
        energy = 0.0
        size = len(lattice)
        for i in range(size):
            for j in range(size):
                S = lattice[i, j]
                neighbors = lattice[(i+1)%size, j] + lattice[i, (j+1)%size] + lattice[(i-1)%size, j] + lattice[i, (j-1)%size]
                energy += -self.J * S * neighbors
        return energy / 2  # Each pair counted twice

class Magnetization:
    def evaluate(self, lattice: np.ndarray) -> float:
        return np.mean(lattice.ravel())


class IsingSwendsenWang:
    def __init__(self, size: int, warm_up: int, temp: float):
        self.size = size
        self.lattice = np.random.choice([-1, 1], (size, size))
        self.warm_up = warm_up
        self.temp = temp  # Temperature of the system
        self.snapshots = []

    def initialize_clusters(self) -> Dict[int, int]:
        return {i: i for i in range(self.size * self.size)}

    def find_root(self, site: int, labels: Dict[int, int]) -> int:
        root = site
        while root != labels[root]:
            root = labels[root]
        while site != root:
            parent = labels[site]
            labels[site] = root
            site = parent
        return root

    def union(self, site1: int, site2: int, labels: Dict[int, int]):
        root1, root2 = self.find_root(site1, labels), self.find_root(site2, labels)
        if root1 != root2:
            labels[root2] = root1

    def build_clusters(self, labels: Dict[int, int]):
        p = 1 - np.exp(-2 / self.temp)
        for x in range(self.size):
            for y in range(self.size):
                if random.random() < p:
                    if x + 1 < self.size and self.lattice[x, y] == self.lattice[x + 1, y]:
                        self.union(x * self.size + y, (x + 1) * self.size + y, labels)
                    if y + 1 < self.size and self.lattice[x, y] == self.lattice[x, y + 1]:
                        self.union(x * self.size + y, x * self.size + (y + 1), labels)

    def flip_clusters(self, labels: Dict[int, int]):
        should_flip = {root: random.choice([True, False]) for root in set(labels.values())}
        for x in range(self.size):
            for y in range(self.size):
                root = self.find_root(x * self.size + y, labels)
                if should_flip[root]:
                    self.lattice[x, y] *= -1

    def swendsen_wang_step(self):
        labels = self.initialize_clusters()
        self.build_clusters(labels)
        self.flip_clusters(labels)

    # Method to simulate the lattice
    def simulate(self, steps: int, observables: Dict[str, Callable[[np.ndarray], float]], sampling_frequency: int):
        for _ in range(self.warm_up):  # Warm-up period
            self.swendsen_wang_step()

        for i in range(steps):
            self.swendsen_wang_step()
            if i % sampling_frequency == 0:
                snapshot = {
                    'lattice': self.lattice.copy(),
                    'observables': {name: obs.evaluate(self.lattice) for name, obs in observables.items()}
                }
                self.snapshots.append(snapshot)

    def save_snapshots(self, filename: str, metadata: dict, observables_to_save: list):
        snapshot_series_dict = {key: [d[key] for d in self.snapshots] for key in self.snapshots[0]}
        for i, snapshot in enumerate(self.snapshots):
            np.savez_compressed(f"{filename}_snapshot_{i}.npz", **snapshot_series_dict)

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=int, default=16)
    argparser.add_argument("--warm-up", type=int, default=1000)
    argparser.add_argument("--steps", type=int, default=10000)
    argparser.add_argument("--temp", type=float, default=2.0)
    argparser.add_argument("--sampling-frequency", type=int, default=100)
    argparser.add_argument("--filename", type=str, default="ising")
    args = argparser.parse_args()

    ising = IsingSwendsenWang(args.size, args.warm_up, args.temp)
    ising.simulate(args.steps, {"energy": Energy(), "magnetization": Magnetization()}, args.sampling_frequency)

    ising.save_snapshots(args.filename, {"size": args.size, "warm_up": args.warm_up, "temp": args.temp},
                            ["energy", "magnetization"])
