
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Type
from data.utils import ArrayWrapper

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Observable(ABC):
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def evaluate(self, trajectory: Type['Trajectory']):
        raise NotImplementedError

    def set(self, quantity: np.ndarray):
        self.quantity = quantity

    def as_vector(self):
        if len(self.quantity.shape) > 1:
            return ArrayWrapper(self.quantity.reshape(self.quantity.shape[0], -1))
        else:
            raise ValueError("Quantity cannot be reshaped into a 2D array.")

    def as_tensor(self):
        return ArrayWrapper(self.quantity)

    def __getitem__(self, index: Union[int, slice]) -> 'Observable':
        """
        Create a new Observable instance with a subset of the quantity.

        Args:
            index (int or slice): The index or slice to retrieve.

        Returns:
            Observable: A new Observable instance with the sliced quantity.
        """
        if hasattr(self.quantity, '__getitem__'):
            new_obs = copy.copy(self)
            new_obs.set(self.quantity[index])
            return new_obs
        else:
            raise TypeError("Quantity does not support indexing or slicing.")

    def __add__(self, other: 'Observable') -> 'Observable':
        """
        Add two Observable instances together.

        Args:
            other (Observable): The other Observable instance to add.

        Returns:
            Observable: A new Observable instance with the summed quantity.
        """
        try:
            new_quantity = self.quantity + other.quantity
            return type(self)(name=self.name, quantity=new_quantity)
        except TypeError:
            raise TypeError("Quantity cannot be added.")

    def __sub__(self, other: 'Observable') -> 'Observable': 
        """
        Subtract two Observable instances.

        Args:
            other (Observable): The other Observable instance to subtract.

        Returns:
            Observable: A new Observable instance with the subtracted quantity.
        """
        try:
            new_quantity = self.quantity - other.quantity
            return type(self)(name=self.name, quantity=new_quantity)
        except TypeError:
            raise TypeError("Quantity cannot be subtracted.") 
        
    def __mul__(self, other: 'Observable') -> 'Observable':
        """
        Multiply two Observable instances together.

        Args:
            other (Observable): The other Observable instance to multiply.

        Returns:
            Observable: A new Observable instance with the multiplied quantity.
        """
        try:
            new_quantity = self.quantity * other.quantity
            return type(self)(name=self.name, quantity=new_quantity)
        except TypeError:
            raise TypeError("Quantity cannot be multiplied.")
        
    def __truediv__(self, other: 'Observable') -> 'Observable':
        """
        Divide two Observable instances.

        Args:
            other (Observable): The other Observable instance to divide.

        Returns:
            Observable: A new Observable instance with the divided quantity.
        """
        try:
            new_quantity = self.quantity / other.quantity
            return type(self)(name=self.name, quantity=new_quantity)
        except TypeError:
            raise TypeError("Quantity cannot be divided.")
        
    def __listadd__(self, other: 'Observable') -> 'Observable':
        """
        Add two Observable instances together.

        Args:
            other (Observable): The other Observable instance to add.

        Returns:
            Observable: A new Observable instance with the summed quantity.
        """
        try:
            new_quantity = list(self.quantity) + list(other.quantity)
            type(self)(name=self.name, quantity=new_quantity)
        except TypeError:
            raise TypeError("Quantity cannot be converted to a list or added.") 

