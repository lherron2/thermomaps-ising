import os
import numpy as np
from data.observables import Observable
from data.generic import DataFormat, Summary
from typing import List, Optional, Dict, Union, Iterable
import collections

import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Trajectory(DataFormat):
    def __init__(self, summary: Summary, coordinates: np.ndarray = None):
        """
        Initialize a Trajectory object.

        Args:
            summary (Summary): The summary of the trajectory.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(summary)
        self.summary = summary
        self.observables = {}
        self.coordinates = coordinates


    def add_observable(self, observables: Union[Observable, List[Observable]]):
        """
        Add one or more observables to the trajectory.

        Args:
            observables (Union[Observable, List[Observable]]): The observable or list of observables to add.
        """
        if not isinstance(observables, list):
            observables = [observables]

        for observable in observables:
            self.observables[observable.name] = observable

    def __getitem__(self, index: Union[int, slice, Iterable[int]]):
        """
        Get a specific frame or a slice of frames from the trajectory.

        Args:
            index (int or slice): The index or slice to retrieve.

        Returns:
            Trajectory: A new Trajectory object with the specific frame or slice of frames and the corresponding observables.

        Raises:
            IndexError: If the index is out of range.
        """

        if self.coordinates is None:
            frame = None
        else:
            frame = self.coordinates[index]

        observables = {name: obs[index] for name, obs in self.observables.items()}

        # Create a new Trajectory object with the specific frame and observables
        new_trajectory = Trajectory(self.summary, frame)
        for name, value in observables.items():
            new_trajectory.add_observable(value)

        return new_trajectory
    
    @classmethod
    def merge(cls, summary: Summary, trajectories: List['Trajectory'], frame_indices: Optional[List[List[int]]] = None) -> 'Trajectory':
        """
        Merge multiple Trajectory objects into a single Trajectory.

        Args:
            summary (Summary): The summary of the merged trajectory.
            trajectories (List[Trajectory]): The list of Trajectory objects to merge.
            frame_indices (Optional[List[List[int]]]): The list of frame indices to include from each trajectory. 
                If None, all frames from each trajectory are included.

        Returns:
            Trajectory: The merged Trajectory object.

        Raises:
            ValueError: If the number of trajectories does not match the number of frame index lists.
        """
        if frame_indices is not None and len(trajectories) != len(frame_indices):
            raise ValueError("The number of trajectories must match the number of frame index lists.")

        if frame_indices is None:
            frame_indices = [list(range(len(traj))) for traj in trajectories]

        # Concatenate frames from each trajectory
        merged_frames = [traj[i] for traj, indices in zip(trajectories, frame_indices) for i in indices]
        merged_frames = np.concatenate(merged_frames)

        # Initialize the merged trajectory
        merged_trajectory = cls(summary, merged_frames)

        # Merge observables
        merged_observables = {}
        for traj, indices in zip(trajectories, frame_indices):
            for name, obs in traj.observables.items():
                if name not in merged_observables:
                    merged_observables[name] = obs[indices]
                else:
                    merged_observables[name] = merged_observables[name].__listadd__(obs[indices])

        # Set the merged observables to the new trajectory
        merged_trajectory.observables = merged_observables

        return merged_trajectory

    def sort_by(self, observable_name: str, reverse: bool = False):
        """
        Sort the frames in the trajectory by an observable.

        Args:
            observable_name (str): The name of the observable to sort by.
            reverse (bool, optional): Whether to sort in reverse order. Defaults to False.

        Raises:
            ValueError: If no observable with the given name is found.
        """
        if observable_name not in self.observables:
            raise ValueError(f"No observable named '{observable_name}' found.")

        # Get the observable values and sort the indices
        quantity = self.observables[observable_name].quantity
        sorted_indices = sorted(range(len(quantity)), # indices
                                key=quantity.__getitem__, # sort values[indices]
                                reverse=reverse)

        # Create a new trajectory with the sorted frames
        sorted_trajectory = self[sorted_indices]

        # Reset the current trajectory's attributes to the sorted trajectory's attributes
        self.__dict__ = sorted_trajectory.__dict__
    
    def __len__(self):
        """
        Get the number of frames in the trajectory.

        Returns:
            int: The number of frames in the trajectory.
        """
        return len(self.coordinates)

class EnsembleTrajectory(Trajectory):
    def __init__(self, summary: Summary, state_variables: Summary, coordinates: np.ndarray = None):
        super().__init__(summary,  coordinates)
        self.state_variables = state_variables

class EnsembleIsingTrajectory(EnsembleTrajectory):
    def __init__(self, summary: Summary, state_variables: Summary, coordinates: np.ndarray = None):
        super().__init__(summary, state_variables, coordinates)
        self.state_variables = state_variables

        # The time series of the 2D ising model has shape (num_frames, size, size)
        if coordinates is not None:
            coordinates = np.array(coordinates)
            if len(coordinates.shape) == 3:
                self.coordinates = coordinates
            elif len(coordinates.shape) == 2:
                self.coordinates = coordinates.reshape((1, *coordinates.shape))

    def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the trajectory.

        Args:
            frame (np.ndarray): The frame to add.
        """
        frame = np.array(frame)

        # The frame should have shape (size, size) or (n_frames, size, size)
        if len(frame.shape) == 3:
            frame = frame
        elif len(frame.shape) == 2:
            frame = frame.reshape((1, *frame.shape)) 
        logger.debug(f"Adding frame of shape {frame.shape} to trajectory.")

        if self.coordinates is None:
            logger.debug(f"Initializing trajectory with frame of shape {frame.shape}.")
            self.coordinates = frame
            logger.debug(f"Initialized trajectory with shape {self.coordinates.shape}.")
        else:
            logger.debug(f"Concatenating frame of shape {frame.shape} to trajectory.")
            self.coordinates = np.concatenate((self.coordinates, frame))
            logger.debug(f"Concatenated frame to trajectory with shape {self.coordinates.shape}.")


class MultiEnsembleTrajectory:
    def __init__(self, trajectories: List[EnsembleTrajectory]):
        self.trajectories = {i:traj for i, traj in enumerate(trajectories)}