from data.trajectory import Trajectory
from data.generic import Summary
from typing import List, Dict, Union, Iterable
from slurmflow.serializer import ObjectSerializer
from sklearn.model_selection import ShuffleSplit
from tm.core.loader import Loader
import numpy as np
import pandas as pd

import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class MultiEnsembleDataset:
    def __init__(self, trajectories: Union[Iterable[Trajectory], Iterable[str]], summary: Summary = Summary()):
        """
        Initialize a MultiEnsembleDataset.

        Args:
            trajectories (Union[Iterable[Trajectory], Iterable[str]]): Either an iterable of Trajectory objects or an iterable of strings.
                If an iterable of strings is provided, it is assumed that these are paths to the trajectories, and the
                trajectories are loaded from these paths using the ObjectSerializer.
            summary (Summary): The summary of the dataset.
        """
        # If trajectories is an iterable of strings, load the trajectories from the provided paths
        trajectories_ = []
        is_iterable = isinstance(trajectories, Iterable)
        is_strs = all(isinstance(path, str) for path in trajectories)
        if is_iterable and is_strs:
            for path in trajectories:
                OS = ObjectSerializer(path)
                trajectories_.append(OS.load())
        else:
            trajectories_ = trajectories

        self.trajectories = trajectories_
        self.summary = summary

    def save(self, filename: str, overwrite: bool = True):
        """
        Save the dataset to disk.

        Args:
            filename (str): The filename to save to.
            overwrite (bool, optional): Whether to overwrite an existing file. Defaults to False.
        """
        OS = ObjectSerializer(filename)
        OS.serialize(self, overwrite=overwrite)

    @classmethod
    def load(cls, filename: str) -> 'Dataset':
        """
        Load a dataset from disk.

        Args:
            filename (str): The filename to load from.

        Returns:
            Dataset: The loaded dataset.
        """
        OS = ObjectSerializer(filename)
        return OS.load()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        def create_or_append_df(existing_df, new_data):
            new_df = pd.DataFrame([new_data])
            if existing_df is None:
                return new_df
            else:
                return pd.concat([existing_df, new_df], ignore_index=True)
            
        df = None
        for index, traj in enumerate(self.trajectories):
            row = {"index": index, **traj.summary.__dict__}
            df = create_or_append_df(df, row)
        return df
    
    def from_dataframe(self, df: pd.DataFrame) -> 'Dataset':
        """
        Convert a pandas DataFrame to a dataset.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            Dataset: The dataset.
        """
        trajectories = [self.trajectories[row['index']] for _, row in df.iterrows()]
        new_dataset = MultiEnsembleDataset(trajectories, summary=self.summary)
        return new_dataset

    def get_loader_args(self, state_variables: List[str]) -> Dict[str, List]:
        complete_dataset, paired_state_vars = [], []
        for trajectory in self.trajectories:
            state_var_chs = []
            state_var_vector = []
            if len(trajectory.coordinates.shape) == 3: # No channel dim
                coord_ch = np.expand_dims(trajectory.coordinates, 1)
            else:
                coord_ch = trajectory.coordinates
            
            for k in state_variables:
                state_var_chs.append(np.ones_like(coord_ch) * trajectory.summary[k])
                state_var_vector.append(np.ones((len(coord_ch), 1)) * trajectory.summary[k])
                
            state_var_chs = np.concatenate(state_var_chs, 1)
            state_vector = np.concatenate([coord_ch, state_var_chs], 1)
            state_var_vector = np.concatenate(state_var_vector, 1)

            complete_dataset.append(state_vector)
            paired_state_vars.append(state_var_vector)
            
        complete_dataset = np.concatenate(complete_dataset)
        paired_state_vars = np.concatenate(paired_state_vars)

        n_coords_ch = coord_ch.shape[1]
        n_state_var_ch = state_var_vector.shape[1]

        control_dims = (n_coords_ch, n_coords_ch + n_state_var_ch)
            
        return complete_dataset, paired_state_vars, control_dims

    def to_TMLoader(self, train_size: float, test_size: float, state_variables: List[str], **TMLoader_kwargs) -> Loader:
        """
        Convert the dataset to a DataLoader.

        Args:
            trajectories (Union[Iterable[Trajectory], Iterable[str]]): Either a Trajectories object or an iterable of strings.
                If an iterable of strings is provided, it is assumed that these are paths to the trajectories, and the
                trajectories are loaded from these paths using the ObjectSerializer.
            TMLoader_kwargs: Additional keyword arguments for the Loader.

        Returns:
            DataLoader: The DataLoader.
        """

        tm_dataset, paired_state_vars, control_dims = self.get_loader_args(state_variables)
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size)
        train_idxs, test_idxs = next(splitter.split(tm_dataset))
        train_loader = Loader(data=tm_dataset[train_idxs], temperatures=paired_state_vars[train_idxs], control_dims=control_dims, **TMLoader_kwargs)
        test_loader = Loader(data=tm_dataset[test_idxs], temperatures=paired_state_vars[test_idxs], control_dims=control_dims, **TMLoader_kwargs)

        return train_loader, test_loader