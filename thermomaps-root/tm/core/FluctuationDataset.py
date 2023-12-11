import numpy as np
from inspect import signature
from scipy.optimize import curve_fit
import types


def safe_index(arr, idx):
    if idx < len(arr):
        return arr[idx]
    else:
        return arr[-1]

def linear_fit(x, a, b):
    return x*a + b

class Fluctuation:
    """
    A class for performing fluctuation analysis and fitting.

    Args:
        func (callable): The fitting function.
        func_kwargs (dict): Additional keyword arguments for curve fitting.
    """
    def __init__(self, func, **func_kwargs):
        sig = signature(func)
        self.func = func
        self.free_params = len(sig.parameters) - 1
        self.func_kwargs = func_kwargs

    def filter_bounds(self, data, mult=None, cutoff=None):
        """
        Filter the bounds of the data based on provided criteria.

        Args:
            data (numpy.ndarray): The data to be filtered.
            mult (float, optional): Multiplicative factor for filtering. Defaults to None.
            cutoff (float, optional): Cutoff value for filtering. Defaults to None.

        Returns:
            numpy.ndarray: The filtered data.
        """
        mean, std = data.mean(0), data.std(0)
        if mult is not None:
            lb = mean - mult * std
            ub = mean + mult * std

        data_ = []
        for x in data:
            if mult is not None:
                np.putmask(x, x > ub, ub)
                np.putmask(x, x < lb, lb)
            if cutoff is not None:
                x[x < cutoff] = cutoff
            data_.append(x)
        data = np.stack(data_)
        return data

    def rmsd_traj(self, coords):
        """
        Calculate the root mean square deviation (RMSD) for a trajectory.

        Args:
            coords (numpy.ndarray): The trajectory coordinates.

        Returns:
            numpy.ndarray: The RMSD values.
        """
        return ((coords - coords.mean(0))**2).mean(0).sum(0)[None, :, :]

    def rmsd_multitraj(self, coords_dict):
        """
        Calculate the RMSD for multiple trajectories.

        Args:
            coords_dict (dict): A dictionary of trajectory coordinates.

        Returns:
            numpy.ndarray: The RMSD values.
        """
        return np.concatenate([self.rmsd_traj(v) for k, v in coords_dict.items()], axis=0)

    def param_from_rmsd(self, coords_dict, mult, cutoff):
        """
        Calculate parameters from RMSD values.

        Args:
            coords_dict (dict): A dictionary of trajectory coordinates.
            mult (float): Multiplicative factor for filtering.
            cutoff (float): Cutoff value for filtering.

        Returns:
            numpy.ndarray: The calculated parameters.
        """
        temps = np.array([float(k.split("_")[0]) for k in coords_dict.keys()])
        fluctuations = self.rmsd_multitraj(coords_dict)
        fluctuations_filt = self.filter_bounds(fluctuations, mult=mult, cutoff=cutoff)
        params = np.ones_like(fluctuations_filt[0])[None, :].repeat(self.free_params, 0)
        return temps, fluctuations_filt, params

    def rmsd_fit(self, coords_dict, mult=None, cutoff=None):
        """
        Fit a function to the RMSD data.

        Args:
            coords_dict (dict): A dictionary of trajectory coordinates.
            mult (float, optional): Multiplicative factor for filtering. Defaults to None.
            cutoff (float, optional): Cutoff value for filtering. Defaults to None.

        Returns:
            numpy.ndarray: The fitted parameters.
        """
        self.temps, self.fluctuations, params = self.param_from_rmsd(coords_dict, mult, cutoff)
        (_, xdims, ydims) = self.fluctuations.shape
        for i in range(xdims):
            for j in range(ydims):
                self.popt, pcov = curve_fit(self.func,
                                            self.temps.astype(float),
                                            self.fluctuations[:, i, j],
                                            **self.func_kwargs,
                                            maxfev=10000
                                           )
                params[:, i, j] = self.popt
        return params

class RNAFluctuationPrior:
    """
    A class for computing fluctuation data and saving it.

    Args:
        mult_schedule (list): A list of multiplicative factors for filtering.
        cutoff (float): Cutoff value for filtering.
    """
    def __init__(self, mult_schedule, cutoff):
        self.mult_schedule = mult_schedule
        self.cutoff = cutoff

    def compute_from_simulation_data(self, rvec_path, gvec_path, data_path, temps_path, iter, save):
        """
        Compute fluctuation data from simulation data.

        Args:
            args (argparse.Namespace, optional): Command-line arguments. Defaults to None.
            save (bool, optional): Whether to save the computed data. Defaults to False.

        Returns:
            numpy.ndarray: The computed data.
            numpy.ndarray: The corresponding keys.
        """

        rvecs = np.load(rvec_path, allow_pickle=True).item(0)
        gvecs = np.load(gvec_path, allow_pickle=True).item(0)

        linear_model = Fluctuation(linear_fit, bounds=([0, -np.inf], [np.inf, np.inf]))
        params = linear_model.rmsd_fit(rvecs,
                                       mult=safe_index(self.mult_schedule, int(iter)),
                                       cutoff=self.cutoff)
        self.params = params

        data = []
        keys = []
        for i, (k, v) in enumerate(gvecs.items()):
            v = v.astype(np.float32)
            fluct_ch = linear_model.fluctuations[i][None, None, :, :]
            fluct_ch = np.repeat(fluct_ch, v.shape[0], axis=0)
            data.append(np.concatenate([v, fluct_ch], axis=1))
            keys.append([k for _ in range(v.shape[0])])

        data = np.concatenate(data, axis=0)
        keys = np.concatenate(keys, axis=0)
        if save:
            np.save(data_path, data)
            np.save(temps_path, keys)

        return data, keys

    @staticmethod
    def load(data_path, key_path):
        """
        Load data and keys from files.

        Args:
            data_path (str): Path to the data file.
            key_path (str): Path to the keys file.

        Returns:
            numpy.ndarray: The loaded data.
            numpy.ndarray: The loaded keys.
        """
        return np.load(data_path), np.load(key_path)
