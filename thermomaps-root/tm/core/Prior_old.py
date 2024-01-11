import torch
from scipy.optimize import curve_fit
import numpy as np
from typing import Any, Callable, Dict, List
import logging

logging.basicConfig(level=logging.DEBUG)

def temperature_density_rescaling(std_temp, ref_temp):
    """
    Calculate temperature density rescaling factor.

    Args:
        std_temp (torch.Tensor): Standardized temperature.
        ref_temp (float): Reference temperature.

    Returns:
        torch.Tensor: Rescaling factor.
    """
    return (std_temp / ref_temp).pow(0.5)


def identity(t, *args, **kwargs):
    """
    Identity function.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Unchanged input tensor.
    """
    return t


RESCALE_FUNCS = {"density": temperature_density_rescaling, "no_rescale": identity}


def linear_fit(x, a, b):
    """
    Linear fit function.

    Args:
        x (torch.Tensor): Input tensor.
        a (float): Slope.
        b (float): Intercept.

    Returns:
        torch.Tensor: Fitted values.
    """
    return x * a + b


FIT_FUNCS = {"linear": linear_fit}
BOUNDS = {"linear": ([0, -np.inf], [np.inf, np.inf])}
INITIAL_GUESS = {"linear": [1, 1]}


def parse_kwargs(**kwargs):
    """
    Parse keyword arguments.

    Args:
        **kwargs: Keyword arguments to be parsed.

    Returns:
        dict: Parsed keyword arguments with default values.
    """
    kwargs_ = {"mean": 0, "std": 1}
    for k, v in kwargs.items():
        kwargs_[k] = v
    return kwargs_


def rmsd(v, temp):
    """
    Calculate root mean square deviation (RMSD).

    Args:
        v (torch.Tensor): Input tensor.
        temp: Temperature.

    Returns:
        torch.Tensor: RMSD.
    """
    return ((v - v.mean(0)) ** 2).mean(0).sum(0)[None, :, :]


def filter_bounds(RMSD, mult=None, cutoff=None):
    """
    Filter RMSD values based on bounds.

    Args:
        RMSD (list of np.ndarray): List of RMSD arrays.
        mult (float, optional): Multiplication factor for bounds. Defaults to None.
        cutoff (float, optional): Cutoff value for RMSD. Defaults to None.

    Returns:
        np.ndarray: Filtered RMSD array.
    """
    mean, std = RMSD.mean(0), RMSD.std(0)
    RMSD_ = []
    if mult is not None:
        ub = mean + mult * std
        lb = mean - mult * std
    for x in RMSD:
        if mult is not None:
            np.putmask(x, x > ub, ub)
            np.putmask(x, x < lb, lb)
        x[x < cutoff] = cutoff
        RMSD_.append(x)
    RMSD = np.stack(RMSD_)
    return RMSD


def parallel_curve_fit(func, x, y, **kwargs):
    """
    Perform parallel curve fitting.

    Args:
        func (callable): Function to fit.
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
        **kwargs: Additional keyword arguments for curve_fit.

    Returns:
        np.ndarray: Fitted parameters.
    """
    params = np.ones_like(y[0])[None, :].repeat(2, 0)
    (_, xdims, ydims) = y.shape
    for i in range(xdims):
        for j in range(ydims):
            popt, pcov = curve_fit(func, x, y[:, i, j], bounds=kwargs["bounds"])
            params[:, i, j] = popt
    return params


class NormalPrior:
    """
    Normal prior distribution.
    """

    def __init__(self, **kwargs):
        """
        Initialize a NormalPrior.

        Args:
            **kwargs: Keyword arguments for the prior distribution (mean and std).
        """
        self.kwargs = parse_kwargs(kwargs)
        self.prior = torch.distributions.normal.Normal(
            self.kwargs["mean"], self.kwargs["std"]
        )

    def sample_prior(self, batch_size, *args, **kwargs):
        """
        Sample from the prior distribution.

        Args:
            batch_size (int): Batch size.
            **kwargs: Additional keyword arguments for sampling.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        return self.prior.sample(sample_shape=batch_size)


class LocalEquilibriumHarmonicPrior(NormalPrior):
    """
    Local equilibrium harmonic prior.
    """

    def __init__(
        self,
        data,
        temps,
        fit_key,
        cutoff,
        BOUNDS=BOUNDS,
        INITIAL_GUESS=INITIAL_GUESS,
        FIT_FUNCS=FIT_FUNCS,
        **kwargs
    ):
        """
        Initialize a LocalEquilibriumHarmonicPrior.

        Args:
            data (dict): Data dictionary.
            temps (list): List of temperatures.
            fit_key (str): Key for the fitting function.
            cutoff (float): Cutoff value for RMSD.
            BOUNDS (dict, optional): Bounds for fitting. Defaults to BOUNDS.
            INITIAL_GUESS (dict, optional): Initial guess for fitting. Defaults to INITIAL_GUESS.
            FIT_FUNCS (dict, optional): Fitting functions. Defaults to FIT_FUNCS.
            **kwargs: Keyword arguments for the prior distribution (mean and std).
        """
        self.kwargs = parse_kwargs(**kwargs)
        self.fit = FIT_FUNCS[fit_key]

        self.cutoff = cutoff

        RMSD = np.concatenate([rmsd(v, k) for k, v in data.items()], axis=0)
        RMSD_arr = filter_bounds(RMSD, mult=None, cutoff=cutoff)

        T = [float(k.split("_")[0]) for k in temps]

        self.params = parallel_curve_fit(self.fit, T, RMSD, bounds=BOUNDS["linear"])

        self.RMSD_d = {}
        for i, temp in enumerate(data.keys()):
            self.RMSD_d[temp] = RMSD_arr[i]

    def sample_prior_from_data(self, batch_size, temp, n_dims=4):
        """
        Sample prior from data.

        Args:
            batch_size (int): Batch size.
            temp (list): List of temperatures.
            n_dims (int, optional): Number of dimensions. Defaults to 4.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        stds = []

        for t in temp:
            std = torch.Tensor(self.RMSD_d[t]) ** 0.5
            std = torch.repeat_interleave(std[None, :, :], n_dims, dim=0)
            stds.append(std)

        stds = torch.stack(stds)
        stds[stds < self.cutoff] = self.cutoff
        extra_dims = torch.ones_like(stds)[:, 0, :, :]
        stds = torch.cat([stds, extra_dims[:, None, :, :]], dim=1)
        prior = torch.distributions.normal.Normal(0, stds)
        return prior.sample()

    def fit_prior(self, batch_size, temp, **kwargs):
        """
        Fit prior distribution.

        Args:
            batch_size (int): Batch size.
            temp (float): Temperature.
            **kwargs: Additional keyword arguments for fitting.

        Returns:
            torch.Tensor: Fitted standard deviations.
        """
        temp = np.repeat(np.array([temp]), batch_size, axis=0)
        stds = self.fit(temp[:, None, None, None], *self.params)
        stds = torch.Tensor(stds) ** 0.5
        stds[torch.isnan(stds)] = self.cutoff
        return stds

    def sample_prior_from_fit(self, batch_size, temp, n_dims=4):
        """
        Sample prior from fitted parameters.

        Args:
            batch_size (int): Batch size.
            temp (float): Temperature.
            n_dims (int, optional): Number of dimensions. Defaults to 4.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        stds = self.fit_prior(batch_size, temp)
        stds[stds < self.cutoff] = self.cutoff
        stds = torch.repeat_interleave(stds, n_dims, dim=1)
        extra_dims = torch.ones_like(stds)[:, 0, :, :]
        stds = torch.cat([stds, extra_dims[:, None, :, :]], dim=1)
        prior = torch.distributions.normal.Normal(0, torch.Tensor(stds))
        return prior.sample()

    def sample_prior(self, batch_size, temp, sample_type, n_dims, *args, **kwargs):
        """
        Sample prior distribution based on the sample_type.

        Args:
            batch_size (int): Batch size.
            temp (list): List of temperatures.
            sample_type (str): Type of sampling ('from_data' or 'from_fit').
            n_dims (int): Number of dimensions.
            **kwargs: Additional keyword arguments for sampling.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        if sample_type == "from_data":
            return self.sample_prior_from_data(batch_size, temp, n_dims=n_dims)
        if sample_type == "from_fit":
            samp = self.sample_prior_from_fit(batch_size, temp, n_dims=n_dims)
            return samp


class GlobalEquilibriumHarmonicPrior(LocalEquilibriumHarmonicPrior):
    """
    Global equilibrium harmonic prior.
    """

    def __init__(
        self,
        data: Dict,
        fit_key: str = "linear",
        BOUNDS: Dict[str, tuple] = BOUNDS,
        INITIAL_GUESS: Dict[str, float] = INITIAL_GUESS,
        FIT_FUNCS: Dict[str, Callable] = FIT_FUNCS,
        **kwargs: Any
    ):
        """
        Initialize a GlobalEquilibriumHarmonicPrior.

        Args:
            data (dict): Data dictionary.
            temps (list): List of temperatures.
            fit_key (str): Key for the fitting function.
            BOUNDS (dict, optional): Bounds for fitting. Defaults to BOUNDS.
            INITIAL_GUESS (dict, optional): Initial guess for fitting. Defaults to INITIAL_GUESS.
            FIT_FUNCS (dict, optional): Fitting functions. Defaults to FIT_FUNCS.
            **kwargs: Keyword arguments for the prior distribution (mean and std).
        """
        self.kwargs = parse_kwargs(**kwargs)
        self.fit = FIT_FUNCS[fit_key]
        self.cutoff = 1e-2
        self.mult = None

        T = [float(str(k).split("_")[0]) for k in data.keys()]
        RMSD = np.concatenate([rmsd(v, k) for k, v in data.items()], axis=0)

        for i, temp in enumerate(T):
            RMSD[i] = np.ones_like(RMSD[i]) * temp

        self.params = parallel_curve_fit(self.fit, T, RMSD, bounds=BOUNDS["linear"])

        self.RMSD_d = {}
        for i, temp in enumerate(T):
            self.RMSD_d[str(temp)] = np.ones_like(RMSD[i]) * float(temp)

        
        logging.debug(f"Fluctuation keys: {self.RMSD_d.keys()}")


    def sample_prior_from_data(self, batch_size, temp, n_dims=4):
        """
        Sample prior from data.

        Args:
            batch_size (int): Batch size.
            temp (list): List of temperatures.
            n_dims (int, optional): Number of dimensions. Defaults to 4.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        stds = []

        for t in temp:
            std = torch.Tensor(self.RMSD_d[t]) ** 0.5
            std = torch.repeat_interleave(std[None, :, :], n_dims, dim=0)
            stds.append(std)

        stds = torch.stack(stds)
        stds[stds < self.cutoff] = self.cutoff
        extra_dims = torch.ones_like(stds)[:, 0, :, :]
        stds = torch.cat([stds, extra_dims[:, None, :, :]], dim=1)
        prior = torch.distributions.normal.Normal(0, stds)
        return prior.sample()

    def fit_prior(self, batch_size, temp, **kwargs):
        """
        Fit prior distribution.

        Args:
            batch_size (int): Batch size.
            temp (float): Temperature.
            **kwargs: Additional keyword arguments for fitting.

        Returns:
            torch.Tensor: Fitted standard deviations.
        """
        temp = np.repeat(np.array([temp]), batch_size, axis=0)
        stds = self.fit(temp[:, None, None, None], *self.params)
        stds = torch.Tensor(stds) ** 0.5
        stds[torch.isnan(stds)] = self.cutoff
        return stds

    def sample_prior_from_fit(self, batch_size, temp, n_dims=4):
        """
        Sample prior from fitted parameters.

        Args:
            batch_size (int): Batch size.
            temp (float): Temperature.
            n_dims (int, optional): Number of dimensions. Defaults to 4.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        stds = self.fit_prior(batch_size, temp)
        stds[stds < self.cutoff] = self.cutoff
        stds = torch.repeat_interleave(stds, n_dims, dim=1)
        extra_dims = torch.ones_like(stds)[:, 0, :, :]
        stds = torch.cat([stds, extra_dims[:, None, :, :]], dim=1)
        prior = torch.distributions.normal.Normal(0, torch.Tensor(stds))
        return prior.sample()

    def sample_prior(self, batch_size, temp, sample_type, n_dims, *args, **kwargs):
        """
        Sample prior distribution based on the sample_type.

        Args:
            batch_size (int): Batch size.
            temp (list): List of temperatures.
            sample_type (str): Type of sampling ('from_data' or 'from_fit').
            n_dims (int): Number of dimensions.
            **kwargs: Additional keyword arguments for sampling.

        Returns:
            torch.Tensor: Sampled values from the prior.
        """
        if sample_type == "from_data":
            samp = self.sample_prior_from_data(batch_size, temp, n_dims=n_dims)
        if sample_type == "from_fit":
            samp = self.sample_prior_from_fit(batch_size, temp, n_dims=n_dims)
        return samp
