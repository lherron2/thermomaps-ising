from torch.utils.data import Dataset
import torch
from tm.core.Directory import Directory
import numpy as np


class Transform:
    """
    Base class for data transforms.
    """

    def __init__(self, data):
        """
        Initialize a Transform.

        Args:
            data (torch.Tensor): Input data.
        """
        pass


class WhitenTransform(Transform):
    """
    Whitening data transform.
    """

    def __init__(self, data):
        """
        Initialize a WhitenTransform.

        Args:
            data (torch.Tensor): Input data.
        """
        super().__init__(data)
        self.mean = data.mean(0)
        self.std = data.std(0)

    def forward(self, x):
        """
        Forward transformation: Standardizes the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Standardized data.
        """
        try:
            (_, nc, _, _) = x.shape
            return (x - self.mean[:nc, :, :]) / (self.std[:nc, :, :])
        except:
            return (x - self.mean[-1, :, :]) / (self.std[-1, :, :])

    def reverse(self, x):
        """
        Reverse transformation: Reverts standardized data to its original scale.

        Args:
            x (torch.Tensor): Standardized data.

        Returns:
            torch.Tensor: Original-scale data.
        """
        try:
            (_, nc, _, _) = x.shape
            return x * (self.std[:nc, :, :]) + self.mean[:nc, :, :]
        except:
            return x * (self.std[-1, :, :]) + self.mean[-1, :, :]


class MinMaxTransform(Transform):
    """
    Min-Max scaling data transform.
    """

    def __init__(self, data, dim, pos):
        """
        Initialize a MinMaxTransform.

        Args:
            data (torch.Tensor): Input data.
            dim (int): Dimension to standardize.
            pos (tuple): Position within the dimension to standardize.
        """
        super().__init__(data, dim, pos)

        self.min_data = data.min(0)[pos]
        self.max_data = data.max(0)[pos]

    def forward(self, x):
        """
        Forward transformation: Applies Min-Max scaling to the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Transformed data.
        """
        return (x - self.min_data) / 2 * (self.max_data - self.min_data)

    def reverse(self, x):
        """
        Reverse transformation: Reverts Min-Max scaled data to its original scale.

        Args:
            x (torch.Tensor): Transformed data.

        Returns:
            torch.Tensor: Original-scale data.
        """
        return 2 * (self.max_data - self.min_data) * x + self.min_data


class IdentityTransform(Transform):
    """
    Identity data transform.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an IdentityTransform.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        Forward transformation: Returns the input data unchanged.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Unchanged data.
        """
        return x

    def reverse(self, x):
        """
        Reverse transformation: Returns the input data unchanged.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Unchanged data.
        """
        return x


TRANSFORMS = {
    "whiten": WhitenTransform,
    "min_max": MinMaxTransform,
    "identity": IdentityTransform,
}


class Dequantizer:
    """
    Base class for dequantization methods.
    """

    def __init__(self, scale):
        """
        Initialize a Dequantizer.

        Args:
            scale (float): Dequantization scale.
        """
        self.scale = scale


class NormalDequantization(Dequantizer):
    """
    Normal dequantization method.
    """

    def __init__(self, scale):
        """
        Initialize a NormalDequantization.

        Args:
            scale (float): Dequantization scale.
        """
        super().__init__(scale)

    def forward(self, x):
        """
        Forward dequantization: Adds normal noise to the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Dequantized data.
        """
        return x + torch.randn(*x.shape) * self.scale


class UniformDequantization(Dequantizer):
    """
    Uniform dequantization method.
    """

    def __init__(self, scale):
        """
        Initialize a UniformDequantization.

        Args:
            scale (float): Dequantization scale.
        """
        super().__init__(scale)

    def forward(self, x):
        """
        Forward dequantization: Adds uniform noise to the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Dequantized data.
        """
        return x + torch.rand(*x.shape) * self.scale


DEQUANTIZERS = {"normal": NormalDequantization, "uniform": UniformDequantization}


class Loader(Dataset):
    """
    Dataset loader with optional data transformations and dequantization.
    """

    def __init__(
        self,
        data: torch.Tensor = None,
        temperatures: np.ndarray = None,
        transform_type: str = "whiten",
        control_axis: int = 1,
        control_dims: tuple = (3,5),
        dequantize: bool = True,
        dequantize_type: str = "normal",
        dequantize_scale: float = 1e-2,
        TRANSFORMS: dict = TRANSFORMS,
        DEQUANTIZERS: dict = DEQUANTIZERS,
    ):
        """
        Initialize a Loader instance.

        Args:
            directory (Directory): Data directory.
            transform_type (str, optional): Type of data transformation. Defaults to "whiten".
            control_tuple (tuple, optional): Control parameter settings. Defaults to ((1),(3,5)).
            dequantize (bool, optional): Whether to apply dequantization. Defaults to False.
            dequantize_type (str, optional): Type of dequantization. Defaults to "normal".
            dequantize_scale (float, optional): Dequantization scale. Defaults to 1e-2.
            TRANSFORMS (dict, optional): Dictionary of data transforms. Defaults to TRANSFORMS.
            DEQUANTIZERS (dict, optional): Dictionary of dequantization methods. Defaults to DEQUANTIZERS.
        """
        # Load data from npz file using path from Directory.
        self.control_tuple = (control_axis, control_dims)

        # Check the type of 'data' and load it accordingly
        if isinstance(data, str):
            self.data = torch.from_numpy(np.load(data)).float()
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            self.data = data

        # Check the type of 'temperatures' and load it accordingly
        if isinstance(temperatures, str):
            self.temps = np.load(temperatures)
        elif isinstance(temperatures, np.ndarray):
            self.temps = temperatures

        # If dequantize is True, apply the dequantization
        if dequantize:
            self.dequantizer = DEQUANTIZERS[dequantize_type](dequantize_scale)
            self.data = self.dequantize(self.data)

        # Get the dimensions of the data
        self.data_dim = self.data.shape[-1]
        self.num_channels = self.data.shape[1]
        self.num_dims = len(self.data.shape)

        # Build slice objects to retrieve control params and batch from Tensor.
        self.control_slice = self.build_control_slice(control_axis, control_dims, self.num_dims)
        self.batch_slice = self.build_batch_slice(self.num_dims)

        # Apply the specified transform to the data
        self.transform = TRANSFORMS[transform_type](self.data)

        # Standardize the control data
        self.unstd_control = self.data[self.control_slice][self.batch_slice]
        self.std_control = self.standardize(self.data)[self.batch_slice]

    def build_control_slice(self, control_axis, control_dims, data_dim):
        """
        Builds a slice object to retrieve the control parameters from the tensor.

        Args:
            control_tuple (tuple): Control parameter settings.
            data_dim (int): Number of dimensions in the tensor.

        Returns:
            tuple: Control slice, control dimension, and control position.
        """
        # # Create a list of slice objects that select all elements along each dimension
        # control_slice = [slice(None) for _ in range(data_dim)]

        # # If control_dims is not None, modify the slice objects for the control dimensions
        # if control_dims is not None:
        #     for dim in control_dims:
        #         control_slice[dim] = slice(control_dims[0], control_dims[1])


        if control_axis is None and control_dims is None:
            control_slice = [slice(None) for _ in range(data_dim)]
        else:
            control_slice = [slice(None, None) for _ in range(data_dim)] 
            for axis in [control_axis]:
                control_slice[axis] = slice(control_dims[0], control_dims[1])    

        return tuple(control_slice)

    def build_batch_slice(self, data_dim, batch_dim=0):
        """
        Preserves the batch dimension of a tensor while taking the first element along
        the other dimensions.

        Args:
            data_dim (int): Number of dimensions in the tensor.
            batch_dim (int, optional): Batch dimension. Defaults to 0.

        Returns:
            list: Batch slice.
        """
        batch_slice = [slice(0, 1) for dim in range(data_dim)]
        batch_slice[batch_dim] = slice(None, None)
        return batch_slice

    def dequantize(self, x):
        """
        Calls the dequantization method defined in DEQUANTIZERS.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Dequantized data.
        """
        return self.dequantizer.forward(x)

    def standardize(self, x):
        """
        Calls the standardizing transform defined in TRANSFORMS.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Standardized data.
        """
        return self.transform.forward(x)

    def unstandardize(self, x):
        """
        Calls the inverse of the standardizing transform defined in TRANSFORMS.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Unstandardized data.
        """
        return self.transform.reverse(x)

    def get_data_dim(self):
        """
        Get the data dimension.

        Returns:
            int: Data dimension.
        """
        return self.data_dim

    def get_num_dims(self):
        """
        Get the number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return self.num_dims

    def get_num_channels(self):
        """
        Get the number of channels.

        Returns:
            int: Number of channels.
        """
        return self.num_channels

    def get_all_but_batch_dim(self):
        """
        Get dimensions of data excluding the batch dimension.

        Returns:
            tuple: Dimensions of data excluding batch dimension.
        """
        return self.data.shape[1:]

    def get_batch(self, index):
        """
        Get a batch of data by index (for testing purposes).

        Args:
            index (int): Index of the batch.

        Returns:
            torch.Tensor: Batch of data.
        """
        x = self.data[index : index + 1]
        std_control = self.std_control[index : index + 1]
        x[self.control_slice] = std_control
        return x.float()

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple containing standardized control parameters and data.
        """
        x = torch.clone(self.data[index : index + 1])
        temps = self.temps[index]
        return temps, x.float()[0]

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return np.shape(self.data)[0]
