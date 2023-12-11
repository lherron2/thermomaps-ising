import os


class Directory:
    """
    Reads relevant paths from a YAML file defined for an experiment.
    """

    def __init__(
        self,
        pdb: str,
        expid: str,
        iter: str,
        identifier: str,
        device: str,
        num_devices: int,
        **paths
    ):
        """
        Initialize a Directory object with experiment-specific paths.

        Args:
            pdb (str): PDB identifier.
            expid (str): Experiment identifier.
            iter (str): Iteration identifier.
            identifier (str): Identifier for the directory.
            device (str): Device name.
            num_devices (int): Number of devices.
            **paths: Dictionary of paths loaded from a YAML file.
        """
        self.device_ids = list(range(0, num_devices))

        # Replacing wildcards in loaded YAML files
        self.pdb = pdb
        wildcards = {"PDBID": self.pdb, "EXPID": expid, "ITER": iter}
        self.paths = self.replace_wildcards(paths, wildcards)

        self.identifier = identifier
        self.base_path = self.paths["experiment_path"]
        self.model_path = os.path.join(self.base_path, self.paths["model_path"])
        self.data_path = self.paths["dataset_path"]
        self.temps_path = os.path.join(self.base_path, self.paths["temperature_path"])
        self.fluct_path = os.path.join(self.base_path, self.paths["fluctuation_path"])
        self.sample_path = os.path.join(self.base_path, self.paths["sample_path"])
        self.device = device

        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

    @staticmethod
    def replace_wildcards(d, wildcard_d):
        """
        Replace wildcards in dictionary values.

        Args:
            d (dict): Dictionary with values.
            wildcard_d (dict): Dictionary with wildcard replacements.

        Returns:
            dict: Dictionary with wildcard-replaced values.
        """
        for k_header, d_ in d.items():
            for k, v in d_.items():
                if isinstance(v, str):
                    for k_, v_ in wildcard_d.items():
                        v = v.replace(k_, v_)
                    d[k_header][k] = v
        return d

    def get_backbone_path(self):
        """
        Get the model path.

        Returns:
            str: Model path.
        """
        return self.model_path

    def get_dataset_path(self):
        """
        Get the dataset path.

        Returns:
            str: Dataset path.
        """
        return self.data_path

    def get_temps_path(self):
        """
        Get the temperature path.

        Returns:
            str: Temperature path.
        """
        return self.temps_path

    def get_sample_path(self):
        """
        Get the sample path.

        Returns:
            str: Sample path.
        """
        return self.sample_path

    def get_device(self):
        """
        Get the device name.

        Returns:
            str: Device name.
        """
        return self.device

    def get_pdb(self):
        """
        Get the PDB identifier.

        Returns:
            str: PDB identifier.
        """
        return self.pdb
