import torch
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import logging

def regex_list(regex, l):
    """
    Filter a list using a regular expression.

    Args:
        regex (str): The regular expression pattern.
        l (list): The list to be filtered.

    Returns:
        list: Filtered list containing elements that match the pattern.
    """
    return list(filter(re.compile(regex).match, l))

class ArrayWrapper:
    def __init__(self, array):
        self.array = array

    def as_torch(self):
        return torch.from_numpy(self.array)
    
    def as_numpy(self):
        return self.array
    