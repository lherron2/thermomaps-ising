
import pandas as pd
from typing import Any, Iterable, List
import logging
from slurmflow.serializer import ObjectSerializer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Summary:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __getitem__(self, key):
        return getattr(self, key)

class Report:
    def __init__(self, report_path: str):
        self.name = report_path
        with open(report_path, 'r') as file:
            self.report = file.read()

    def save(self, filename: str):
        with open(filename, 'w') as file:
            file.write(self.report)

class DataFormat:
    def __init__(self, summary: Summary):
        self.summary = summary
    
    def save(self, filename: str, overwrite: bool = False):
        OS = ObjectSerializer(filename)
        OS.serialize(self, overwrite=overwrite)
    
    @classmethod
    def load(cls, filename: str) -> object:
        OS = ObjectSerializer(filename)
        return OS.load()


class Registry:

    def __init__(self, objects: Iterable[object]):
        self.objects = []
        for obj in objects:
            self.add_object(obj)
        self.lookup_table = self.create_lookup_table()

    def add_object(self, obj: object):
        assert isinstance(obj, Summary) or isinstance(obj, DataFormat), "Object must be a Summary or a subclass of DataFormat."
        self.objects.append(obj)
        self.lookup_table = self.create_lookup_table()

    def create_lookup_table(self, method: str = 'intersection') -> pd.DataFrame:
        if method == 'intersection':
            common_attrs = set.intersection(*(set(vars(obj)) for obj in self.objects))
            return pd.DataFrame([{attr: getattr(obj, attr, None) for attr in common_attrs} for obj in self.objects])
        elif method == 'union':
            all_attrs = set.union(*(set(vars(obj)) for obj in self.objects))
            return pd.DataFrame([{attr: getattr(obj, attr, None) for attr in all_attrs} for obj in self.objects])
        else:
            raise ValueError("Method must be 'intersection' or 'union'.")
    
    def lookup_by_index(self, index: int) -> object:
        return self.objects[index]

    def lookup_by_attribute(self, attr_name: str, attr_value: Any) -> List[object]:
        return [obj for obj in self.objects if getattr(obj, attr_name, None) == attr_value]
