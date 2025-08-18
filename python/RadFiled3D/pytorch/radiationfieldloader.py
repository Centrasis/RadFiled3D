import zipfile
import os
from torch.utils.data import random_split, DataLoader
from typing import Type, Union, Callable
from torch.utils.data import _utils as pt_utils
from .datasets import RadiationFieldDataset
from .types import TrainingInputData


class DataLoaderBuilder(object):
    """
    A class that builds a RadiationFieldDataset from a directory or zip file and constructs DataLoaders for training, validation and testing.
    The dataset is split into train, validation and test sets according to the ratios provided in the constructor.
    Please note, that when using custom dataset classes to inherit from RadiationFieldDataset, the class must be provided as a parameter to the constructor.
    When using a custom dataset class, only the arguments file_paths and zip_file are passed to the constructor.
    """ 

    def __init__(self, dataset_path: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, dataset_class: Type[RadiationFieldDataset] = RadiationFieldDataset, on_dataset_created: Callable[[RadiationFieldDataset], None] = None):
        """
        :param dataset_path: The path to the directory or zip file containing the radiation field files.
        :param train_ratio: The ratio of the dataset that is used for training. Default is 0.7.
        :param val_ratio: The ratio of the dataset that is used for validation. Default is 0.15.
        :param test_ratio: The ratio of the dataset that is used for testing. Default is 0.15.
        :param dataset_class: The class of the dataset. Default is RadiationFieldDataset.
        :param on_dataset_created: A callback function that is called when the dataset is created. The function is called with the dataset as parameter.
        """
        self.on_dataset_created = on_dataset_created
        self.dataset_path = dataset_path
        self.dataset_path = str(self.dataset_path)
        self.dataset_class = dataset_class
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        if os.path.isdir(dataset_path):
            self.zip_file = None
            self.file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".rf3")]
            if len(self.file_paths) == 0 and os.path.isdir(os.path.join(dataset_path, "fields")):
                self.file_paths = [os.path.join(dataset_path, "fields", f) for f in os.listdir(os.path.join(dataset_path, "fields")) if f.endswith(".rf3")]
            elif len(self.file_paths) == 0:
                raise FileNotFoundError(f"No radiation field files found in directory {dataset_path}.")
        elif os.path.isfile(dataset_path) and zipfile.is_zipfile(dataset_path):
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                self.file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
            self.zip_file = dataset_path
        else:
            raise ValueError(f"Dataset path {dataset_path} is neither a directory nor a zip file.")
        
        self.train_files, self.val_files, self.test_files = random_split(self.file_paths, [train_ratio, val_ratio, test_ratio])

    def build_train_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.train_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        if self.on_dataset_created is not None:
            self.on_dataset_created(ds)
        return ds
    
    def build_val_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.val_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        if self.on_dataset_created is not None:
            self.on_dataset_created(ds)
        return ds
    
    def build_test_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.test_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        if self.on_dataset_created is not None:
            self.on_dataset_created(ds)
        return ds

    @staticmethod
    def collate_wrapper(batch):
        if isinstance(batch, list) and len(batch) > 0:
            sample = batch[0]
            if hasattr(sample, '_fields'):  # It's a named tuple
                fields = sample._fields
                collated_fields = {}
                
                for field_name in fields:
                    # Collect all non-None values for this field across the batch
                    field_values = [getattr(item, field_name) for item in batch if getattr(item, field_name) is not None]
                    assert len(field_values) == len(batch) or len(field_values) == 0, f"All items in the batch must have the same fields filled. That means either {field_name} is None for all elements of the batch or not None for all."

                    if len(field_values) > 0:  # Only collate if there are non-None value
                        if hasattr(field_values[0], '_fields'):  # another named tuple
                            collated_fields[field_name] = DataLoaderBuilder.collate_wrapper(field_values)
                        else:
                            try:
                                collated_fields[field_name] = pt_utils.collate.default_collate(field_values)
                            except Exception:
                                raise ValueError(f"Failed to collate field '{field_name}': {field_values}")
                    else:
                        collated_fields[field_name] = None
                return sample.__class__(**collated_fields)
            else:
                return pt_utils.collate.default_collate(batch)
        else:
            return batch

    def build_dataloader(self, dataset: RadiationFieldDataset, batch_size=1, shuffle=True, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the dataset.
        :param dataset: The dataset to build the DataLoader for.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        is_multiprocessing = worker_count is None or worker_count != 0
        if worker_count is None or worker_count < 0:
            worker_count = max(1, os.cpu_count() - 1)

        
        
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=worker_count,
            pin_memory=is_multiprocessing,
            persistent_workers=is_multiprocessing,
            collate_fn=DataLoaderBuilder.collate_wrapper
        )
        dl.dataset._field_accessor = None
        return dl

    def build_train_dataloader(self, batch_size=1, shuffle=True, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the training dataset.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        return self.build_dataloader(self.build_train_dataset(), batch_size=batch_size, shuffle=shuffle, worker_count=worker_count)
    
    def build_val_dataloader(self, batch_size=1, shuffle=False, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the validation dataset.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        return self.build_dataloader(self.build_val_dataset(), batch_size=batch_size, shuffle=shuffle, worker_count=worker_count)
    
    def build_test_dataloader(self, batch_size=1, shuffle=False, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the test dataset.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        return self.build_dataloader(self.build_test_dataset(), batch_size=batch_size, shuffle=shuffle, worker_count=worker_count)
