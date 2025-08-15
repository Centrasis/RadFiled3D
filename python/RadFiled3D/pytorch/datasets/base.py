from RadFiled3D.RadFiled3D import FieldStore, RadiationField as RawRadiationField, PolarRadiationField, CartesianRadiationField, RadiationFieldMetadata, VoxelGrid, PolarSegments, FieldAccessor, CartesianFieldAccessor, PolarFieldAccessor, Voxel
import zipfile
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, TaskProgressColumn, SpinnerColumn, MofNCompleteColumn
from rich import print
from torch.multiprocessing import Manager
from RadFiled3D.pytorch.types import TrainingInputData, RadiationField, RadiationFieldChannel, DirectionalInput, PositionalInput
from typing import Any


class MetadataLoadMode(Enum):
    FULL = 1
    HEADER = 2
    DISABLED = 3


class RadiationFieldDataset(Dataset):
    """
    A dataset that loads radiation field files and returns them as (field, metadata)-tuples.
    The dataset can be initialized with either a list of file paths in the file system (uncompressed) or a path to a zip file containing radiation field files.
    In the latter case, the file paths are either extracted from the zip file or can be provided as a list of relative paths. This is encouraged, as the splitting of the dataset in train, validation and test should be random an therefore all file paths should be known at the time of initialization.

    The dataset can be created by using the DatasetBuilder class. This allows the Builder to parse the zip or folder structure correctly and link the metadata to the radiation field files.

    The dataset is designed to be used with a DataLoader. The DataLoader should be initialized with a batch size of 1, as the radiation field files are already stored in memory and the dataset is not designed to be used with multiprocessing.
    """

    def __init__(self, file_paths: Union[list[str], str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        """
        :param file_paths: List of file paths to radiation field files. If zip_file is provided, this parameter can be None. In this case the file paths are extracted from the zip file. If file_paths is str, then it will be checked if it is an hdf5 file. If so it will be treated as an preprocessed dataset and loaded as such.
        :param zip_file: Path to a zip file containing radiation field files. If file_paths is provided, this parameter can be None. In this case the file paths are extracted from the zip file.
        :param metadata_load_mode: Mode for loading metadata. FULL loads the full metadata, HEADER only loads the header, DISABLED does not load metadata. Default is HEADER. The provided metdata is a RadiationFieldMetadata object or None.
        """
        if isinstance(file_paths, str) or isinstance(file_paths, Path):
            file_paths = [file_paths]

        if file_paths is not None:
            file_paths = [str(p) for p in file_paths]

        manager = Manager()
        self.file_paths = file_paths
        
        self.zip_file = zip_file
        self.metadata_load_mode = metadata_load_mode
        if self.file_paths is None and self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                self.file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
        elif self.file_paths is None and self.zip_file is None:
            raise ValueError("Either file_paths or zip_file must be provided.")
        
        self._field_accessor: FieldAccessor = None
        self.file_paths = manager.list(file_paths) if file_paths is not None else None

    def _get_field_accessor(self) -> Union[FieldAccessor, CartesianFieldAccessor, PolarFieldAccessor]:
        if self._field_accessor is None:
            if self.is_dataset_zipped:
                self._field_accessor = FieldStore.construct_field_accessor_from_buffer(self.load_file_buffer(0))
            else:
                self._field_accessor = FieldStore.construct_field_accessor(self.file_paths[0])
        return self._field_accessor
    
    field_accessor: Union[FieldAccessor, CartesianFieldAccessor, PolarFieldAccessor] = property(_get_field_accessor)
    is_dataset_zipped: bool = property(lambda self: self.zip_file is not None)

    def __len__(self):
        return len(self.file_paths)

    def load_file_buffer(self, idx: int) -> bytes:
        """
        Loads a binary file buffer from the dataset given an file index.
        :param idx: The index of the file in the dataset.
        :return: The binary file buffer.
        """
        return self.load_file_buffer_by_path(self.file_paths[idx])
        
    def load_file_buffer_by_path(self, file_path: str) -> bytes:
        if self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(file_path) as file:
                    return file.read()
        else:
            return open(file_path, 'rb').read()
    
    def _get_field(self, idx: int) -> Union[RawRadiationField, CartesianRadiationField, PolarRadiationField]:
        """
        Loads a radiation field from the dataset given a file index.
        :param idx: The index of the file in the dataset.
        :return: The radiation field.
        """
        return self._get_field_by_path(self.file_paths[idx])
        
    def _get_field_by_path(self, file_path: str) -> Union[RawRadiationField, CartesianRadiationField, PolarRadiationField]:
        """
        Loads a radiation field from the dataset given a file path.
        :param file_path: The path to the file in the dataset.
        :return: The radiation field.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_field_from_buffer(self.load_file_buffer_by_path(file_path))
        else:
            return self.field_accessor.access_field(file_path)

    def check_dataset_integrity(self) -> bool:
        """
        Checks if all radiation field files in the dataset are valid.
        :return: True, if all files are valid, False otherwise.
        """
        valid = True
        invalid_files_count = 0
        progressbar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn()
        )
        with progressbar as progress:
            task = progress.add_task("Checking dataset integrity...", total=len(self.file_paths))
            for idx in range(len(self.file_paths)):
                try:
                    field = self._get_field(idx)
                    if field is None:
                        raise ValueError("Field is None.")
                except Exception as e:
                    valid = False
                    print(f"Error loading file {self.file_paths[idx]}: {str(e)}")
                    invalid_files_count += 1
                progress.update(task, advance=1)
        if not valid:
            print(f"Dataset contains {invalid_files_count} invalid files.")
        return valid
    
    def _get_metadata(self, idx: int) -> Union[RadiationFieldMetadata, None]:
        """
        Loads the metadata of a radiation field from the dataset given a file index. This method respects the metadata_load_mode.
        :param idx: The index of the file in the dataset.
        :return: The metadata of the radiation field.
        """
        return self._get_metadata_by_path(self.file_paths[idx])
    
    def _get_metadata_by_path(self, file_path: str) -> Union[RadiationFieldMetadata, None]:
        """
        Loads the metadata of a radiation field from the dataset given a file path. This method respects the metadata_load_mode.
        :param file_path: The path to the file in the dataset.
        :return: The metadata of the radiation field.
        """
        if self.is_dataset_zipped:
            file_buffer = self.load_file_buffer_by_path(file_path)
            if self.metadata_load_mode == MetadataLoadMode.FULL:
                metadata: RadiationFieldMetadata = FieldStore.peek_metadata_from_buffer(file_buffer)
            elif self.metadata_load_mode == MetadataLoadMode.HEADER:
                metadata: RadiationFieldMetadata = FieldStore.peek_metadata_from_buffer(file_buffer)
            else:
                metadata = None
        else:
            if self.metadata_load_mode == MetadataLoadMode.FULL:
                metadata: RadiationFieldMetadata = FieldStore.load_metadata(file_path)
            elif self.metadata_load_mode == MetadataLoadMode.HEADER:
                metadata: RadiationFieldMetadata = FieldStore.peek_metadata(file_path)
            else:
                metadata = None
        return metadata

    def _get_voxel_flat(self, file_idx: int, vx_idx: int, channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel index.
        :param file_idx: The index of the file in the dataset.
        :param vx_idx: The index of the voxel in the radiation field.
        :param channel_name: The name of the channel to load.
        :param layer_name: The name of the layer to load.
        :return: The voxel.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel_flat_from_buffer(self.load_file_buffer(file_idx), channel_name, layer_name, vx_idx)
        else:
            return self.field_accessor.access_voxel_flat(self.file_paths[file_idx], channel_name, layer_name, vx_idx)

    def __getitem__(self, idx: int) -> tuple[Union[RadiationField, RawRadiationField, RadiationFieldChannel, PolarSegments, Voxel, Tensor], Union[PositionalInput, DirectionalInput, RadiationFieldMetadata]]:
        """
        Loads a radiation field and its metadata from the dataset given a file index.
        :param idx: The index of the file in the dataset.
        :return: A named tuple containing the radiation field and its metadata. Format: An instance of TrainingInputData, which contains the ground truth and and input.
        """
        idx = idx % len(self.file_paths)
        field = self._get_field(idx)
        metadata = self._get_metadata(idx)
        return (self.transform(field, idx), self.transform_origin(metadata, idx))

    def __getitems__(self, indices: list[int]) -> Union[TrainingInputData, list[TrainingInputData]]:
        return [self.__getitem__(idx) for idx in indices]

    def __iter__(self):
        return RadiationFieldDatasetIterator(self)

    def transform(self, field: Union[RawRadiationField, VoxelGrid, PolarSegments, Voxel], idx: int) -> Union[RadiationField, RadiationFieldChannel, PolarSegments, Voxel, Tensor]:
        """
        Override to transform a RadFiled3D type into a torch tensor.
        This should be used as the target for the model.
        By default this just returns the original RadFiled3D type.
        :param field: The original RadFiled3D type.
        :param idx: The index of the element in the dataset.
        :return: The transformed RadFiled3D type.
        """
        return field

    def transform_origin(self, metadata: RadiationFieldMetadata, idx: int) -> Union[PositionalInput, DirectionalInput, RadiationFieldMetadata]:
        """
        Override to transform a RadiationFieldMetadata into a torch tensor.
        This should be used as the input for the model.
        By default this just returns the original RadiationFieldMetadata.
        :param metadata: The RadiationFieldMetadata to transform.
        :param idx: The index of the metadata in the dataset.
        :return: The transformed RadiationFieldMetadata.
        """
        return metadata


class RadiationFieldDatasetIterator:
    def __init__(self, dataset: RadiationFieldDataset):
        self.dataset = dataset
        self.idx = 0

    def __iter__(self):
        return self
    
    def __next__(self) -> TrainingInputData:
        if self.idx < len(self.dataset):
            item = self.dataset[self.idx]
            self.idx += 1
            return item
        else:
            raise StopIteration
