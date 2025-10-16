from RadFiled3D.RadFiled3D import uvec2, vec2, FieldStore, RadiationField, PolarRadiationField, RadiationFieldMetadata, VoxelGrid, PolarSegments, FieldAccessor, PolarFieldAccessor, Voxel
from typing import Union, Tuple
from .base import MetadataLoadMode, RadiationFieldDataset
from torch import Tensor


class PolarFieldDataset(RadiationFieldDataset):
    def __init__(self, file_paths = None, zip_file = None, metadata_load_mode = MetadataLoadMode.HEADER):
        super().__init__(file_paths, zip_file, metadata_load_mode)
        self._field_dimensions = None

    def _get_field_accessor(self) -> PolarFieldAccessor:
        return super()._get_field_accessor()
    
    field_accessor: PolarFieldAccessor = property(_get_field_accessor)

    @property
    def field_dimensions(self) -> vec2:
        if self._field_dimensions is None:
            field = self._get_field(0)
            self._field_dimensions = field.get_segments_count()
        return self._field_dimensions

    def _get_field(self, idx: int) -> PolarRadiationField:
        return super()._get_field(idx)
    
    def _get_layer(self, idx: int, channel_name: str, layer_name: str) -> PolarSegments:
        """
        Loads a radiation layer from the dataset given a file index, a channel name and a layer name.
        :param idx: The index of the file in the dataset.
        :param channel_name: The name of the channel to load.
        :param layer_name: The name of the layer to load.
        :return: The radiation layer.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_layer_from_buffer(self.load_file_buffer(idx), channel_name, layer_name)
        else:
            return self.field_accessor.access_layer(self.file_paths[idx], channel_name, layer_name)
    
    def _get_voxel(self, file_idx: int, vx_idx: Union[Tuple[int, int], uvec2], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel index.
        :param file_idx: The index of the file in the dataset.
        :param vx_idx: The index of the voxel in the radiation field.
        :return: The voxel.
        """
        vx_idx = vx_idx if isinstance(vx_idx, uvec2) else uvec2(vx_idx[0], vx_idx[1])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel(self.load_file_buffer(file_idx), channel_name, layer_name, vx_idx)
        else:
            return self.field_accessor.access_voxel(self.file_paths[file_idx], channel_name, layer_name, vx_idx)

    def _get_voxel_by_coord(self, file_idx: int, vx_coord: Union[Tuple[float, float], vec2], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel coordinate.
        :param file_idx: The index of the file in the dataset.
        :param vx_coord: The coordinate of the voxel in the radiation field in world space.
        :return: The voxel.
        """
        vx_coord = vx_coord if isinstance(vx_coord, vec2) else vec2(vx_coord[0], vx_coord[1])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel_by_coord(self.load_file_buffer(file_idx), channel_name, layer_name, vx_coord)
        else:
            return self.field_accessor.access_voxel_by_coord(self.file_paths[file_idx], channel_name, layer_name, vx_coord)


class PolarFieldSingleLayerDataset(PolarFieldDataset):
    """
    A dataset that loads single layers from a single channel of a radiation field as PolarSegments.
    Useful, when only a single layer of a single channel is needed for training.
    """
    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        self.channel_name: str = None
        self.layer_name: str = None

    def set_channel_and_layer(self, channel_name: str, layer_name: str):
        self.channel_name = channel_name
        self.layer_name = layer_name

    def __getitem__(self, idx: int) -> Tuple[PolarSegments, Union[RadiationFieldMetadata, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        layer = self._get_layer(idx, self.channel_name, self.layer_name)
        metadata = self._get_metadata(idx)
        return (self.transform(layer, idx), self.transform_origin(metadata, idx))


class PolarSingleVoxelDataset(PolarFieldSingleLayerDataset):
    def __init__(self, file_paths = None, zip_file = None, metadata_load_mode = MetadataLoadMode.HEADER):
        super().__init__(file_paths, zip_file, metadata_load_mode)
        self.zip_ref = None # remove zip reference to avoid pickling issues
        self._field_accessor = None # remove field accessor to avoid pickling issues

    def __len__(self) -> int:
        vx_count = int(self.field_accessor.get_voxel_count())
        self._field_accessor = None  # remove field accessor to avoid pickling issues
        return super().__len__() * vx_count
    
    def _get_field(self, idx: int) -> PolarRadiationField:
        return super()._get_field(idx // self.field_accessor.get_voxel_count())
    
    def _get_metadata(self, idx) -> Union[RadiationFieldMetadata, None]:
        return super()._get_metadata(idx // self.field_accessor.get_voxel_count())

    def __getitem__(self, idx) -> Tuple[Tensor, Union[Tensor, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        vx_idx = idx % self.field_accessor.get_voxel_count()
        ret_val = (self._get_voxel_flat(idx // self.field_accessor.get_voxel_count(), vx_idx, self.channel_name, self.layer_name), self._get_metadata(idx // self.field_accessor.get_voxel_count()))
        return (self.transform(ret_val[0], idx), self.transform_origin(ret_val[1], idx))
