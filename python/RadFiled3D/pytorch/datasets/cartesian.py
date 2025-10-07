from RadFiled3D.RadFiled3D import uvec3, vec3, CartesianRadiationField, RadiationFieldMetadata, VoxelGrid, CartesianFieldAccessor, Voxel
from .base import MetadataLoadMode, RadiationFieldDataset
from typing import Union, Tuple
from torch import Tensor


class CartesianFieldDataset(RadiationFieldDataset):
    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        field = self._get_field(0)
        assert isinstance(field, CartesianRadiationField), "Dataset must contain CartesianRadiationFields."

    def _get_field(self, idx: int) -> CartesianRadiationField:
        return super()._get_field(idx)
    
    def _get_field_accessor(self) -> CartesianFieldAccessor:
        return super()._get_field_accessor()
    
    field_accessor: CartesianFieldAccessor = property(_get_field_accessor)

    def _get_channel(self, idx: int, channel_name: str) -> VoxelGrid:
        """
        Loads a radiation channel from the dataset given a file index and a channel name.
        :param idx: The index of the file in the dataset.
        :param channel_name: The name of the channel to load.
        :return: The radiation channel.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_channel_from_buffer(self.load_file_buffer(idx), channel_name)
        else:
            return self.field_accessor.access_channel(self.file_paths[idx], channel_name)
        
    def _get_layer(self, idx: int, channel_name: str, layer_name: str) -> VoxelGrid:
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
  
    def _get_voxel(self, file_idx: int, vx_coord: Union[Tuple[int, int, int], uvec3], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel coordinate.
        :param file_idx: The index of the file in the dataset.
        :param vx_coord: The coordinate of the voxel in the radiation field in quantized space.
        :return: The voxel.
        """
        vx_coord = vx_coord if isinstance(vx_coord, uvec3) else uvec3(vx_coord[0], vx_coord[1], vx_coord[2])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel(self.load_file_buffer(file_idx), channel_name, layer_name, vx_coord)
        else:
            return self.field_accessor.access_voxel(self.file_paths[file_idx], channel_name, layer_name, vx_coord)
        
    def _get_voxel_by_coord(self, file_idx: int, vx_coord: Union[Tuple[float, float, float], vec3], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel coordinate.
        :param file_idx: The index of the file in the dataset.
        :param vx_coord: The coordinate of the voxel in the radiation field in world space.
        :return: The voxel.
        """
        vx_coord = vx_coord if isinstance(vx_coord, vec3) else vec3(vx_coord[0], vx_coord[1], vx_coord[2])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel_by_coord(self.load_file_buffer(file_idx), channel_name, layer_name, vx_coord)
        else:
            return self.field_accessor.access_voxel_by_coord(self.file_paths[file_idx], channel_name, layer_name, vx_coord)


class CartesianFieldSingleLayerDataset(CartesianFieldDataset):
    """
    A dataset that loads single layers from a single channel of a radiation field as VoxelGrids.
    Useful, when only a single layer of a single channel is needed for training.
    """

    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        self.channel_name: str = None
        self.layer_name: str = None

    def set_channel_and_layer(self, channel_name: str, layer_name: str):
        self.channel_name = channel_name
        self.layer_name = layer_name

    def __getitem__(self, idx: int) -> Tuple[VoxelGrid, Union[RadiationFieldMetadata, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        layer = self._get_layer(idx, self.channel_name, self.layer_name)
        metadata = self._get_metadata(idx)
        return (self.transform(layer, idx), self.transform_origin(metadata, idx))


class CartesianFieldLayeredDataset(CartesianFieldDataset):
    """
    A dataset that loads all layers by name across all available channels of a radiation field as VoxelGrids.
    Useful, when a special layer that occurs in multiple channels is needed for training.
    To utilize this dataset class, the method transform must be implemented in a derived class.
    """

    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER, layer_name: str = None):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        self.layer_name: str = layer_name

    def set_layer(self, layer_name: str):
        self.layer_name = layer_name

    def __getitem__(self, idx: int) -> Tuple[dict[str, VoxelGrid], Union[RadiationFieldMetadata, None]]:
        assert self.layer_name is not None, "Layer must be set before loading the radiation field."
        if self.is_dataset_zipped:
            layers = self.field_accessor.access_layer_across_channels_from_buffer(self.load_file_buffer(idx), self.layer_name)
        else:
            layers = self.field_accessor.access_layer_across_channels(self.file_paths[idx], self.layer_name)
        metadata = self._get_metadata(idx)
        return (self.transform(layers, idx), self.transform_origin(metadata, idx))
    
    def transform(self, layers: dict[str, VoxelGrid], idx: int) -> Tensor:
        raise NotImplementedError("transform must be implemented in derived class.")


class CartesianSingleVoxelDataset(CartesianFieldSingleLayerDataset):
    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        field = self._get_field(0)
        self.field_voxel_counts = field.get_voxel_counts()
        self.voxels_per_field = self.field_voxel_counts.x * self.field_voxel_counts.y * self.field_voxel_counts.z
        self.zip_ref = None # remove zip reference to avoid pickling issues
        self._field_accessor = None # remove field accessor to avoid pickling issues

    def __len__(self) -> int:
        vx_count = int(self.field_accessor.get_voxel_count())
        self._field_accessor = None  # remove field accessor to avoid pickling issues
        return super().__len__() * vx_count
    
    def _get_field(self, idx: int) -> CartesianRadiationField:
        return super()._get_field(idx // self.field_accessor.get_voxel_count())
    
    def _get_metadata(self, idx) -> Union[RadiationFieldMetadata, None]:
        return super()._get_metadata(idx // self.field_accessor.get_voxel_count())

    def __getitem__(self, idx) -> Tuple[Tensor, Union[Tensor, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        vx_idx = idx % self.field_accessor.get_voxel_count()
        ret_val = (self._get_voxel_flat(idx // self.field_accessor.get_voxel_count(), vx_idx, self.channel_name, self.layer_name), self._get_metadata(idx // self.field_accessor.get_voxel_count()))
        return (self.transform(ret_val[0], idx), self.transform_origin(ret_val[1], idx))
