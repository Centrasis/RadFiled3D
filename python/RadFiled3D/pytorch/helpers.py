import torch
from RadFiled3D.RadFiled3D import PolarSegments, CartesianRadiationField, VoxelGrid, PolarRadiationField
from typing import Union


class RadiationFieldHelper:
    @staticmethod
    def load_tensor_from_field(radiation_field: Union[CartesianRadiationField, PolarRadiationField], channel_name: str, layer_name: str) -> torch.Tensor:
        """
        Load a layer of a channel from a RadiationField object into a PyTorch tensor.
        :param radiation_field: The RadiationField object to load the layer from.
        :param channel_name: The name of the channel to load the layer from.
        :param layer_name: The name of the layer to load.
        :return: The layer as a PyTorch tensor. The tensor will have the shape (c, x, y) or (c, x, y, z) depending on the field type (cartesian/polar) where c is the number of channels.
        """
        field_tensor = torch.tensor(radiation_field.get_channel(channel_name).get_layer_as_ndarray(layer_name).astype("float32"))
        field_tensor = field_tensor.permute(-1, *range(field_tensor.ndimension() - 1))
        return field_tensor
    
    @staticmethod
    def load_tensor_from_layer(layer: Union[VoxelGrid, PolarSegments]) -> torch.Tensor:
        """
        Load a VoxelGrid or PolarSegments object into a PyTorch tensor.
        :param voxel_grid: The VoxelGrid object to load.
        :return: The VoxelGrid as a PyTorch tensor. The tensor will have the shape (c, x, y) or (c, x, y, z) where c is the number of channels.
        """
        layer_tensor = torch.tensor(layer.get_as_ndarray().astype("float32"))
        layer_tensor = layer_tensor.permute(-1, *range(layer_tensor.ndimension() - 1))
        return layer_tensor
