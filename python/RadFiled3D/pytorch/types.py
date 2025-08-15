from typing import NamedTuple, Union
from torch import Tensor


class RadiationFieldChannel(NamedTuple):
    spectrum: Tensor
    fluence: Tensor
    error: Tensor = None


class RadiationField(NamedTuple):
    scatter_field: RadiationFieldChannel
    xray_beam: RadiationFieldChannel


class DirectionalInput(NamedTuple):
    direction: Tensor
    spectrum: Tensor
    geometry: Union[Tensor, None] = None


class PositionalInput(NamedTuple):
    direction: Tensor
    spectrum: Tensor
    position: Tensor
    geometry: Union[Tensor, None] = None


class TrainingInputData(NamedTuple):
    input: Union[DirectionalInput, PositionalInput]
    ground_truth: Union[RadiationField, RadiationFieldChannel]
