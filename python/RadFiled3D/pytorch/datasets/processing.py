from torch import nn
from RadFiled3D.pytorch.types import TrainingInputData
from typing import Any


class DataProcessing(nn.Module):
    """
    A module for processing training data from a RadField3D dataset.
    Takes an instance of TrainingInputData as input.
    The contained tensors shall be batched with batch dimension first.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: TrainingInputData) -> TrainingInputData:
        """
        Apply data processing to the input data.
        """
        ...

    def dataset_multiplier(self) -> float:
        """
        Get the dataset multiplier for the current processing module.
        Its used to increase the dataset size during training by applying data augmentation techniques.
        Default: 1.0
        """
        return 1.0

    def get_parameters(self) -> dict[str, Any]:
        """
        Get the parameters for the current processing module as a dictionary
        such that each key is the name of a parameter and each value is the value of the parameter.
        This way, the parameters can be easily logged.
        """
        return {}
