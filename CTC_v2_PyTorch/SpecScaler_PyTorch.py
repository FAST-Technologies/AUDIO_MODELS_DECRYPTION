import torch
from torch import Tensor
import torch.nn as nn

from Constants import Constants
MY_CONSTANTS = Constants()

class SpecScaler_PyTorch(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                x: Tensor
    ) -> Tensor:
        return torch.log(x.clamp_(MY_CONSTANTS.MIN_LOG_TOLERANCE,
                                  MY_CONSTANTS.MAX_LOG_TOLERANCE))
