import os
import sys
from typing import Tuple
from torch import Tensor
import torch.nn as nn
import torchaudio

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Constants import Constants
from .SpecScaler import SpecScaler
MY_CONSTANTS = Constants()

class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """
    def __init__(self,
                 sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
                 features: int = MY_CONSTANTS.FEAT_IN
    ) -> None:
        super().__init__()
        self.hop_length = sample_rate // 100
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=sample_rate // 40,
                win_length=sample_rate // 40,
                hop_length=self.hop_length,
                n_mels=features,
            ),
            SpecScaler(),
        )

    def out_len(self,
                input_lengths: Tensor
    ) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def forward(self,
                input_signal: Tensor,
                length: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)