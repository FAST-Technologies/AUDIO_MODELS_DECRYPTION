import os
import sys
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
import librosa

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Constants import Constants
from .SpecScaler_PyTorch import SpecScaler_PyTorch
from .GraphicsModule_PyTorch import PyTorchGraphicsModule
MY_CONSTANTS = Constants()

class FeatureExtractor_PyTorch(nn.Module):
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
        self.sample_rate = sample_rate
        self.features = features
        self.hop_length = sample_rate // 100
        self.n_fft = sample_rate // 40
        self.win_length = sample_rate // 40

        self.register_buffer(
                "mel_fb",
                torchaudio.functional.melscale_fbanks(
                n_freqs=int(self.n_fft // 2 + 1),
                f_min=0.0,
                f_max=self.sample_rate / 2.0,
                n_mels=self.features,
                sample_rate=self.sample_rate,
                norm="slaney"
        ))

        # Строим Mel Filter Blank с помощью библиотеки PyTorch
        mel_filters = torchaudio.functional.melscale_fbanks(
                  n_freqs=int(self.n_fft // 2 + 1),
                  f_min=0.0,
                  f_max=self.sample_rate / 2.0,
                  n_mels=self.features,
                  sample_rate=self.sample_rate,
                  norm="slaney"
                )
        PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=mel_filters, title="Mel Filter Bank - torchaudio")

        # Строим Mel Filter Blank с помощью библиотеки Librossa
        mel_filters_librosa = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.features,
            fmin=0.0,
            fmax=self.sample_rate / 2.0,
            norm="slaney",
            htk=True,
        ).T
        PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=torch.from_numpy(mel_filters_librosa), title="Mel Filter Bank - librosa")

        # Находим MSE метрику между результатами на PyTorch и Librossa
        mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
        print(f"MSE between torchaudio and librosa: {mse}")

        self.spec_scaler = SpecScaler_PyTorch()

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

        spectrogram = torchaudio.functional.spectrogram(
            waveform=input_signal,
            pad=0,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.win_length, device=input_signal.device),
            power=2.0,
            normalized=False,
            # onesided=True,
            # center=True,
            # pad_mode="reflect",
            # return_complex=False,
        )

        # mel_spec = torch.matmul(spectrogram.pow(2.0).transpose(-1, -2), self.mel_fb.to(spectrogram.device))
        mel_spec = torch.matmul(spectrogram.transpose(-2, -1), self.mel_fb).transpose(-2, -1)

        return self.spec_scaler(mel_spec), self.out_len(length)

