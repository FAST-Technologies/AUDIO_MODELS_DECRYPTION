from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
import librosa

from Constants import Constants
from .GraphicsModule_PyTorch import PyTorchGraphicsModule
MY_CONSTANTS = Constants()

# Инициализация PyTorch CTC модели
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

        PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=mel_filters,
                                                 title="Mel Filter Bank - TorchAudio (viridis)",
                                                 xlabel = "Frequency Bin",
                                                 ylabel = "Mel Filter Index",
                                                 colorbar_label = "Filter Weight",
                                                 cmap='viridis',
                                                 interpolation="spline16",
                                                 grid_flag=False)

        PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=mel_filters,
                                                 title="Mel Filter Bank - TorchAudio (jet)",
                                                 xlabel="Frequency Bin",
                                                 ylabel="Mel Filter Index",
                                                 colorbar_label="Filter Weight",
                                                 cmap='jet',
                                                 interpolation="bicubic",
                                                 grid_flag=False)

        PyTorchGraphicsModule.plot_fbank_HZ_PyTorch(mel_filters=mel_filters,
                                                    title="Mel Filter Hz - TorchAudio (viridis)",
                                                    sample_rate=MY_CONSTANTS.SAMPLE_RATE,
                                                    n_fft=MY_CONSTANTS.N_FFT_TEST,
                                                    xlabel = "Frequency (Hz)",
                                                    ylabel = "Mel Filter Index",
                                                    colorbar_label = "Filter Weight",
                                                    cmap='viridis',
                                                    interpolation="bicubic",
                                                    grid_flag=False)

        PyTorchGraphicsModule.plot_fbank_HZ_PyTorch(mel_filters=mel_filters,
                                                    title="Mel Filter Hz - TorchAudio (jet)",
                                                    sample_rate=MY_CONSTANTS.SAMPLE_RATE,
                                                    n_fft=MY_CONSTANTS.N_FFT_TEST,
                                                    xlabel="Frequency (Hz)",
                                                    ylabel="Mel Filter Index",
                                                    colorbar_label="Filter Weight",
                                                    cmap='jet',
                                                    interpolation="bicubic",
                                                    grid_flag=False)

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
        PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=torch.from_numpy(mel_filters_librosa),
                                                 title="Mel Filter Bank - Librosa (viridis)",
                                                 xlabel="Частотный диапазон",
                                                 ylabel="Индекс Мел Фильтра",
                                                 colorbar_label="Веса фильтра (Цветовой градиент)",
                                                 cmap='viridis',
                                                 interpolation="spline16",
                                                 grid_flag=False)

        PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=torch.from_numpy(mel_filters_librosa),
                                                 title="Mel Filter Bank - Librosa (jet)",
                                                 xlabel="Частотный диапазон",
                                                 ylabel="Индекс Мел Фильтра",
                                                 colorbar_label="Веса фильтра (Цветовой градиент)",
                                                 cmap='jet',
                                                 interpolation="bicubic",
                                                 grid_flag=False)

        PyTorchGraphicsModule.plot_fbank_HZ_PyTorch(mel_filters=torch.from_numpy(mel_filters_librosa),
                                                    title="Mel Filter Bank Hz - Librosa (viridis)",
                                                    sample_rate=MY_CONSTANTS.SAMPLE_RATE,
                                                    n_fft=MY_CONSTANTS.N_FFT_TEST,
                                                    xlabel="Частота (Hz)",
                                                    ylabel="Индекс Мел Фильтра",
                                                    colorbar_label="Веса фильтра (Цветовой градиент)",
                                                    cmap='viridis',
                                                    interpolation="bicubic",
                                                    grid_flag=False)

        PyTorchGraphicsModule.plot_fbank_HZ_PyTorch(mel_filters=torch.from_numpy(mel_filters_librosa),
                                                    title="Mel Filter Bank Hz - Librosa (jet)",
                                                    sample_rate=MY_CONSTANTS.SAMPLE_RATE,
                                                    n_fft=MY_CONSTANTS.N_FFT_TEST,
                                                    xlabel="Частота (Hz)",
                                                    ylabel="Индекс Мел Фильтра",
                                                    colorbar_label="Веса фильтра (Цветовой градиент)",
                                                    cmap='jet',
                                                    interpolation="bicubic",
                                                    grid_flag=False)

        # Находим MSE метрику между результатами на PyTorch и Librossa
        mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
        print(f"MSE between Torchaudio and Librosa: {mse}")

    def out_len(self,
                input_lengths: Tensor
    ) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length,
                                 rounding_mode="floor").add(1).long()

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
            window=torch.hann_window(self.win_length,
                                     device=input_signal.device),
            power=2.0,
            normalized=False,
            onesided=True,
            center=True,
            pad_mode="reflect",
            return_complex=False,
        )
        mel_spec = torch.matmul(spectrogram.transpose(-2, -1),
                                self.mel_fb).transpose(-2, -1)
        spec_scaler = torch.log(mel_spec.clamp_(MY_CONSTANTS.MIN_LOG_TOLERANCE,
                          MY_CONSTANTS.MAX_LOG_TOLERANCE))
        return spec_scaler, self.out_len(length)

