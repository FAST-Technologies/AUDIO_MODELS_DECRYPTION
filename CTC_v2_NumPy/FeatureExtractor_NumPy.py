import os
import sys
from typing import Tuple
import numpy as np

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Constants import Constants
from .SpecScaler_NumPy import SpecScaler_V2_0
MY_CONSTANTS = Constants()

class FeatureExtractor_V2_0:
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """
    def __init__(self,
                 sample_rate: int,
                 features: int
    ) -> None:
        self.sample_rate = sample_rate
        self.features = features
        self.hop_length = sample_rate // 100
        self.n_fft = sample_rate // 40
        self.win_length = sample_rate // 40

        self.mel_fb = self._create_mel_filterbank()
        self.spec_scaler = SpecScaler_V2_0()

    def _create_mel_filterbank(self) -> np.ndarray:
        # Простая реализация мел-фильтров (линейная аппроксимация)
        n_freqs = int(self.n_fft // 2 + 1)
        f_min, f_max = 0.0, self.sample_rate / 2.0
        mel_min = 1125.0 * np.log1p(f_min / 700.0)
        mel_max = 1125.0 * np.log1p(f_max / 700.0)
        mel_points = np.linspace(mel_min, mel_max, self.features + 2)
        freq_points = 700.0 * (np.expm1(mel_points / 1125.0))

        fb = np.zeros((self.features, n_freqs))
        freqs = np.linspace(0, f_max, n_freqs)
        for m in range(self.features):
            f_left = freq_points[m]
            f_center = freq_points[m + 1]
            f_right = freq_points[m + 2]
            for f in range(n_freqs):
                if f_left <= freqs[f] <= f_center:
                    fb[m, f] = (freqs[f] - f_left) / (f_center - f_left)
                elif f_center < freqs[f] <= f_right:
                    fb[m, f] = (f_right - freqs[f]) / (f_right - f_center)
        return fb

    def forward(self,
                 input_signal: np.ndarray,
                 length: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # input_signal: [B, channels, T] или [B, T]
        if input_signal.ndim == 2:
            input_signal = input_signal[:, np.newaxis, :]  # Добавляем канал

        batch_size, channels, time = input_signal.shape
        num_frames = (time - self.n_fft) // self.hop_length + 1

        # Окно Ханна
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.win_length) / (self.win_length - 1))

        # Вычисление STFT вручную
        spectrogram = []
        for b in range(batch_size):
            for c in range(channels):
                signal = input_signal[b, c]
                frames = []
                for i in range(num_frames):
                    start = i * self.hop_length
                    frame = signal[start:start + self.n_fft]
                    if len(frame) < self.n_fft:
                        frame = np.pad(frame, (0, self.n_fft - len(frame)))
                    frame = frame * window[:len(frame)]
                    fft = np.fft.rfft(frame, n=self.n_fft)
                    power = np.abs(fft) ** 2
                    frames.append(power)
                spectrogram.append(np.stack(frames, axis=0))
        spectrogram = np.stack(spectrogram, axis=0).reshape(batch_size, channels, num_frames, -1)
        # spectrogram = np.stack(spectrogram, axis=0)  # [B, channels, T, n_freqs]

        # Применение мел-фильтров
        mel_spec = np.matmul(spectrogram, self.mel_fb.T)  # [B, channels, T, n_mels]
        mel_spec = mel_spec.transpose(0, 1, 3, 2)  # [B, channels, n_mels, T]

        # Логарифмическое масштабирование
        mel_spec = self.spec_scaler(mel_spec)

        # Убираем лишнюю размерность для моно
        if channels == 1:
            mel_spec = mel_spec.squeeze(1)  # [B, n_mels, T]

        return mel_spec, self.out_len(length)

    def __call__(self,
                 input_signal: np.ndarray,
                 length: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Делает класс вызываемым, перенаправляя вызов на forward."""
        return self.forward(input_signal, length)

    def out_len(self,
                input_lengths: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the output length after the feature extraction process.
        """
        return (input_lengths // self.hop_length + 1).astype(np.int64)