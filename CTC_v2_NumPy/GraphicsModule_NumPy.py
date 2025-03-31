from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes

class NumpyGraphicsModule:
    def __init__(self) -> None:
        pass

    @classmethod
    def plot_fbank(self,
                   mel_filters: np.ndarray,
                   title: str
    )-> None:
        """Plot the filter bank.

        Args:
            mel_filters (Tensor): The filter bank to plot.
            title (str): The title of the plot.
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_filters, aspect="auto", origin="lower")
        plt.title(title)
        plt.xlabel("Frequency Bin")
        plt.ylabel("Mel Filter")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    """
    Функция для построения графика моноканала.
    """

    @classmethod
    def mono_graph_V2_0(self,
                        features: np.ndarray
    ) -> None:
        plt.figure(figsize=(11, 4))
        plt.imshow(features[0], aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Значение фичи')
        plt.xlabel('Временные кадры')
        plt.ylabel('Фичи')
        plt.title('Спектрограмма фич (первые 64 канала)')
        plt.show()

    """
    Функция для построения графика стереоканала (через subploats).
    """

    @classmethod
    def stereo_subploats_graph_V2_0(self,
                                    features: np.ndarray
    ) -> None:
        # Для примера: отображение первой фичи в батче (если features.shape = [1, 64, T])
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns
        # Левый канал
        im1 = axes[0].imshow(features[0][0], aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Левый канал')
        axes[0].set_xlabel('Временные кадры')
        axes[0].set_ylabel('Фичи')
        # Правый канал
        im2 = axes[1].imshow(features[0][1], aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Правый канал')
        axes[1].set_xlabel('Временные кадры')
        axes[1].set_ylabel('Фичи')
        fig.colorbar(im1, ax=axes, shrink=0.7, location='right', label='Значение фичи')
        fig.suptitle('Спектрограмма фич (первые 64 канала) (Subplots)', fontsize=14, y=0.98)
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0, right=0.75, wspace=0.1)
        plt.show()

    """
    Функция для построения графика стереоканала (через GridSpec).
    """

    @classmethod
    def stereo_gridspec_graph_V2_0(self,
                                   features: np.ndarray
    ) -> None:
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.2])  # 1 строка, 3 столбца, колорбар уже
        # Левый канал
        ax0 = plt.subplot(gs[0])
        im1 = ax0.imshow(features[0][0], aspect='auto', origin='lower', cmap='viridis')
        ax0.set_title('Левый канал')
        ax0.set_xlabel('Временные кадры')
        ax0.set_ylabel('Фичи')
        # Правый канал
        ax1 = plt.subplot(gs[1])
        im2 = ax1.imshow(features[0][1], aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('Правый канал')
        ax1.set_xlabel('Временные кадры')
        ax1.set_ylabel('Фичи')
        # Колорбар
        ax_cb = plt.subplot(gs[2])
        fig.colorbar(im1, cax=ax_cb, label='Значение фичи') #Передаём ось для колорбара
        # Общий заголовок
        fig.suptitle('Спектрограмма фич (первые 64 канала) (GridSpec)', fontsize=14, y=0.98)
        # Настройка расположения - убираем plt.tight_layout и plt.subplots_adjust
        gs.update(top=0.85, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
        plt.show()

    @classmethod
    def plot_waveform(self,
                      waveform: np.ndarray,
                      sr: int,
                      title: Optional[str] = "Waveform",
                      ax: Optional[Axes] = None
    )-> None:
        if not isinstance(waveform, np.ndarray):
            raise ValueError("Waveform должен быть PyTorch-тензором или NumPy-массивом")

        # Обрабатываем размерности
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # Добавляем канал: (T,) -> (1, T)

        num_channels, num_frames = waveform.shape
        time_axis = np.arange(0, num_frames) / sr  # Используем NumPy вместо torch.arange

        if ax is None:
            _, ax = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels))

        # Если один канал, рисуем просто
        if num_channels == 1:
            ax.plot(time_axis, waveform[0], linewidth=1)
            ax.grid(True)
            ax.set_xlim([0, time_axis[-1]])
            ax.set_title(title)
        else:
            for ch in range(num_channels):
                ax[ch].plot(time_axis, waveform[ch], linewidth=1)
                ax[ch].grid(True)
                ax[ch].set_xlim([0, time_axis[-1]])
                ax[ch].set_title(f"{title} - Channel {ch + 1}")

        plt.tight_layout()

    @classmethod
    def plot_spectrogram(self,
                         specgram: np.ndarray,
                         title: Optional[str] = None,
                         ylabel: str = "freq_bin",
                         ax: Optional[Axes] = None
    ) -> None:
        if not isinstance(specgram, np.ndarray):
            raise ValueError("specgram должен быть NumPy-массивом")

        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
        plt.tight_layout()

    @classmethod
    def plot_pitch(
                  self,
                  waveform: np.ndarray,
                  sr: float,
                  pitch: np.ndarray
    )-> None:
        if not isinstance(waveform, np.ndarray) or not isinstance(pitch, np.ndarray):
            raise ValueError("specgram должен быть NumPy-массивом")
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # Добавляем канал: (T,) -> (1, T)
        elif pitch.ndim == 1:
            pitch = pitch[np.newaxis, :]  # Добавляем канал: (T,) -> (1, T)

        figure, axis = plt.subplots(1, 1)
        axis.set_title("Pitch Feature")
        axis.grid(True)

        end_time = waveform.shape[1] / sr
        time_axis = torch.linspace(0, end_time, waveform.shape[1])
        axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

        axis2 = axis.twinx()
        time_axis = torch.linspace(0, end_time, pitch.shape[1])
        axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

        axis2.legend(loc=0)
        plt.tight_layout()