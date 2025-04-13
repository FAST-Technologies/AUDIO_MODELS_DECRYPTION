import os
import time
from typing import Optional
import numpy as np
import torch
from csvw.datatypes import boolean
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
import librosa
from scipy.ndimage import zoom

from GigaAM_to_ONNX.LoadClass import MY_CONSTANTS


class PyTorchGraphicsModule:
    def __init__(self) -> None:
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.graphics_dir = os.path.join(project_root, 'Graphics')
        try:
            if not os.path.exists(self.graphics_dir):
                os.makedirs(self.graphics_dir)
        except OSError as e:
            print(f"Ошибка при создании директории {self.graphics_dir}: {e}")
            raise

    @classmethod
    def save_to_directory(cls,
                          graphics_dir: str = 'Graphics'
    ) -> str:
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        graphics_dir = os.path.join(project_root, graphics_dir)
        try:
            if not os.path.exists(graphics_dir):
                os.makedirs(graphics_dir)
        except OSError as e:
            print(f"Ошибка при создании директории {graphics_dir}: {e}")
            raise
        return graphics_dir

    @classmethod
    def save_to_file(cls,
                     graphics_dir: str,
                     title: str
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(graphics_dir,
                                f'{title}_{timestamp}.png')
        plt.savefig(filename,
                    dpi=300,
                    bbox_inches='tight')
        if os.path.exists(filename):
            print(f"Выполнено успешное сохранение графика в файл: {filename}")
        else:
            print(f"Не удалось сохранить график в файл: {filename}")


    # Функция построения графика Mel Filter Blank
    @classmethod
    def plot_fbank_PyTorch(cls,
                   mel_filters: Tensor,
                   title: Optional[str] = "Filter Bank",
                   xlabel: Optional[str] = "Frequency Bin",
                   ylabel: Optional[str] = "Mel Filter Index",
                   colorbar_label: Optional[str] = "Filter Weight",
                   cmap: str = "viridis",
                   interpolation: str = "spline16",
                   grid_flag: bool = False
    ) -> None:
        """Plot the filter bank with a specified colormap and interpolation.

        Args:
            mel_filters (Tensor): The filter bank to plot, shape [n_freqs, n_mels].
            title (str): The title of the plot.
            sample_rate (int): Sample rate of the audio.
            n_fft (int): FFT size used to compute the filter bank.
            cmap (str): Colormap to use for visualization ('viridis_r', 'jet', etc.).
            interpolation (str): Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.).
            :param mel_filters:
            :param cmap:
            :param interpolation:
            :param grid_flag:
        """
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)

        if torch.is_tensor(mel_filters):
            mel_filters = mel_filters.numpy()

        # Нормализуем mel_filters, чтобы максимум был равен 1 (если это нужно)
        # mel_filters = mel_filters / mel_filters.max() if mel_filters.max() > 0 else mel_filters
        zoom_factor = 2  # Увеличиваем разрешение в 2 раза
        mel_filters = zoom(mel_filters,
                           zoom_factor,
                           order=3)

        # Количество частотных бинов (n_freqs)
        n_freqs, n_mels = mel_filters.shape

        # Проверка: соответствует ли n_freqs ожидаемому значению (n_fft // 2 + 1 для одностороннего спектра)
        expected_n_freqs = int(MY_CONSTANTS.N_FFT_TEST // 2 + 1) * zoom_factor
        if n_freqs != expected_n_freqs:
            raise ValueError(
                f"Number of frequency bins ({n_freqs}) does not match expected value ({expected_n_freqs}) "
                f"for n_fft={MY_CONSTANTS.N_FFT_TEST}. Expected n_freqs = n_fft // 2 + 1 for onesided=True."
            )

        # Логирование для отладки
        print(f"Shape of mel_filters after zoom: {mel_filters.shape}")
        print(f"Max value of mel_filters: {mel_filters.max()}")
        print(f"Using cmap for Filter Bank: {cmap}")
        print(f"Using interpolation for Filter Bank: {interpolation}")

        plt.figure(figsize=(10, 4))
        if grid_flag:
            plt.grid(True)
        else:
            plt.grid(False)
        im = plt.imshow(mel_filters,
                   aspect="auto",
                   origin="lower",
                   cmap=cmap,
                   interpolation=interpolation,
                   vmin=0,
                   vmax=0.025,
                   extent=(0, (MY_CONSTANTS.N_FFT_TEST // 2 + 1) - 1, 0, (n_mels // zoom_factor) - 1))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(im,
                     label=colorbar_label)
        plt.tight_layout()
        PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                           title='Plot_Filter_Bank_PyTorch')
        plt.show()
        plt.close()

    @classmethod
    def plot_fbank_HZ_PyTorch(cls,
                               mel_filters: Tensor,
                               sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
                               n_fft: int = MY_CONSTANTS.N_FFT_TEST,
                               title: Optional[str] = "Filter Bank (Hz)",
                               xlabel: Optional[str] = "Frequency (Hz)",
                               ylabel: Optional[str] = "Mel Filter Index",
                               colorbar_label: Optional[str] = "Filter Weight",
                               cmap: str = "viridis_r",
                               interpolation: str = "bicubic",
                               grid_flag: bool = False
                               ) -> None:
        """Plot the filter bank with a specified colormap and interpolation.

        Args:
            mel_filters (Tensor): The filter bank to plot, shape [n_freqs, n_mels].
            title (str): The title of the plot.
            sample_rate (int): Sample rate of the audio.
            n_fft (int): FFT size used to compute the filter bank.
            cmap (str): Colormap to use for visualization ('viridis_r', 'jet', etc.).
            interpolation (str): Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.).
        """
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)

        if torch.is_tensor(mel_filters):
            mel_filters = mel_filters.numpy()

        # Нормализуем mel_filters, чтобы максимум был равен 1 (если это нужно)
        mel_filters = mel_filters / mel_filters.max() if mel_filters.max() > 0 else mel_filters

        # Количество частотных бинов (n_freqs)
        n_freqs, n_mels = mel_filters.shape

        expected_n_freqs = n_fft // 2 + 1
        if n_freqs != expected_n_freqs:
            raise ValueError(
                f"Number of frequency bins ({n_freqs}) does not match expected value ({expected_n_freqs}) "
                f"for n_fft={MY_CONSTANTS.N_FFT_TEST}. Expected n_freqs = n_fft // 2 + 1 for onesided=True."
            )

        # Логирование для отладки
        print(f"Shape of mel_filters after zoom: {mel_filters.shape}")
        print(f"Max value of mel_filters: {mel_filters.max()}")
        print(f"Using cmap for Filter Bank (Hz): {cmap}")
        print(f"Using interpolation for Filter Bank (Hz): {interpolation}")

        # Вычисление частот в HZ
        freqs = np.linspace(0, sample_rate / 2, n_freqs)

        plt.figure(figsize=(10, 4))
        if grid_flag:
            plt.grid(True)
        else:
            plt.grid(False)
        im = plt.imshow(mel_filters,
                        aspect="auto",
                        origin="lower",
                        cmap=cmap,
                        interpolation=interpolation,
                        vmin=0,  # Минимальное значение для отображения
                        vmax=1,  # Максимальное значение для отображения
                        # extent=(freqs[0], freqs[-1], 0, mel_filters.shape[1] - 1)
                        extent=(0, n_freqs - 1, 0, n_mels - 1)) # extent для корректного масштабирования
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xticks_bins = np.linspace(0, n_freqs - 1, 5)
        xticks_freqs = np.linspace(freqs[0], freqs[-1], 5)
        plt.xticks(xticks_bins, [f"{int(f)}" for f in xticks_freqs])

        plt.colorbar(im,
                     label=colorbar_label)
        plt.tight_layout()
        PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                           title='Plot_Filter_Bank_HZ_PyTorch')
        plt.show()
        plt.close()
    """
    Функция для построения графика моноканала.
    """
    @classmethod
    def mono_graph_PyTorch(cls,
                           features: Tensor,
                           title: Optional[str] = 'Спектрограмма фич',
                           xlabel: Optional[str] = 'Временные кадры',
                           ylabel: Optional[str] = 'Фичи',
                           colorbar_label: Optional[str] = 'Значение фичи',
                           cmap: str = "viridis",
                           interpolation: str = "spline16",
                           grid_flag: bool = False
    ) -> None:
        global features_fix
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)

        print(f"Using cmap for Mono Graph: {cmap}")
        print(f"Using interpolation for Mono Graph: {interpolation}")
        print(f"Shape of features: {features.shape}")
        print(f"Max value of features: {features.max()}")
        print(f"Min value of features: {features.min()}")

        plt.figure(figsize=(11, 4))
        if features.ndim == 4:  # (batch_size, channels, n_mels, time_frames)
            features_fix = features[0, 0]  # Первый батч, первый канал
        elif features.ndim == 3:  # (channels, n_mels, time_frames)
            features_fix = features[0]  # Первый канал
        if grid_flag:
            plt.grid(True)
        else:
            plt.grid(False)
        plt.imshow(features_fix,
                   aspect='auto',
                   origin='lower',
                   cmap=cmap,
                   interpolation=interpolation)
        plt.colorbar(label=colorbar_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                           title='Mono_Graph_PyTorch')
        plt.show()
        plt.close()

    """
    Функция для построения графика стереоканала (через subploats).
    """

    @classmethod
    def stereo_subploats_graph_PyTorch(cls,
                                       features: Tensor,
                                       suptitle: Optional[str] = 'Feature spectrogram (First 64 Channels) (Subplots)',
                                       colorbar_label: Optional[str] = 'Feature Values',
                                       cmap: str = "viridis",
                                       interpolation: str = "spline16",
                                       language_type: str = "EN",
                                       grid_flag: bool = False
    ) -> None:
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)
        # Для примера: отображение первой фичи в батче (если features.shape = [1, 64, T])
        fig, axes = plt.subplots(1,
                                 2,
                                 figsize=(15, 6))  # 1 row, 2 columns

        if torch.is_tensor(features):
            features = features.numpy()

        print(f"Using cmap for Stereo_Subplots Graph: {cmap}")
        print(f"Using interpolation for Stereo_Subplots Graph: {interpolation}")
        print(f"Shape of features: {features.shape}")
        print("Data type:", features.dtype)
        print("Min value:", np.min(features))
        print("Max value:", np.max(features))

        # Логирование для отладки
        print(f"Max value of features[0][0]: {features[0][0].max()}")
        print(f"Min value of features[0][0]: {features[0][0].min()}")
        print(f"Max value of features[0][1]: {features[0][1].max()}")
        print(f"Min value of features[0][1]: {features[0][1].min()}")
        # Левый канал
        im1 = axes[0].imshow(features[0][0],
                             aspect='auto',
                             origin='lower',
                             cmap=cmap,
                             interpolation=interpolation,
                             label='Left Channel')
        if grid_flag:
            axes[0].grid(True)
        else:
            axes[0].grid(False)
        if language_type == "RU":
            axes[0].set_title('Левый канал')
            axes[0].set_xlabel('Временные кадры')
            axes[0].set_ylabel('Фичи')
        elif language_type == "EN":
            axes[0].set_title('Left Channel')
            axes[0].set_xlabel('Temporary Frames')
            axes[0].set_ylabel('Features')
        # Правый канал
        im2 = axes[1].imshow(features[0][1],
                             aspect='auto',
                             origin='lower',
                             cmap=cmap,
                             interpolation=interpolation,
                             label='Right Channel')
        if grid_flag:
            axes[1].grid(True)
        else:
            axes[1].grid(False)
        if language_type == "RU":
            axes[1].set_title('Правый канал')
            axes[1].set_xlabel('Временные кадры')
            axes[1].set_ylabel('Фичи')
        elif language_type == "EN":
            axes[1].set_title('Right Channel')
            axes[1].set_xlabel('Temporary Frames')
            axes[1].set_ylabel('Features')

        # Добавляем легенду для каждого подграфика
        for ax in axes:
            legend = ax.legend(loc="upper left")
            legend.set_zorder(10)  # Помещаем легенду на передний план
            legend.get_frame().set_alpha(0.8)  # Делаем фон легенды полупрозрачным
            legend.get_frame().set_facecolor('white')  # Белый фон легенды
            legend.get_frame().set_edgecolor('black')  # Чёрная рамка
        fig.colorbar(im1,
                     ax=axes,
                     location='right',
                     label=colorbar_label)
        fig.suptitle(suptitle,
                     fontsize=14,
                     y=0.98)
        plt.subplots_adjust(top=0.9,
                            bottom=0.15,
                            left=0.1,
                            right=0.78,
                            wspace=0.15)
        PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                           title='Stereo_Subplots_Graph_PyTorch')
        plt.show()
        plt.close()

    """
    Функция для построения графика стереоканала (через GridSpec).
    """

    @classmethod
    def stereo_gridspec_graph_PyTorch(cls,
                                      features: Tensor,
                                      suptitle: Optional[str] = 'Feature spectrogram (First 64 Channels) (GridSpec)',
                                      colorbar_label: Optional[str] = 'Feature Values',
                                      cmap: str = "viridis",
                                      interpolation: str = "spline16",
                                      language_type: str = "EN",
                                      grid_flag: bool = False
    ) -> None:
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)

        print(f"Using cmap for Stereo_GridSpec Graph: {cmap}")
        print(f"Using interpolation for Stereo_GridSpec Graph: {interpolation}")
        print(f"Shape of features: {features.shape}")
        print(f"Max value of features: {features.max()}")
        print(f"Min value of features: {features.min()}")

        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1,
                      3,
                      width_ratios=[1, 1, 0.2])  # 1 строка, 3 столбца, колорбар уже
        # Левый канал
        ax0 = plt.subplot(gs[0])
        if grid_flag:
            ax0.grid(True)
        else:
            ax0.grid(False)
        im1 = ax0.imshow(features[0][0],
                         aspect='auto',
                         origin='lower',
                         cmap=cmap,
                         interpolation=interpolation)
        if language_type == "RU":
            ax0.set_title('Левый канал')
            ax0.set_xlabel('Временные кадры')
            ax0.set_ylabel('Фичи')
        elif language_type == "EN":
            ax0.set_title('Left Channel')
            ax0.set_xlabel('Temporary Frames')
            ax0.set_ylabel('Features')
        # Правый канал
        ax1 = plt.subplot(gs[1])
        if grid_flag:
            ax1.grid(True)
        else:
            ax1.grid(False)
        im2 = ax1.imshow(features[0][1],
                         aspect='auto',
                         origin='lower',
                         cmap=cmap,
                         interpolation=interpolation)
        if language_type == "RU":
            ax1.set_title('Правый канал')
            ax1.set_xlabel('Временные кадры')
            ax1.set_ylabel('Фичи')
        elif language_type == "EN":
            ax1.set_title('Right Channel')
            ax1.set_xlabel('Temporary Frames')
            ax1.set_ylabel('Features')
        # Колорбар
        ax_cb = plt.subplot(gs[2])
        fig.colorbar(im1,
                     cax=ax_cb,
                     label=colorbar_label) #Передаём ось для колорбара
        # Общий заголовок
        fig.suptitle(suptitle,
                     fontsize=14,
                     y=0.98)
        # Настройка расположения - убираем plt.tight_layout и plt.subplots_adjust
        gs.update(top=0.85,
                  bottom=0.1,
                  left=0.05,
                  right=0.95,
                  wspace=0.3)
        PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                           title='Stereo_GridSpec_Graph_PyTorch')
        plt.show()
        plt.close()

    # Функция для построения WaveForm
    @classmethod
    def plot_waveform_PyTorch(cls,
                              waveform: torch.Tensor,
                              sr: int,
                              title: Optional[str] = "Waveform",
                              xlabel: Optional[str] = "Time (seconds)",
                              ylabel: Optional[str] = "Amplitude (normalized)",
                              flag: str = "CW",
                              ax: Optional[Axes] = None,
                              grid_flag: bool = False
    )-> None:
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)

        if torch.is_tensor(waveform):
            waveform = waveform.numpy()
        elif not isinstance(waveform, np.ndarray):
            raise ValueError("WaveForm должен быть PyTorch-тензором или NumPy-массивом")

        print(f"Shape of waveform: {waveform.shape}")
        print(f"Max value of waveform: {waveform.max()}")
        print(f"Min value of waveform: {waveform.min()}")

        # Обрабатываем размерности
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # Добавляем канал: (T,) -> (1, T)

        num_channels, num_frames = waveform.shape
        if sr != 0:
            time_axis = np.arange(0, num_frames) / sr  # Используем NumPy вместо torch.arange
        else:
            time_axis = 0.0

        if ax is None:
            _, ax = plt.subplots(num_channels,
                                 1,
                                 figsize=(10, 2 * num_channels))

        if num_channels == 1:
            ax.plot(time_axis,
                    waveform[0],
                    linewidth=1)
            if grid_flag:
                ax.grid(True)
            else:
                ax.grid(False)
            ax.set_xlim([0, time_axis[-1]])
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            for ch in range(num_channels):
                ax[ch].plot(time_axis,
                            waveform[ch],
                            linewidth=1)
                if grid_flag:
                    ax[ch].grid(True)
                else:
                    ax[ch].grid(False)
                ax[ch].set_xlim([0, time_axis[-1]])
                ax[ch].set_title(f"{title} - Channel {ch + 1}")
                ax[ch].set_xlabel(xlabel)
                ax[ch].set_ylabel(ylabel)

        plt.tight_layout()
        if flag == "CW":
            PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                               title='Plot_Classic_WaveForm_PyTorch')
        elif flag == "RW":
            PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                               title='Plot_Reconstructed_WaveForm_PyTorch')
        plt.show()
        plt.close()

    # Функция для построения спектрограммы
    @classmethod
    def plot_spectrogram_PyTorch(cls,
                                 specgram: torch.Tensor | np.ndarray,
                                 title: Optional[str] = None,
                                 xlabel: Optional[str] = "Frame Index",
                                 ylabel: Optional[str] = "freq_bin",
                                 colorbar_label: Optional[str] = "Spectrogram Colorbar",
                                 ax: Optional[Axes] = None,
                                 type: str = "MFCC",
                                 util_type: str = "TorchAudio",
                                 cmap: str = "viridis",
                                 interpolation: str = "nearest",
                                 grid_flag: bool = False
    ) -> None:
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        if torch.is_tensor(specgram):
            specgram = specgram.numpy()

        # Количество частотных бинов (n_freq_bins) и фреймов (n_frames)
        n_freq_bins, n_frames = specgram.shape

        print(f"Using cmap for Spectrogram Graph: {cmap}")
        print(f"Using interpolation for Spectrogram Graph: {interpolation}")
        # Логирование для отладки
        print(f"Shape of specgram: {specgram.shape}")
        print(f"Max value of specgram: {specgram.max()}")
        print(f"Min value of specgram: {specgram.min()}")

        im = ax.imshow(librosa.power_to_db(specgram),
                  origin="lower",
                  aspect="auto",
                  cmap=cmap,
                  interpolation=interpolation,
                  extent=(0, n_frames - 1, 0, n_freq_bins - 1))

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid_flag:
            ax.grid(True)
        else:
            ax.grid(False)
        plt.colorbar(im,
                     ax=ax,
                     label=colorbar_label)

        plt.tight_layout()
        if type == "MFCC":
            if util_type == "TorchAudio":
                PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                                   title='Plot_Spectrogram_MFCC_(TorchAudio)_PyTorch')
            elif util_type == "Librosa":
                PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                                   title='Plot_Spectrogram_MFCC_(Librosa)_PyTorch')
        elif type == "LFCC":
            if util_type == "TorchAudio":
                PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                                   title='Plot_Spectrogram_LFCC_(TorchAudio)_PyTorch')
            elif util_type == "Librosa":
                PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                                   title='Plot_Spectrogram_LFCC_(Librosa)_PyTorch')
        else:
            PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                               title='Plot_Classic_Spectrogram_PyTorch')
        plt.show()
        plt.close()

    # Функция для построения Pitch
    @classmethod
    def plot_pitch_PyTorch(cls,
                           waveform: torch.Tensor,
                           sr: float,
                           pitch: torch.Tensor,
                           title: Optional[str] = "Pitch Feature",
                           language_type: str = "EN",
                           grid_flag: bool = False
                           ) -> None:
        """Plot waveform and pitch on the same graph with dual y-axes.

        Args:
            waveform (Tensor): The waveform tensor, expected shape [batch, time].
            sr (float): Sampling rate of the waveform.
            pitch (Tensor): The pitch tensor, expected shape [batch, time].
            title (str, optional): Title of the plot.
            language_type (str): Language for labels ("EN" for English, "RU" for Russian).
            grid_flag (bool): Whether to display grid lines.
        """
        global axis1_plot_label, axis2_plot_label, axis1_x_label, axis2_x_label, axis1_y_label, axis2_y_label
        graphics_dir = 'Graphics'
        graphics_dir = PyTorchGraphicsModule.save_to_directory(graphics_dir=graphics_dir)

        try:
            # Проверка входных данных
            if not torch.is_tensor(waveform) or not torch.is_tensor(pitch):
                raise ValueError("Both 'waveform' and 'pitch' must be PyTorch tensors")

            # Преобразуем в NumPy для вычислений статистики
            waveform_np = waveform.numpy() if torch.is_tensor(waveform) else waveform
            pitch_np = pitch.numpy() if torch.is_tensor(pitch) else pitch

            # Проверка размерности
            if len(waveform.shape) != 2 or len(pitch.shape) != 2:
                raise ValueError(
                    f"Expected 2D tensors for waveform and pitch, got shapes {waveform.shape} and {pitch.shape}")

            if waveform.shape[0] < 1 or pitch.shape[0] < 1:
                raise ValueError("Batch size must be at least 1 for both waveform and pitch")

            if waveform.shape[1] != pitch.shape[1]:
                raise ValueError(
                    f"Time dimension mismatch: waveform has {waveform.shape[1]} samples, pitch has {pitch.shape[1]} samples")

            # Проверка частоты дискретизации
            if sr <= 0:
                raise ValueError(f"Sampling rate must be positive, got {sr}")

            # Расширенное логирование для waveform
            print("=== Waveform Statistics ===")
            print(f"Shape of waveform: {waveform.shape}")
            print(f"Data type: {waveform.dtype}")
            print(f"Min value: {np.min(waveform_np):.4f}")
            print(f"Max value: {np.max(waveform_np):.4f}")
            print(f"Mean value: {np.mean(waveform_np):.4f}")
            print(f"Std deviation: {np.std(waveform_np):.4f}")
            print(f"Median value: {np.median(waveform_np):.4f}")
            print(f"25th and 75th percentiles: {np.percentile(waveform_np, [25, 75])}")
            print(f"Number of non-zero elements: {np.count_nonzero(waveform_np)}")
            print(f"Sum of all elements: {np.sum(waveform_np):.4f}")
            print(f"Number of NaN values: {np.isnan(waveform_np).sum()}")
            print(f"Number of infinite values: {np.isinf(waveform_np).sum()}")

            # Расширенное логирование для pitch
            print("=== Pitch Statistics ===")
            print(f"Shape of pitch: {pitch.shape}")
            print(f"Data type: {pitch.dtype}")
            print(f"Min value: {np.min(pitch_np):.4f}")
            print(f"Max value: {np.max(pitch_np):.4f}")
            print(f"Mean value: {np.mean(pitch_np):.4f}")
            print(f"Std deviation: {np.std(pitch_np):.4f}")
            print(f"Median value: {np.median(pitch_np):.4f}")
            print(f"25th and 75th percentiles: {np.percentile(pitch_np, [25, 75])}")
            print(f"Number of non-zero elements: {np.count_nonzero(pitch_np)}")
            print(f"Sum of all elements: {np.sum(pitch_np):.4f}")
            print(f"Number of NaN values: {np.isnan(pitch_np).sum()}")
            print(f"Number of infinite values: {np.isinf(pitch_np).sum()}")

            # Статистика по каналам (если batch > 1)
            for i in range(waveform.shape[0]):
                channel_waveform = waveform_np[i]
                channel_pitch = pitch_np[i]
                print(
                    f"Waveform Channel {i + 1}: min={np.min(channel_waveform):.4f}, max={np.max(channel_waveform):.4f}, mean={np.mean(channel_waveform):.4f}")
                print(
                    f"Pitch Channel {i + 1}: min={np.min(channel_pitch):.4f}, max={np.max(channel_pitch):.4f}, mean={np.mean(channel_pitch):.4f}")

            # Создание графика
            figure, axis = plt.subplots(1, 1)
            axis.set_title(title)
            if grid_flag:
                axis.grid(True)
            else:
                axis.grid(False)

            # Вычисление временной оси
            end_time = waveform.shape[1] / sr
            time_axis = torch.linspace(0, end_time, waveform.shape[1])

            # Установка меток в зависимости от языка
            if language_type == "RU":
                axis1_plot_label = "Волновая форма"
                axis2_plot_label = "Питч"
                axis1_x_label = "Время (секунды) [s]"
                axis2_x_label = "Время (секунды) [s]"
                axis1_y_label = "Частота (Амплитуда) [Hz]"
                axis2_y_label = "Питч [Hz]"
            elif language_type == "EN":
                axis1_plot_label = "WaveForm"
                axis2_plot_label = "Pitch"
                axis1_x_label = "Time [s]"
                axis2_x_label = "Time [s]"
                axis1_y_label = "Frequency(Amplitude) [Hz]"
                axis2_y_label = "Pitch [Hz]"

            # Построение waveform
            axis.plot(time_axis,
                      waveform[0],
                      linewidth=1,
                      color="gray",
                      alpha=0.3,
                      label=axis1_plot_label)
            axis.set_xlabel(axis1_x_label)
            axis.set_ylabel(axis1_y_label, color="gray")

            # Построение pitch на второй оси
            axis2 = axis.twinx()
            time_axis = torch.linspace(0, end_time, pitch.shape[1])
            axis2.plot(time_axis,
                       pitch[0],
                       linewidth=2,
                       label=axis2_plot_label,
                       color="green")
            axis2.set_xlabel(axis2_x_label)
            axis2.set_ylabel(axis2_y_label, color="green")

            # Добавление легенд
            legend = axis.legend(loc="upper left")
            legend.set_zorder(10)
            legend2 = axis2.legend(loc="upper right")
            legend2.set_zorder(10)

            plt.tight_layout()
            PyTorchGraphicsModule.save_to_file(graphics_dir=graphics_dir,
                                               title='Plot_Pitch_PyTorch')
            plt.show()
            plt.close()

        except Exception as e:
            print(f"Error while plotting pitch: {str(e)}")

