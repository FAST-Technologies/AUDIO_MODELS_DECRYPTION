import os
import time
import logging
from typing import Optional
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
import librosa
from scipy.ndimage import zoom
from scipy.interpolate import interp1d

from Constants import Constants, VALID_CMAPS, VALID_INTERPOLATIONS, VALID_LANGUAGES, FORMATS_IMG

MY_CONSTANTS = Constants()
from HelpFunction_PyTorch import print_statistic_data_PyTorch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PyTorchGraphicsModule:
    """
    A class for visualizing audio-related data such as filter banks, spectrograms, waveforms, and pitch using PyTorch Library.

    Attributes
    ----------
    graphics_dir : str
        Directory where the generated plots will be saved.
    """
    def __init__(self,
                 graphics_dir: str = 'Graphics'
    ) -> None:
        """
        Initialize the PyTorchGraphicsModule by setting up the graphics directory.

        Parameters
        ----------
        graphics_dir : str, optional
            The name of the directory to create for saving graphics. Defaults to 'Graphics'.

        Raises
        ------
        OSError
            If the graphics directory cannot be created or there are insufficient permissions.
        ValueError
            If `graphics_dir` is not a valid string.
        """
        if not isinstance(graphics_dir, str) or not graphics_dir:
            raise ValueError(f"Graphics directory must be a non-empty valid string, got {type(graphics_dir)} {graphics_dir}.")
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        if not os.access(project_root, os.W_OK):
            raise OSError(f"No write permissions for directory {project_root}")

        self.graphics_dir = os.path.join(project_root, graphics_dir)
        try:
            if not os.path.exists(self.graphics_dir):
                os.makedirs(self.graphics_dir)
                logger.info(f"Created graphics directory {self.graphics_dir}")
        except OSError as e:
            logger.error(f"Failed to create directory {self.graphics_dir}: {e}")
            raise

        plt.style.use('seaborn') # А НУЖНО ЛИ

    @classmethod
    def save_to_directory(cls,
                          graphics_dir: str = 'Graphics',
                          clear_if_exists: bool = True,
    ) -> str:
        """
        Create a directory for saving graphics if it does not exist.

        Parameters
        ----------
        graphics_dir : str, optional
            The name of the directory to create. Defaults to 'Graphics'.
        clear_if_exists : bool, optional
            If True, clear the directory if it already exists. Defaults to False.

        Returns
        -------
        str
            The full path to the created directory.

        Raises
        ------
        ValueError
            If `graphics_dir` is not a valid string.
        OSError
            If the directory cannot be created.
        """
        if not isinstance(graphics_dir, str) or not graphics_dir:
            raise ValueError(f"graphics_dir must be a non-empty string, got {graphics_dir}")

        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        graphics_dir_path = os.path.join(project_root, graphics_dir)
        try:
            # if os.path.exists(graphics_dir_path) and clear_if_exists:
            #     for file in os.listdir(graphics_dir_path):
            #         file_path = os.path.join(graphics_dir_path, file)
            #         if os.path.isfile(file_path):
            #             os.remove(file_path)
            #     logger.info(f"Cleared existing directory: {graphics_dir_path}")
            if not os.path.exists(graphics_dir_path):
                os.makedirs(graphics_dir_path)
        except OSError as e:
            logger.error(f"Failed to create or clear directory {graphics_dir_path}: {e}")
            raise
        return graphics_dir_path

    @classmethod
    def save_to_file(cls,
                     graphics_dir: str,
                     title: str,
                     format: str = "png"
    ) -> None:
        """
        Save the current matplotlib figure to a file with a timestamp.

        Parameters
        ----------
        graphics_dir : str
            The directory where the plot will be saved.
        title : str
            The base name of the file (a timestamp will be appended).
        format : str, optional
            The format of the file ('png', 'jpg', etc.). Defaults to 'png'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `graphics_dir` does not exist or `format` is invalid.
        IOError
            If the file cannot be saved.
        """
        if not os.path.exists(graphics_dir):
            raise ValueError(f"Graphics directory {graphics_dir} does not exist.")
        if format not in FORMATS_IMG:
            raise ValueError(f"Unsupported format {format}, supported formats: {' | '.join(FORMATS_IMG)}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(graphics_dir,
                                f'{title}_{timestamp}.{format}')
        try:
            plt.savefig(filename,
                        dpi=300,
                        bbox_inches='tight',
                        format=format)
            if os.path.exists(filename):
                logger.info(f"Successfully saved plot to file: {filename}")
            else:
                logger.warning(f"Failed to save plot to file: {filename}")
        except IOError as e:
            logger.error(f"Failed to save figure to {filename}: {e}")
            raise

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
                   grid_flag: bool = False,
                   normalize: bool = False,
                   vmin: float = 0.0,
                   vmax: float = 0.025,
    ) -> None:
        """
        Plot the Mel filter bank with a specified colormap and interpolation.

        Parameters
        ----------
        mel_filters : Tensor
            The filter bank to plot, expected shape [n_freqs, n_mels].
        title : Optional[str], optional
            The title of the plot. Defaults to "Filter Bank".
        xlabel : Optional[str], optional
            Label for the x-axis. Defaults to "Frequency Bin".
        ylabel : Optional[str], optional
            Label for the y-axis. Defaults to "Mel Filter Index".
        colorbar_label : Optional[str], optional
            Label for the colorbar. Defaults to "Filter Weight".
        cmap : str, optional
            Colormap to use for visualization ('viridis', 'jet', etc.). Defaults to "viridis".
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.). Defaults to "spline16".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        normalize : bool, optional
            If True, normalize `mel_filters` to have a maximum value of 1. Defaults to False.
        vmin : float, optional
            Minimum value for the colormap. Defaults to 0.0.
        vmax : float, optional
            Maximum value for the colormap. Defaults to 0.025.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the shape of `mel_filters` does not match the expected number of frequency bins,
            or if `cmap` or `interpolation` are invalid.
        """
        if not torch.is_tensor(mel_filters):
            raise ValueError("Mel_filters must be a PyTorch Tensor.")
        if len(mel_filters.shape) != 2:
            raise ValueError(f"Mel filters must be a 2D tensor, got shape: {mel_filters.shape}.")
        if cmap not in VALID_CMAPS:
            raise ValueError(f"Invalid cmap {cmap}, must be one of {VALID_CMAPS}")
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(f"Invalid interpolation {interpolation}, must be one of {VALID_INTERPOLATIONS}")

        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
        mel_filters = mel_filters.numpy()

        zoom_factor = 2  # Увеличиваем разрешение в 2 раза
        mel_filters = zoom(mel_filters,
                           zoom_factor,
                           order=3)

        if normalize:
            mel_filters = mel_filters / mel_filters.max() if mel_filters.max() > 0 else mel_filters

        # Количество частотных бинов (n_freqs)
        n_freqs, n_mels = mel_filters.shape

        # Проверка: соответствует ли n_freqs ожидаемому значению (n_fft // 2 + 1 для одностороннего спектра)
        expected_n_freqs = int(MY_CONSTANTS.N_FFT_TEST // 2 + 1) * zoom_factor
        if n_freqs != expected_n_freqs:
            raise ValueError(
                f"Number of frequency bins ({n_freqs}) does not match expected value ({expected_n_freqs}) "
                f"for n_fft={MY_CONSTANTS.N_FFT_TEST}. Expected n_freqs = n_fft // 2 + 1 for onesided=True."
            )

        print_statistic_data_PyTorch(features=torch.tensor(mel_filters),
                                     stat_param="NumPy")
        logger.info(f"Plotting filter bank with shape {mel_filters.shape}, cmap={cmap}, interpolation={interpolation}")

        fig, ax = plt.subplots(figsize=(10, 4))
        if grid_flag:
            ax.grid(True)
        else:
            ax.grid(False)
        im = plt.imshow(mel_filters,
                   aspect="auto",
                   origin="lower",
                   cmap=cmap,
                   interpolation=interpolation,
                   vmin=vmin,
                   vmax=vmax,
                   extent=(0, (MY_CONSTANTS.N_FFT_TEST // 2 + 1) - 1, 0, (n_mels // zoom_factor) - 1))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(im,
                     label=colorbar_label)
        plt.tight_layout()
        cls.save_to_file(graphics_dir=graphics_dir,
                         title='Plot_Filter_Bank_PyTorch')
        plt.show()
        plt.close(fig)

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
                               grid_flag: bool = False,
                               num_xticks: int = 5
                               ) -> None:
        """
        Plot the Mel filter bank with frequency axis in Hz.

        Parameters
        ----------
        mel_filters : Tensor
            The filter bank to plot, expected shape [n_freqs, n_mels].
        sample_rate : int, optional
            Sample rate of the audio. Defaults to MY_CONSTANTS.SAMPLE_RATE.
        n_fft : int, optional
            FFT size used to compute the filter bank. Defaults to MY_CONSTANTS.N_FFT_TEST.
        title : Optional[str], optional
            The title of the plot. Defaults to "Filter Bank (Hz)".
        xlabel : Optional[str], optional
            Label for the x-axis. Defaults to "Frequency (Hz)".
        ylabel : Optional[str], optional
            Label for the y-axis. Defaults to "Mel Filter Index".
        colorbar_label : Optional[str], optional
            Label for the colorbar. Defaults to "Filter Weight".
        cmap : str, optional
            Colormap to use for visualization ('viridis_r', 'jet', etc.). Defaults to "viridis_r".
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.). Defaults to "bicubic".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        num_xticks : int, optional
            Number of ticks on the x-axis for frequency labels. Defaults to 5.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the shape of `mel_filters` does not match the expected number of frequency bins,
            or if `sample_rate`, `n_fft`, `num_xticks`, `cmap`, or `interpolation` are invalid.
        """
        if not torch.is_tensor(mel_filters):
            raise ValueError("mel_filters must be a PyTorch tensor")
        if len(mel_filters.shape) != 2:
            raise ValueError(f"Expected 2D tensor for mel_filters, got shape {mel_filters.shape}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {n_fft}")
        if num_xticks <= 0:
            raise ValueError(f"num_xticks must be positive, got {num_xticks}")
        if cmap not in VALID_CMAPS:
            raise ValueError(f"Invalid cmap {cmap}, must be one of {VALID_CMAPS}")
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(f"Invalid interpolation {interpolation}, must be one of {VALID_INTERPOLATIONS}")
        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
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
        print_statistic_data_PyTorch(features=torch.tensor(mel_filters),
                                     stat_param="NumPy")
        logger.info(f"Plotting filter bank (Hz) with shape {mel_filters.shape}, cmap={cmap}, interpolation={interpolation}")

        # Вычисление частот в HZ
        freqs = np.linspace(0, sample_rate / 2, n_freqs)

        fig, ax = plt.subplots(figsize=(10, 4))
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
                        extent=(freqs[0], freqs[-1], 0, mel_filters.shape[1] - 1)
                        ) # extent для корректного масштабирования
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # xticks_bins = np.linspace(0, n_freqs - 1, 5)
        xticks_freqs = np.linspace(freqs[0], freqs[-1], num_xticks)
        plt.xticks(xticks_freqs, [f"{int(f)}" for f in xticks_freqs])

        plt.colorbar(im,
                     label=colorbar_label)
        plt.tight_layout()
        cls.save_to_file(graphics_dir=graphics_dir,
                         title='Plot_Filter_Bank_HZ_PyTorch')
        plt.show()
        plt.close(fig)
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
                           grid_flag: bool = False,
                           normalize: bool = False
    ) -> None:
        """
        Plot a spectrogram for a single (mono) channel.

        Parameters
        ----------
        features : Tensor
            The features to plot, expected shape [batch, channels, n_mels, time_frames] or [channels, n_mels, time_frames].
        title : Optional[str], optional
            The title of the plot. Defaults to 'Спектрограмма фич'.
        xlabel : Optional[str], optional
            Label for the x-axis. Defaults to 'Временные кадры'.
        ylabel : Optional[str], optional
            Label for the y-axis. Defaults to 'Фичи'.
        colorbar_label : Optional[str], optional
            Label for the colorbar. Defaults to 'Значение фичи'.
        cmap : str, optional
            Colormap to use for visualization ('viridis', 'jet', etc.). Defaults to "viridis".
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.). Defaults to "spline16".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        normalize : bool, optional
            If True, normalize `features` to have a maximum value of 1. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `features` does not have 3D or 4D shape, or if `cmap` or `interpolation` are invalid.
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        elif not torch.is_tensor(features):
            raise ValueError("features must be a PyTorch tensor or numpy.ndarray")
        if features.ndim not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor for features, got shape {features.shape}")
        if cmap not in VALID_CMAPS:
            raise ValueError(f"Invalid cmap {cmap}, must be one of {VALID_CMAPS}")
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(f"Invalid interpolation {interpolation}, must be one of {VALID_INTERPOLATIONS}")

        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
        logger.info(f"Plotting mono graph with shape {features.shape}, cmap={cmap}, interpolation={interpolation}")
        print_statistic_data_PyTorch(features=features,
                                     stat_param="NumPy")

        features_fix = features[0, 0] if features.ndim == 4 else features[0]
        if normalize:
            features_fix = features_fix / features_fix.max() if features_fix.max() > 0 else features_fix

        features_fix_np = features_fix.cpu().numpy()
        fig, ax = plt.subplots(figsize=(11, 4))
        if grid_flag:
            ax.grid(True)
        else:
            ax.grid(False)
        im = ax.imshow(features_fix_np,
                   aspect='auto',
                   origin='lower',
                   cmap=cmap,
                   interpolation=interpolation)
        plt.colorbar(im,
                     ax=ax,
                     label=colorbar_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        cls.save_to_file(graphics_dir=graphics_dir,
                         title='Mono_Graph_PyTorch')
        plt.show()
        plt.close(fig)

    """
    Функция для построения графика стереоканала (через subploats).
    """

    @classmethod
    def _plot_stereo_common(cls,
                            features: torch.Tensor,
                            suptitle: Optional[str],
                            colorbar_label: Optional[str],
                            cmap: str,
                            interpolation: str,
                            language_type: str,
                            grid_flag: bool,
                            title_fontsize: int,
                            label_fontsize: int,
                            use_gridspec: bool = False,
                            width_ratios: Optional[list] = None,
                            graphics_dir: str = 'Graphics',
                            save_title: str = 'Stereo_Graph'
                            ) -> None:
        """
        Common method for plotting stereo spectrograms using either subplots or GridSpec.

        Parameters
        ----------
        features : Tensor
            The features to plot, expected shape [batch, channels, n_mels, time_frames].
        suptitle : Optional[str]
            The overall title of the plot.
        colorbar_label : Optional[str]
            Label for the colorbar.
        cmap : str
            Colormap to use for visualization.
        interpolation : str
            Interpolation method for imshow.
        language_type : str
            Language for labels ("EN" for English, "RU" for Russian).
        grid_flag : bool
            Whether to display grid lines.
        title_fontsize : int
            Font size for subplot titles.
        label_fontsize : int
            Font size for axis labels.
        use_gridspec : bool, optional
            If True, use GridSpec for layout. Defaults to False.
        width_ratios : Optional[list], optional
            Width ratios for GridSpec layout. Defaults to None.

        Returns
        -------
        None
        """
        global gs, ax_cb
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        elif not torch.is_tensor(features):
            raise ValueError("features must be a PyTorch tensor or numpy.ndarray")
        if features.shape[1] < 2:
            raise ValueError(f"Expected at least 2 channels for stereo plotting, got {features.shape[1]}")
        if language_type not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language_type {language_type}, must be one of {VALID_LANGUAGES}")

        logger.info(f"Plotting stereo graph with shape {features.shape}, cmap={cmap}, interpolation={interpolation}")
        print_statistic_data_PyTorch(features=features,
                                     stat_param="NumPy")

        features_np = features.cpu().numpy()

        if use_gridspec:
            width_ratios = width_ratios or [1, 1, 0.2]
            fig = plt.figure(figsize=(15, 6))
            gs = GridSpec(1,
                          3,
                          width_ratios=width_ratios)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax_cb = plt.subplot(gs[2])
        else:
            fig, (ax0, ax1) = plt.subplots(1,
                                           2,
                                           figsize=(15, 6))

        # Левый канал
        if grid_flag:
            ax0.grid(True)
        else:
            ax0.grid(False)
        im1 = ax0.imshow(features_np[0][0],
                         aspect='auto',
                         origin='lower',
                         cmap=cmap,
                         interpolation=interpolation,
                         label='Left Channel')
        if language_type == "RU":
            ax0.set_title('Левый канал',
                          fontsize=title_fontsize)
            ax0.set_xlabel('Временные кадры',
                           fontsize=label_fontsize)
            ax0.set_ylabel('Фичи',
                           fontsize=label_fontsize)
        else:
            ax0.set_title('Left Channel',
                          fontsize=title_fontsize)
            ax0.set_xlabel('Temporary Frames',
                           fontsize=label_fontsize)
            ax0.set_ylabel('Features',
                           fontsize=label_fontsize)

        # Правый канал
        if grid_flag:
            ax1.grid(True)
        else:
            ax1.grid(False)
        im2 = ax1.imshow(features_np[0][1],
                         aspect='auto',
                         origin='lower',
                         cmap=cmap,
                         interpolation=interpolation,
                         label='Right Channel')
        if language_type == "RU":
            ax1.set_title('Правый канал',
                          fontsize=title_fontsize)
            ax1.set_xlabel('Временные кадры',
                           fontsize=label_fontsize)
            ax1.set_ylabel('Фичи',
                           fontsize=label_fontsize)
        else:
            ax1.set_title('Right Channel',
                          fontsize=title_fontsize)
            ax1.set_xlabel('Temporary Frames',
                           fontsize=label_fontsize)
            ax1.set_ylabel('Features',
                           fontsize=label_fontsize)

        # Колорбар и настройки
        if use_gridspec:
            fig.colorbar(im1,
                         cax=ax_cb,
                         label=colorbar_label)
            gs.update(top=0.85,
                      bottom=0.1,
                      left=0.05,
                      right=0.95,
                      wspace=0.3)
        else:
            fig.colorbar(im1,
                         ax=[ax0, ax1],
                         location='right',
                         label=colorbar_label)
            plt.subplots_adjust(top=0.9,
                                bottom=0.15,
                                left=0.1,
                                right=0.78,
                                wspace=0.15)

        fig.suptitle(suptitle,
                     fontsize=14,
                     y=0.98)
        cls.save_to_file(graphics_dir=graphics_dir,
                         title=save_title)
        plt.show()
        plt.close(fig)

    @classmethod
    def stereo_subplots_graph_PyTorch(cls,
                                       features: Tensor,
                                       suptitle: Optional[str] = 'Feature spectrogram (First 64 Channels) (Subplots)',
                                       colorbar_label: Optional[str] = 'Feature Values',
                                       cmap: str = "viridis",
                                       interpolation: str = "spline16",
                                       language_type: str = "EN",
                                       grid_flag: bool = False,
                                       title_fontsize: int = 12,
                                       label_fontsize: int = 10
    ) -> None:
        """
        Plot a stereo spectrogram using subplots for left and right channels.

        Parameters
        ----------
        features : Tensor
            The features to plot, expected shape [batch, channels, n_mels, time_frames].
        suptitle : Optional[str], optional
            The overall title of the plot. Defaults to 'Feature spectrogram (First 64 Channels) (Subplots)'.
        colorbar_label : Optional[str], optional
            Label for the colorbar. Defaults to 'Feature Values'.
        cmap : str, optional
            Colormap to use for visualization ('viridis', 'jet', etc.). Defaults to "viridis".
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.). Defaults to "spline16".
        language_type : str, optional
            Language for labels ("EN" for English, "RU" for Russian). Defaults to "EN".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        title_fontsize : int, optional
            Font size for subplot titles. Defaults to 12.
        label_fontsize : int, optional
            Font size for axis labels. Defaults to 10.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `features` has fewer than 2 channels, or if `cmap`, `interpolation`, or `language_type` are invalid.
        """
        if cmap not in VALID_CMAPS:
            raise ValueError(f"Invalid cmap {cmap}, must be one of {VALID_CMAPS}")
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(f"Invalid interpolation {interpolation}, must be one of {VALID_INTERPOLATIONS}")

        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
        cls._plot_stereo_common(features,
                                suptitle,
                                colorbar_label,
                                cmap,
                                interpolation,
                                language_type,
                                grid_flag,
                                title_fontsize,
                                label_fontsize,
                                use_gridspec=False,
                                graphics_dir=graphics_dir,
                                save_title='Stereo_Subplots_Graph_PyTorch')



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
                                      grid_flag: bool = False,
                                      title_fontsize: int = 12,
                                      label_fontsize: int = 10,
                                      width_ratios: Optional[list] = None
    ) -> None:
        """
        Plot a stereo spectrogram using GridSpec for left and right channels.

        Parameters
        ----------
        features : Tensor
            The features to plot, expected shape [batch, channels, n_mels, time_frames].
        suptitle : Optional[str], optional
            The overall title of the plot. Defaults to 'Feature spectrogram (First 64 Channels) (GridSpec)'.
        colorbar_label : Optional[str], optional
            Label for the colorbar. Defaults to 'Feature Values'.
        cmap : str, optional
            Colormap to use for visualization ('viridis', 'jet', etc.). Defaults to "viridis".
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.). Defaults to "spline16".
        language_type : str, optional
            Language for labels ("EN" for English, "RU" for Russian). Defaults to "EN".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        title_fontsize : int, optional
            Font size for subplot titles. Defaults to 12.
        label_fontsize : int, optional
            Font size for axis labels. Defaults to 10.
        width_ratios : Optional[list], optional
            Width ratios for GridSpec layout. Defaults to [1, 1, 0.2].

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `features` has fewer than 2 channels, or if `cmap`, `interpolation`, or `language_type` are invalid.
        """
        if cmap not in VALID_CMAPS:
            raise ValueError(f"Invalid cmap {cmap}, must be one of {VALID_CMAPS}")
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(f"Invalid interpolation {interpolation}, must be one of {VALID_INTERPOLATIONS}")
        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')

        cls._plot_stereo_common(features,
                                suptitle,
                                colorbar_label,
                                cmap,
                                interpolation,
                                language_type,
                                grid_flag,
                                title_fontsize,
                                label_fontsize,
                                use_gridspec=True,
                                width_ratios=width_ratios,
                                graphics_dir=graphics_dir,
                                save_title='Stereo_GridSpec_Graph_PyTorch')


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
                              grid_flag: bool = False,
                              line_color: str = "blue",
                              linewidth: float = 1.0
                              ) -> None:
        """
        Plot the waveform of an audio signal.

        Parameters
        ----------
        waveform : torch.Tensor
            The waveform tensor, expected shape [channels, time] or [time].
        sr : int
            Sampling rate of the waveform.
        title : Optional[str], optional
            The title of the plot. Defaults to "Waveform".
        xlabel : Optional[str], optional
            Label for the x-axis. Defaults to "Time (seconds)".
        ylabel : Optional[str], optional
            Label for the y-axis. Defaults to "Amplitude (normalized)".
        flag : str, optional
            Type of waveform plot ("CW" for classic, "RW" for reconstructed). Defaults to "CW".
        ax : Optional[Axes], optional
            Matplotlib axes to plot on. If None, a new figure is created. Defaults to None.
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        line_color : str, optional
            Color of the waveform line. Defaults to "blue".
        linewidth : float, optional
            Width of the waveform line. Defaults to 1.0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `waveform` is not a PyTorch tensor, or if `sr` is not positive.
        TypeError
            If `ax` is provided but does not match the number of channels in `waveform`.
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        elif not torch.is_tensor(waveform):
            raise ValueError("features must be a PyTorch tensor or numpy.ndarray")
        if sr <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sr}")

        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
        print_statistic_data_PyTorch(features=waveform,
                                     stat_param="NumPy")

        # Обрабатываем размерности
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Добавляем канал: (T,) -> (1, T)

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0,
                                 num_frames,
                                 device=waveform.device,
                                 dtype=torch.float32) / sr
        time_axis = time_axis.cpu().numpy()
        waveform = waveform.cpu().numpy()

        # Создаём оси, если ax не передан
        if ax is None:
            fig, ax = plt.subplots(num_channels,
                                   1,
                                   figsize=(10, 2 * num_channels),
                                   sharex=True)
        else:
            fig = ax.get_figure()
            # Проверяем, является ли ax массивом осей или одиночным объектом
            if num_channels > 1 and not isinstance(ax, (list, np.ndarray)):
                raise TypeError("For multi-channel waveforms, `ax` must be a list or array of Axes objects")

        if num_channels == 1:
            # Если один канал, ax — это одиночный объект Axes
            ax.plot(time_axis,
                    waveform[0],
                    color=line_color,
                    linewidth=linewidth)
            if grid_flag:
                ax.grid(True)
            else:
                ax.grid(False)
            ax.set_xlim([0, time_axis[-1]])
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            # Если несколько каналов, ax — это массив осей
            for ch in range(num_channels):
                ax_ch = ax[ch] if isinstance(ax, (list, np.ndarray)) else ax
                ax_ch.plot(time_axis,
                           waveform[ch],
                           color=line_color,
                           linewidth=linewidth)
                if grid_flag:
                    ax_ch.grid(True)
                else:
                    ax_ch.grid(False)
                ax_ch.set_xlim([0, time_axis[-1]])
                ax_ch.set_title(f"{title} - Channel {ch + 1}")
                ax_ch.set_xlabel(xlabel)
                ax_ch.set_ylabel(ylabel)

        plt.tight_layout()
        title_suffix = 'Plot_Classic_WaveForm_PyTorch' if flag == "CW" else 'Plot_Reconstructed_WaveForm_PyTorch'
        cls.save_to_file(graphics_dir=graphics_dir,
                         title=title_suffix)
        plt.show()
        plt.close(fig)

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
                                 grid_flag: bool = False,
                                 normalize: bool = False,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None
    ) -> None:
        """
        Plot a spectrogram with specified colormap and interpolation.

        Parameters
        ----------
        specgram : torch.Tensor or np.ndarray
            The spectrogram to plot, expected shape [n_freq_bins, n_frames].
        title : Optional[str], optional
            The title of the plot. Defaults to None.
        xlabel : Optional[str], optional
            Label for the x-axis. Defaults to "Frame Index".
        ylabel : Optional[str], optional
            Label for the y-axis. Defaults to "freq_bin".
        colorbar_label : Optional[str], optional
            Label for the colorbar. Defaults to "Spectrogram Colorbar".
        ax : Optional[Axes], optional
            Matplotlib axes to plot on. If None, a new figure is created. Defaults to None.
        type : str, optional
            Type of spectrogram ("MFCC", "LFCC", or other). Defaults to "MFCC".
        util_type : str, optional
            Utility used to compute the spectrogram ("TorchAudio" or "Librosa"). Defaults to "TorchAudio".
        cmap : str, optional
            Colormap to use for visualization ('viridis', 'jet', etc.). Defaults to "viridis".
        interpolation : str, optional
            Interpolation method for imshow ('nearest', 'bicubic', 'spline16', etc.). Defaults to "nearest".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        normalize : bool, optional
            If True, normalize `specgram` before converting to dB. Defaults to False.
        vmin : Optional[float], optional
            Minimum value for the colormap. Defaults to None (auto-scaling).
        vmax : Optional[float], optional
            Maximum value for the colormap. Defaults to None (auto-scaling).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `specgram` is not a 2D array, or if `cmap` or `interpolation` are invalid.
        """

        if not (torch.is_tensor(specgram) or isinstance(specgram, np.ndarray)):
            raise ValueError("Specgram must be a PyTorch tensor or NumPy array")
        if len(specgram.shape) != 2:
            raise ValueError(f"Expected 2D array for specgram, got shape {specgram.shape}")
        if cmap not in VALID_CMAPS:
            raise ValueError(f"Invalid cmap {cmap}, must be one of {VALID_CMAPS}")
        if interpolation not in VALID_INTERPOLATIONS:
            raise ValueError(f"Invalid interpolation {interpolation}, must be one of {VALID_INTERPOLATIONS}")

        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
        if torch.is_tensor(specgram):
            specgram = specgram.numpy()

        # Количество частотных бинов (n_freq_bins) и фреймов (n_frames)
        n_freq_bins, n_frames = specgram.shape
        logger.info(f"Plotting spectrogram with shape {specgram.shape}, cmap={cmap}, interpolation={interpolation}")
        print_statistic_data_PyTorch(features=torch.tensor(specgram),
                                     stat_param="NumPy")

        if normalize:
            specgram = specgram / specgram.max() if specgram.max() > 0 else specgram

        specgram_db = librosa.power_to_db(specgram)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        im = ax.imshow(specgram_db,
                  origin="lower",
                  aspect="auto",
                  cmap=cmap,
                  interpolation=interpolation,
                  extent=(0, n_frames - 1, 0, n_freq_bins - 1),
                  vmin=vmin,
                  vmax=vmax)

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
            title_suffix = f'Plot_Spectrogram_MFCC_({util_type})_PyTorch'
        elif type == "LFCC":
            title_suffix = f'Plot_Spectrogram_LFCC_({util_type})_PyTorch'
        else:
            title_suffix = 'Plot_Classic_Spectrogram_PyTorch'
        cls.save_to_file(graphics_dir=graphics_dir,
                         title=title_suffix)
        plt.show()
        plt.close(fig)

    # Функция для построения Pitch
    @classmethod
    def plot_pitch_PyTorch(cls,
                           waveform: torch.Tensor,
                           sr: float,
                           pitch: torch.Tensor,
                           title: Optional[str] = "Pitch Feature",
                           language_type: str = "EN",
                           grid_flag: bool = False,
                           waveform_color: str = "gray",
                           pitch_color: str = "green",
                           waveform_linewidth: float = 1.0,
                           pitch_linewidth: float = 2.0
                           ) -> None:
        """
        Plot waveform and pitch on the same graph with dual y-axes.

        Parameters
        ----------
        waveform : torch.Tensor
            The waveform tensor, expected shape [batch, time].
        sr : float
            Sampling rate of the waveform.
        pitch : torch.Tensor
            The pitch tensor, expected shape [batch, time].
        title : Optional[str], optional
            The title of the plot. Defaults to "Pitch Feature".
        language_type : str, optional
            Language for labels ("EN" for English, "RU" for Russian). Defaults to "EN".
        grid_flag : bool, optional
            Whether to display grid lines. Defaults to False.
        waveform_color : str, optional
            Color of the waveform line. Defaults to "gray".
        pitch_color : str, optional
            Color of the pitch line. Defaults to "green".
        waveform_linewidth : float, optional
            Width of the waveform line. Defaults to 1.0.
        pitch_linewidth : float, optional
            Width of the pitch line. Defaults to 2.0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `waveform` or `pitch` are not PyTorch tensors, have incorrect shapes,
            or if `sr` is not positive or `language_type` is invalid.
        """
        # Проверка входных данных
        if not torch.is_tensor(waveform) or not torch.is_tensor(pitch):
            raise ValueError("Both 'waveform' and 'pitch' must be PyTorch tensors")
        if len(waveform.shape) != 2 or len(pitch.shape) != 2:
            raise ValueError(
                f"Expected 2D tensors for waveform and pitch, got shapes {waveform.shape} and {pitch.shape}")
        if waveform.shape[0] < 1 or pitch.shape[0] < 1:
            raise ValueError("Batch size must be at least 1 for both waveform and pitch")
        if sr <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sr}")
        if language_type not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language_type {language_type}, must be one of {VALID_LANGUAGES}")

        graphics_dir = cls.save_to_directory(graphics_dir='Graphics')
        logger.info(f"Plotting pitch with waveform shape {waveform.shape} and pitch shape {pitch.shape}")

        print("=== Waveform Statistics ===")
        print_statistic_data_PyTorch(features=waveform,
                                     stat_param="NumPy")
        print("=== Pitch Statistics ===")
        print_statistic_data_PyTorch(features=pitch,
                                     stat_param="NumPy")

        # Подготовка данных
        num_samples = waveform.shape[1]
        num_pitch_frames = pitch.shape[1]
        end_time = num_samples / sr

        # Создаём временную ось для waveform
        time_axis_waveform = torch.linspace(0,
                                            end_time,
                                            num_samples,
                                            device=waveform.device).cpu().numpy()

        # Создаём временную ось для pitch (меньшее количество точек)
        time_axis_pitch = torch.linspace(0,
                                         end_time,
                                         num_pitch_frames,
                                         device=pitch.device).cpu().numpy()

        # Интерполируем pitch, чтобы он соответствовал длине waveform
        pitch_np = pitch.cpu().numpy()
        interp_func = interp1d(time_axis_pitch,
                               pitch_np[0],
                               kind='linear',
                               fill_value="extrapolate")
        pitch_interpolated = interp_func(time_axis_waveform)  # Интерполированные значения pitch

        waveform_np = waveform.cpu().numpy()

        # Создаём график
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.set_title(title)
        if grid_flag:
            axis.grid(True)
        else:
            axis.grid(False)

        # Настройка подписей осей в зависимости от языка
        if language_type == "RU":
            axis1_plot_label = "Волновая форма"
            axis2_plot_label = "Питч"
            axis1_x_label = "Время (секунды) [s]"
            axis2_x_label = "Время (секунды) [s]"
            axis1_y_label = "Частота (Амплитуда) [Hz]"
            axis2_y_label = "Питч [Hz]"
        else:
            axis1_plot_label = "WaveForm"
            axis2_plot_label = "Pitch"
            axis1_x_label = "Time [s]"
            axis2_x_label = "Time [s]"
            axis1_y_label = "Frequency(Amplitude) [Hz]"
            axis2_y_label = "Pitch [Hz]"

        # Отрисовка waveform
        axis.plot(time_axis_waveform,
                  waveform_np[0],
                  linewidth=waveform_linewidth,
                  color=waveform_color,
                  alpha=0.3,
                  label=axis1_plot_label)
        axis.set_xlabel(axis1_x_label)
        axis.set_ylabel(axis1_y_label,
                        color=waveform_color)

        # Отрисовка pitch (интерполированного)
        axis2 = axis.twinx()
        axis2.plot(time_axis_waveform,
                   pitch_interpolated,
                   linewidth=pitch_linewidth,
                   label=axis2_plot_label,
                   color=pitch_color)
        axis2.set_xlabel(axis2_x_label)
        axis2.set_ylabel(axis2_y_label,
                         color=pitch_color)

        # Добавляем легенды
        legend = axis.legend(loc="upper left")
        legend.set_zorder(10)
        legend2 = axis2.legend(loc="upper right")
        legend2.set_zorder(10)

        plt.tight_layout()
        cls.save_to_file(graphics_dir=graphics_dir,
                         title='Plot_Pitch_PyTorch')
        plt.show()
        plt.close(fig)
