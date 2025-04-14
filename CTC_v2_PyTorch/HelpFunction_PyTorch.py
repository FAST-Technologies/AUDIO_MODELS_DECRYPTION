import os
import numpy as np
import torch
from typing import Union
from torch import Tensor
import torchaudio
import librosa
from scipy.stats import skew, kurtosis

from Constants import Constants
MY_CONSTANTS = Constants()

# Функция загрузки аудио
def load_audio_PyTorch(
    audio_path: str,
    sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
    return_format: str = "float",
    load_type: str = "mono"
) -> Tensor:
    """
    Load an audio file and preprocess it to the desired format (mono or stereo).

    Parameters
    ----------
    audio_path : str
        Path to the audio file to load.
    sample_rate : int, optional
        Desired sample rate for the audio. Defaults to MY_CONSTANTS.SAMPLE_RATE.
    return_format : str, optional
        Format of the returned tensor. Must be either "float" or "int16".
        If "float", the tensor is normalized to the range [-1.0, 1.0].
        If "int16", the tensor retains its original 16-bit integer values.
        Defaults to "float".
    load_type : str, optional
        Type of audio to load: "mono" or "stereo". Defaults to "mono".

    Returns
    -------
    torch.Tensor
        The loaded audio tensor. Shape is [channels, time], where channels is 1 for mono
        or 2 for stereo.

    Raises
    ------
    FileNotFoundError
        If `audio_path` does not exist.
    ValueError
        If `sample_rate` is not positive, `load_type` is not "mono" or "stereo",
        or the audio has an unsupported number of channels.
    torchaudio.backend.common.AudioIOException
        If the audio file cannot be loaded by torchaudio.

    Notes
    -----
    - Audio is loaded using `torchaudio.load` without normalization.
    - If the sample rate of the audio differs from the target `sample_rate`, resampling is applied.
    - For `load_type="mono"`, stereo audio is converted to mono by taking the left channel.
    - For `load_type="stereo"`, mono audio is converted to stereo by duplicating the channel.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path does not exist at path {audio_path}!")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate can't be less than zero, got {sample_rate}!")

    if load_type not in ["mono", "stereo"]:
        raise ValueError(f"Load_type must be 'mono' or 'stereo', got {load_type}")

    if return_format not in ["float", "int16"]:
        raise ValueError(f"Return_format must be either 'float' or 'int16', got {return_format}")

    # Загружаем аудио как 16-bit PCM без нормализации
    try:
        waveform, original_sample_rate = torchaudio.load(
            audio_path,
            normalize=False
        )
    except torchaudio.backend.common.AudioIOException as e:
        raise torchaudio.backend.common.AudioIOException(f"Failed to load audio file: {str(e)}")

    print(f"Original sample rate: {original_sample_rate}")
    print(f"Target sample rate: {sample_rate}")

    # Конвертируем в float32 для обработки
    waveform = waveform.float()

    if load_type == "mono":
        if waveform.dim() == 2:
            print("Аудио СТЕРЕО -> МОНО (левый канал)")
            waveform = waveform[0:1]
        elif waveform.dim() == 1:
            print("Аудио уже МОНО, продолжаем работу")
            waveform = waveform.unsqueeze(0)
        else:
            raise ValueError("Неподдерживаемое количество каналов в аудио - ошибка МОНО")
    elif load_type == "stereo":
        if waveform.dim() == 1:
            print("Аудио МОНО -> СТЕРЕО (дублирование канала)")
            waveform = torch.stack([waveform, waveform])
        elif waveform.dim() == 2 and waveform.size(0) == 1:
            print("Псевдо-МОНО -> СТЕРЕО (дублирование канала)")
            waveform = torch.cat([waveform, waveform],
                                 dim=0)
        elif waveform.dim() == 2 and waveform.size(0) == 2:
            print("Аудио уже СТЕРЕО, продолжаем работу")
        else:
            raise ValueError("Неподдерживаемое количество каналов в аудио - ошибка СТЕРЕО")

    # Ресемплируем при необходимости
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=original_sample_rate,
            new_freq=sample_rate
        )

    # Точная нормализация как в оригинале
    if return_format == "float":
        waveform = waveform / MY_CONSTANTS.FLOAT_DIVISOR
    elif return_format == "int16":
        waveform = waveform.to(torch.int16)

    return waveform

"""
Функция по выдаче статистики по каждому из каналов.
"""
def print_statistic_data_PyTorch(features: Union[torch.Tensor, np.ndarray],
                                 stat_param: str = "NumPy"
) -> None:
    """
    Print statistical data for the given features.

    Parameters
    ----------
    features : torch.Tensor or np.ndarray
        The features to analyze, expected shape [..., channels, ...] or [n_mels, time_frames].
    stat_param : str, optional
        Method to compute statistics: "Numpy" or "Torch". Defaults to "Numpy".

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `features` is not a PyTorch tensor or NumPy array, or is empty.
    """
    try:
        # Проверяем тип входных данных
        if isinstance(features, np.ndarray):
            features_np = features
            features_torch = torch.from_numpy(features_np)
        elif torch.is_tensor(features):
            features_torch = features
            features_np = features_torch.cpu().numpy()
        else:
            raise ValueError("Features must be a PyTorch tensor or NumPy array")

        # Проверка на пустоту
        if features_torch.numel() == 0:
            raise ValueError("Features must not be empty!")

        # Определяем количество каналов в зависимости от размерности
        if len(features_torch.shape) == 4:  # [batch, channels, n_mels, time]
            num_channels = features_torch.shape[1]
        elif len(features_torch.shape) == 3:  # [channels, n_mels, time]
            num_channels = features_torch.shape[0]
        elif len(features_torch.shape) == 2:  # [n_mels, time]
            num_channels = 1
        else:
            num_channels = 1

        # Базовая статистика
        if stat_param == "Torch":
            print("Shape:", features_torch.shape)
            print("Data type:", features_torch.dtype)
            print("Min value:", torch.min(features_torch).item())
            print("Max value:", torch.max(features_torch).item())
            print("Mean value:", torch.mean(features_torch).item())
            print("Std deviation:", torch.std(features_torch).item())
            print("Median value:", torch.median(features_torch).item())
            print("25th and 75th percentiles:", torch.quantile(features_torch, torch.tensor([0.25, 0.75])).tolist())
            print("Number of non-zero elements:", torch.count_nonzero(features_torch).item())
            print("Sum of all elements:", torch.sum(features_torch).item())
            print("Number of NaN values:", torch.isnan(features_torch).sum().item())
            print("Number of infinite values:", torch.isinf(features_torch).sum().item())
            print(f"Skewness: {skew(features_np.flatten())}")
            print(f"Kurtosis: {kurtosis(features_np.flatten())}")

        elif stat_param == "NumPy":
            print("Shape:", features_np.shape)  # Размерность (например, [1, 64, T] для batch=1)
            print("Data type:", features_np.dtype)  # Тип данных (np.float32)
            print("Min value:", np.min(features_np))  # Минимальное значение
            print("Max value:", np.max(features_np))  # Максимальное значение
            print("Mean value:", np.mean(features_np))  # Среднее значение
            print("Std deviation:", np.std(features_np))  # Стандартное отклонение
            print("Median value:", np.median(features_np))  # Медиана
            print("25th and 75th percentiles:", np.percentile(features_np, [25, 75]))  # Квантили
            print("Number of non-zero elements:", np.count_nonzero(features_np))  # Количество ненулевых элементов
            print("Sum of all elements:", np.sum(features_np))  # Сумма всех элементов
            print("Number of NaN values:", np.isnan(features_np).sum())  # Количество NaN
            print("Number of infinite values:", np.isinf(features_np).sum())  # Количество бесконечных значений
            print(f"Skewness: {skew(features_np.flatten())}")  # Расчёт коэффициента асимметрии
            print(f"Kurtosis: {kurtosis(features_np.flatten())}") # Расчёт коэффициента эксцесса
        else:
            raise ValueError(f"Invalid stat_param: {stat_param}. Must be 'Torch' or 'NumPy'.")

        # Статистика для левого и правого каналов (если применимо)
        if num_channels >= 2:
            if stat_param == "Torch":
                if len(features_torch.shape) == 4:  # [batch, channels, n_mels, time]
                    left_channel = features_torch[0, 0, :, :]
                    right_channel = features_torch[0, 1, :, :]
                elif len(features_torch.shape) == 3:  # [channels, n_mels, time]
                    left_channel = features_torch[0, :, :]
                    right_channel = features_torch[1, :, :]
                else:
                    left_channel = features_torch[0, :]
                    right_channel = features_torch[1, :]
                print("Shape of left channel:", left_channel.shape)
                print("Shape of right channel:", right_channel.shape)
                print("Mean value (left channel):", torch.mean(left_channel).item())
                print("Std deviation (left channel):", torch.std(left_channel).item())
                print("Mean value (right channel):", torch.mean(right_channel).item())
                print("Std deviation (right channel):", torch.std(right_channel).item())
                print(f"Max value of left channel: {torch.max(left_channel).item()}")
                print(f"Min value of left channel: {torch.min(left_channel).item()}")
                print(f"Max value of right channel: {torch.max(right_channel).item()}")
                print(f"Min value of right channel: {torch.min(right_channel).item()}")

            elif stat_param == "NumPy":
                if len(features_np.shape) == 4:  # [batch, channels, n_mels, time]
                    left_channel = features_np[0, 0, :, :]
                    right_channel = features_np[0, 1, :, :]
                elif len(features_np.shape) == 3:  # [channels, n_mels, time]
                    left_channel = features_np[0, :, :]
                    right_channel = features_np[1, :, :]
                else:
                    left_channel = features_np[0, :]
                    right_channel = features_np[1, :]
                print("Shape of left channel:", left_channel.shape)
                print("Shape of right channel:", right_channel.shape)
                print("Mean value (left channel):", np.mean(left_channel))
                print("Std deviation (left channel):", np.std(left_channel))
                print("Mean value (right channel):", np.mean(right_channel))
                print("Std deviation (right channel):", np.std(right_channel))
                print(f"Max value of left channel: {np.max(left_channel)}")
                print(f"Min value of left channel: {np.min(left_channel)}")
                print(f"Max value of right channel: {np.max(right_channel)}")
                print(f"Min value of right channel: {np.min(right_channel)}")
        else:
            print(f"Note: Only {num_channels} channel(s) found. Skipping left and right channel statistics.")

        # Статистика для каждого канала
        for i in range(num_channels):
            if stat_param == "Torch":
                if len(features_torch.shape) == 4:  # [batch, channels, n_mels, time]
                    channel = features_torch[0, i, :, :]
                elif len(features_torch.shape) == 3:  # [channels, n_mels, time]
                    channel = features_torch[i, :, :]
                elif len(features_torch.shape) == 2:  # [n_mels, time]
                    channel = features_torch
                else:
                    channel = features_torch
                print(f"Feature {i + 1}: min={torch.min(channel).item():.4f}, "
                      f"max={torch.max(channel).item():.4f}, mean={torch.mean(channel).item():.4f}")
            elif stat_param == "NumPy":
                if len(features_np.shape) == 4:  # [batch, channels, n_mels, time]
                    channel = features_np[0, i, :, :]
                elif len(features_np.shape) == 3:  # [channels, n_mels, time]
                    channel = features_np[i, :, :]
                elif len(features_np.shape) == 2:  # [n_mels, time]
                    channel = features_np
                else:
                    channel = features_np
                print(f"Feature {i + 1}: min={np.min(channel):.4f}, max={np.max(channel):.4f}, mean={np.mean(channel):.4f}")

    except Exception as e:
        print(f"Error while calculating statistics: {str(e)}")

# Функция для нахождения линейных фильтров (LFCC)
def create_linear_filters_PyTorch(n_filters: int,
                          n_fft: int,
                          fmin: float,
                          fmax: float,
                          freqs: np.ndarray,
                          normalize: bool = False
) -> np.ndarray:
    """
    Create linear filters for LFCC (Linear Frequency Cepstral Coefficients) computation.

    Parameters
    ----------
    n_filters : int
        Number of linear filters to create.
    n_fft : int
        Number of FFT points used in the STFT computation.
    fmin : float
        Minimum frequency for the filters.
    fmax : float
        Maximum frequency for the filters.
    freqs : np.ndarray
        Array of frequency points corresponding to the STFT bins, shape [n_freqs].
    normalize : bool, optional
        If True, normalize the filters so that the sum of weights for each filter equals 1.
        Defaults to False.
    Returns
    -------
    np.ndarray
        Array of linear filters, shape [n_filters, n_freqs].

    Raises
    ------
    ValueError
        If `fmax <= fmin`, `n_filters <= 0`, or `freqs` is empty.

    Notes
    -----
    - Filters are triangular, with each filter spanning between three frequency points.
    - The left slope of each filter increases linearly from 0 to 1, and the right slope
      decreases linearly from 1 to 0.
    - The implementation uses vectorized operations for efficiency.
    """
    if fmax <= fmin:
        raise ValueError(f"fmax <= fmin, but must be greater, got fmin={fmin}, fmax={fmax}")
    if n_filters <= 0:
        raise ValueError(f"n_filters <= 0, but must be a positive value, got {n_filters}")
    if len(freqs) == 0:
        raise ValueError("Freqs array must be not empty")

    filters = np.zeros((n_filters,
                        len(freqs)))
    filter_points = np.linspace(fmin,
                                fmax,
                                n_filters + 2)  # Границы фильтров

    for i in range(n_filters):
        left = filter_points[i]
        center = filter_points[i + 1]
        right = filter_points[i + 2]

        # Левый склон
        left_mask = (freqs >= left) & (freqs <= center)
        if center - left > 0:
            filters[i, left_mask] = (freqs[left_mask] - left) / (center - left)
        else:
            filters[i, left_mask] = 0.0

        # Правый склон
        right_mask = (freqs > center) & (freqs <= right)
        if right - center > 0:
            filters[i, right_mask] = (right - freqs[right_mask]) / (right - center)
        else:
            filters[i, left_mask] = 0.0

    if normalize:
        filter_sums = filters.sum(axis=1,
                                  keepdims=True)
        filters = np.divide(filters,
                            filter_sums,
                            where=filter_sums != 0)

    return filters

# Основная функция вычисления LFCC метрики для библиотеки Librosa
def compute_lfcc_PyTorch(
                 y: np.ndarray | None,
                 sr: float,
                 n_fft: int,
                 win_length: int | None,
                 hop_length: int,
                 n_filters: int,
                 n_lfcc: int,
                 fmin: float = 0.0,
                 fmax: float = None,
                 normalize: bool = False,
                 return_intermediate: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LFCC (Linear Frequency Cepstral Coefficients) features from an audio signal.

    Parameters
    ----------
    y : np.ndarray or None
        Input audio signal as a NumPy array, shape [time]. If None, the function will raise an error.
    sr : float
        Sampling rate of the audio signal.
    n_fft : int
        Number of FFT points for the STFT computation.
    win_length : int or None
        Window length for the STFT. If None, defaults to `n_fft`.
    hop_length : int
        Hop length (step size) between frames in the STFT.
    n_filters : int
        Number of linear filters to apply.
    n_lfcc : int
        Number of LFCC coefficients to compute.
    fmin : float, optional
        Minimum frequency for the filters. Defaults to 0.0.
    fmax : float, optional
        Maximum frequency for the filters. Defaults to `sr / 2` if None.
    normalize : bool, optional
        If True, normalize the LFCC features by subtracting the mean and dividing by the standard deviation.
        Defaults to False.
    return_intermediate : bool, optional
        If True, return intermediate results (power spectrogram and decibel-scaled LFCC spectrogram)
        along with the LFCC features. Defaults to False.

    Returns
    -------
    float
        The computed LFCC features as a NumPy array, shape [n_lfcc, time_frames].

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray]
        If `return_intermediate=False`, returns the LFCC features as a NumPy array, shape [n_lfcc, time_frames].
        If `return_intermediate=True`, returns a tuple containing:
        - LFCC features, shape [n_lfcc, time_frames].
        - Power spectrogram, shape [n_freqs, time_frames].
        - Decibel-scaled LFCC spectrogram, shape [n_filters, time_frames].

    Raises
    ------
    ValueError
        If `y` is None or not a 1D array, or if `sr`, `n_fft`, `hop_length`, `n_filters`, `n_lfcc`
        are not positive, or if `fmax <= fmin` (when `fmax` is not None).

    Notes
    -----
    - The function computes a power spectrogram using STFT with a Hann window.
    - Linear filters are applied to the spectrogram, followed by conversion to decibels.
    - Finally, MFCC computation (using DCT) is applied to obtain the LFCC coefficients.
    """
    if y is None:
        raise ValueError("Input audio signal 'y' must be not None")
    if y.ndim != 1:
        raise ValueError(f"Input audio signal 'y' must be 1-D, got shape: {y.shape}")
    if sr <= 0:
        raise ValueError("Sampling rate must be positive, got sr={:.2f}".format(sr))
    if n_fft <= 0:
        raise ValueError("n_fft value must be positive, got n_fft={:d}".format(n_fft))
    if hop_length <= 0:
        raise ValueError("Hop length must be positive, got hop_length={:.2f}".format(hop_length))
    if n_filters <= 0:
        raise ValueError("Number of filters must be positive, got n_filters={:.2f}".format(n_filters))
    if n_lfcc <= 0:
        raise ValueError("n_lfcc value must be positive, got n_lfcc={:d}".format(n_lfcc))
    if fmax is not None and fmin >= fmax:
        raise ValueError(f"fmax must be greater than fmin, got fmin={fmin}, fmax={fmax}")

    if win_length is None:
        win_length = n_fft

    # Вычисляем спектрограмму (мощность)
    S = librosa.stft(
        y=y,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window='hann'
    )
    power_spec = np.abs(S) ** 2  # Спектрограмма мощности

    # Частоты для STFT
    freqs = np.linspace(fmin,
                        fmax or sr / 2,
                        int(n_fft // 2 + 1))

    # Создаём линейные фильтры
    filters = create_linear_filters_PyTorch(n_filters,
                                            n_fft,
                                            fmin,
                                            fmax or sr / 2,
                                            freqs)

    # Применяем линейные фильтры к спектрограмме
    lfcc_spec = np.dot(filters, power_spec)

    # Преобразуем в децибелы
    lfcc_spec_db = librosa.power_to_db(lfcc_spec)

    # Применяем DCT для получения LFCC
    lfcc = librosa.feature.mfcc(
        S=lfcc_spec_db,
        n_mfcc=n_lfcc,
        dct_type=2,
        norm="ortho"
    )
    if normalize:
        lfcc_mean = np.mean(lfcc,
                            axis=1,
                            keepdims=True)
        lfcc_std = np.std(lfcc,
                          axis=1,
                          keepdims=True)
        lfcc = (lfcc - lfcc_mean) / (lfcc_std + 1e-8)

    if return_intermediate:
        return lfcc, power_spec, lfcc_spec_db
    return lfcc