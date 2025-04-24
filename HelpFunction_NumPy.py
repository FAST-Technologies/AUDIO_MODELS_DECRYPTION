import os
from subprocess import CalledProcessError, run
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import librosa
import shutil
from scipy.stats import skew, kurtosis

from Constants import Constants
MY_CONSTANTS = Constants()

# Функция загрузки аудио
def load_audio_new_V2_0(
    audio_path: str,
    sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
    return_format: str = "float",
    load_type: str = "mono"
) -> np.ndarray:
    """
    Load an audio file and preprocess it to the desired format (mono or stereo).

    Parameters
    ----------
    audio_path : str
        Path to the audio file to load.
    sample_rate : int, optional
        Desired sample rate for the audio. Defaults to MY_CONSTANTS.SAMPLE_RATE.
    return_format : str, optional
        Format of the returned ndarray. Must be either "float" or "int16".
        If "float", the ndarray is normalized to the range [-1.0, 1.0].
        If "int16", the ndarray retains its original 16-bit integer values.
        Defaults to "float".
    load_type : str, optional
        Type of audio to load: "mono" or "stereo". Defaults to "mono".

    Returns
    -------
    np.ndarray
        The loaded audio ndarray. Shape is [channels, time], where channels is 1 for mono
        or 2 for stereo.

    Raises
    ------
    FileNotFoundError
        If `audio_path` does not exist.
    ValueError
        If `sample_rate` is not positive, `load_type` is not "mono" or "stereo",
        or the audio has an unsupported number of channels.
        If the audio file cannot be loaded by wavfile.

    Notes
    -----
    - Audio is loaded using `wavfile.read` without normalization.
    - If the sample rate of the audio differs from the target `sample_rate`, resampling is applied.
    - For `load_type="mono"`, stereo audio is converted to mono by taking the left channel.
    - For `load_type="stereo"`, mono audio is converted to stereo by duplicating the channel.
    """
    # Загружаем аудио
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path does not exist at path {audio_path}!")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate can't be less than zero, got {sample_rate}!")

    if load_type not in ["mono", "stereo"]:
        raise ValueError(f"Load_type must be 'mono' or 'stereo', got {load_type}")

    if return_format not in ["float", "int16"]:
        raise ValueError(f"Return_format must be either 'float' or 'int16', got {return_format}")

    try:
        original_sample_rate, waveform = wavfile.read(audio_path)
        print(f"Original shape from wavfile: {waveform.shape}, sample rate: {original_sample_rate}")
    except Exception as e:
        raise ValueError(f"Error trying to read audiofile: {str(e)}")

    print(f"Original sample rate: {original_sample_rate}")
    print(f"Target sample rate: {sample_rate}")

    # Проверяем, что данные не пустые
    if waveform.size == 0:
        raise ValueError("Аудиофайл пустой или некорректный")

    # Конвертируем в float32
    waveform = waveform.astype(np.float32)
    print(f"Shape after conversion to float32: {waveform.shape}")

    # Исправляем ориентацию: wavfile.read возвращает (samples, channels), а нам нужно (channels, samples)
    if waveform.ndim == 2:
        waveform = waveform.T  # Транспонируем: (samples, channels) -> (channels, samples)
        print(f"Shape after transpose: {waveform.shape}")
    elif waveform.ndim == 1:
        print("Аудио моно, оставляем как есть")

    if load_type == "mono":
        if waveform.ndim == 2:
            if waveform.shape[0] > 1:
                print("Аудио СТЕРЕО -> МОНО (левый канал)")
                waveform = waveform[0][np.newaxis, :]  # [2, time] -> [1, time]
            else:
                print("Аудио уже МОНО (2D)")
        elif waveform.ndim == 1:
            print("Аудио уже МОНО (1D)")
            waveform = waveform[np.newaxis, :]  # [time] -> [1, time]
        else:
            raise ValueError(f"Неподдерживаемое количество каналов в аудио - ошибка МОНО: {waveform.ndim}")
    elif load_type == "stereo":
        if waveform.ndim == 1:
            print("Аудио МОНО -> СТЕРЕО (дублирование канала)")
            waveform = np.stack([waveform, waveform],
                                axis=0)
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            print("Псевдо-МОНО -> СТЕРЕО (дублирование канала)")
            waveform = np.concatenate([waveform, waveform],
                                      axis=0)
        elif waveform.ndim == 2 and waveform.shape[0] == 2:
            print("Аудио уже СТЕРЕО")
            waveform = waveform
        elif waveform.ndim == 2 and waveform.shape[0] > 2:
            print(f"Аудио с {waveform.shape[0]} каналами -> СТЕРЕО (берём первые 2 канала)")
            waveform = waveform[:2, :]
        else:
            raise ValueError(f"Неподдерживаемое количество каналов в аудио - ошибка СТЕРЕО: {waveform.shape}")
    print(f"Shape after channel processing: {waveform.shape}")

    # Ресемплинг
    if original_sample_rate != sample_rate:
        old_length = waveform.shape[-1]
        if original_sample_rate <= 1e-18:
            raise ValueError("Новая длина не может быть посчитана")
        new_length = int(old_length * sample_rate / original_sample_rate)
        if new_length <= 0:
            raise ValueError(f"Ошибка ресемплинга: новая длина = {new_length}")
        old_t = np.linspace(0,
                            old_length - 1,
                            old_length)
        new_t = np.linspace(0,
                            old_length - 1,
                            new_length)
        interpolator = interp1d(old_t,
                                waveform,
                                axis=-1,
                                kind='linear')
        waveform = interpolator(new_t)
        print(f"Shape after resampling: {waveform.shape}")

    # Нормализация
    if return_format == "float":
        waveform = waveform / MY_CONSTANTS.FLOAT_DIVISOR
    elif return_format == "int16":
        waveform = waveform.astype(np.int16)

        # Явное приведение к np.float32 для return_format="float"
    if return_format == "float":
        waveform = waveform.astype(np.float32)

    return waveform

# Первоначальная функция для преобразования аудио (PyTorch вариант)
def load_audio_prev(
    audio_path: str,
    sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
    return_format: str = "float"
) -> np.ndarray:
    """
    Load an audio file using ffmpeg and convert it to a NumPy array.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    sample_rate : int, optional
        Target sample rate for the audio. Defaults to MY_CONSTANTS.SAMPLE_RATE.
    return_format : str, optional
        Format of the output array, either 'float' (float32) or 'int16'. Defaults to 'float'.

    Returns
    -------
    np.ndarray
        Audio waveform as a NumPy array.
        Shape: [T], where T is the number of audio samples.
        Dtype: np.float32 if return_format='float', np.int16 if return_format='int16'.

    Raises
    ------
    FileNotFoundError
        If the audio file is not found or ffmpeg is not installed.
    ValueError
        If sample_rate is less than or equal to zero, or if return_format is invalid.
    RuntimeError
        If ffmpeg fails to load the audio.

    Notes
    -----
    - Audio is loaded in 16-bit PCM format (s16le) using ffmpeg.
    - If return_format='float', the audio is normalized to the range [-1.0, 1.0] by dividing by 32768.0.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path does not exist at path {audio_path}!")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate can't be less than zero, got {sample_rate}!")

    if return_format not in ["float", "int16"]:
        raise ValueError(f"Return_format must be either 'float' or 'int16', got {return_format}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts"))
        ffmpeg_path = os.path.join(venv_path, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(
                f"ffmpeg not found in PATH or at {ffmpeg_path}. "
                f"Please ensure ffmpeg is installed and available in PATH or in .venv/Scripts."
            )

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-threads", "0",
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-"
    ]
    try:
        audio_bytes = run(cmd,
                          capture_output=True,
                          check=True).stdout
    except CalledProcessError as exc:
        print(f"Ошибка ffmpeg: {exc.stderr.decode()}")
        raise RuntimeError("Failed to load audio") from exc

    audio = np.frombuffer(audio_bytes,
                          dtype=np.int16)

    if return_format == "float":
        return audio.astype(np.float32) / MY_CONSTANTS.FLOAT_DIVISOR
    elif return_format == "int16":
        audio = audio.astype(np.int16)
    return audio
"""
Функция по выдаче статистики по каждому из каналов.
"""
def print_statistic_data_V2_0(features: np.ndarray) -> None:
    """
    Print statistical data for the given features.

    Parameters
    ----------
    features : torch.Tensor or np.ndarray
        The features to analyze, expected shape [..., channels, ...] or [n_mels, time_frames].

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `features` is not a NumPy array, or is empty.
    """
    try:
        if len(features.shape) == 4:  # [batch, channels, n_mels, time]
            num_channels = features.shape[1]
        elif len(features.shape) == 3:  # [channels, n_mels, time]
            num_channels = features.shape[0]
        elif len(features.shape) == 2:  # [n_mels, time]
            num_channels = 1
        else:
            num_channels = 1

        print("Shape:", features.shape)
        print("Data type:", features.dtype)
        print("Min value:", np.min(features))
        print("Max value:", np.max(features))
        print("Mean value:", np.mean(features))
        print("Std deviation:", np.std(features))
        print("Median value:", np.median(features))
        print("25th and 75th percentiles:", np.percentile(features, [25, 75]))
        print("Number of non-zero elements:", np.count_nonzero(features))
        print("Sum of all elements:", np.sum(features))
        print("Number of NaN values:", np.isnan(features).sum())
        print("Number of infinite values:", np.isinf(features).sum())
        print(f"Skewness: {skew(features.flatten())}")
        print(f"Kurtosis: {kurtosis(features.flatten())}")

        if num_channels >= 2:
            if len(features.shape) == 4:  # [batch, channels, n_mels, time]
                left_channel = features[0, 0, :, :]
                right_channel = features[0, 1, :, :]
            elif len(features.shape) == 3:  # [channels, n_mels, time]
                left_channel = features[0, :, :]
                right_channel = features[1, :, :]
            else:
                left_channel = features[0, :]
                right_channel = features[1, :]
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
            if len(features.shape) == 4:  # [batch, channels, n_mels, time]
                channel = features[0, i, :, :]
            elif len(features.shape) == 3:  # [channels, n_mels, time]
                channel = features[i, :, :]
            elif len(features.shape) == 2:  # [n_mels, time]
                channel = features
            else:
                channel = features
            print(
                f"Feature {i + 1}: min={np.min(channel):.15f}, max={np.max(channel):.15f}, mean={np.mean(channel):.15f}")
    except Exception as e:
        print(f"Error while calculating statistics: {str(e)}")


def create_linear_filters_V2_0(n_filters: int,
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
        raise ValueError(f"Fmax <= fmin, but must be greater, got fmin={fmin}, fmax={fmax}")
    if n_filters <= 0:
        raise ValueError(f"N_filters <= 0, but must be a positive value, got {n_filters}")
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

def compute_lfcc_V2_0(y: np.ndarray | None,
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
        raise ValueError("N_lfcc value must be positive, got n_lfcc={:d}".format(n_lfcc))
    if fmax is not None and fmin >= fmax:
        raise ValueError(f"Fmax must be greater than fmin, got fmin={fmin}, fmax={fmax}")

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
    freqs = np.linspace(fmin, fmax or sr / 2, n_fft // 2 + 1)

    # Создаём линейные фильтры
    filters = create_linear_filters_V2_0(n_filters, n_fft, fmin, fmax or sr / 2, freqs)

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