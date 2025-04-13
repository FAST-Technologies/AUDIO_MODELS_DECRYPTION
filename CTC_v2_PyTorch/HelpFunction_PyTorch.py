import numpy as np
import torch
from torch import Tensor
import torchaudio
import librosa

from Constants import Constants
MY_CONSTANTS = Constants()

# Функция загрузки аудио
def load_audio_PyTorch(
    audio_path: str,
    sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
    return_format: str = "float",
    load_type: str = "mono"
) -> torch.Tensor:

    # Загружаем аудио как 16-bit PCM без нормализации
    waveform, original_sample_rate = torchaudio.load(
        audio_path,
        normalize=False
    )

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
            waveform = torch.cat([waveform, waveform], dim=0)
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
        waveform = waveform / 32768.0

    return waveform

"""
Функция по выдаче статистики по каждому из каналов.
"""
def print_statistic_data_PyTorch(features: Tensor) -> None:
    """Print statistical data for the given features.

        Args:
            features (Tensor): The features to analyze, expected shape [batch, channels, time].
        """
    try:
        # Преобразуем features в NumPy, если это тензор
        if torch.is_tensor(features):
            features = features.numpy()
        elif not isinstance(features, np.ndarray):
            raise ValueError("Input 'features' must be a PyTorch tensor or NumPy array")

        # Проверка на пустоту
        if features.size == 0:
            raise ValueError("Input 'features' is empty")

        # Проверка размерности
        # if len(features.shape) != 3:
        #     raise ValueError(f"Expected 3D tensor with shape [batch, channels, time], got shape {features.shape}")

        # Базовая статистика
        print("Shape:", features.shape)  # Размерность (например, [1, 64, T] для batch=1)
        print("Data type:", features.dtype)  # Тип данных (np.float32)
        print("Min value:", np.min(features))  # Минимальное значение
        print("Max value:", np.max(features))  # Максимальное значение
        print("Mean value:", np.mean(features))  # Среднее значение
        print("Std deviation:", np.std(features))  # Стандартное отклонение
        print("Median value:", np.median(features))  # Медиана
        print("25th and 75th percentiles:", np.percentile(features, [25, 75]))  # Квантили
        print("Number of non-zero elements:", np.count_nonzero(features))  # Количество ненулевых элементов
        print("Sum of all elements:", np.sum(features))  # Сумма всех элементов
        print("Number of NaN values:", np.isnan(features).sum())  # Количество NaN
        print("Number of infinite values:", np.isinf(features).sum())  # Количество бесконечных значений

        # Проверка количества каналов
        num_channels = features.shape[1]
        if num_channels < 1:
            raise ValueError("No channels found in features (features.shape[1] < 1)")

        # Статистика для левого и правого каналов (если они есть)
        if num_channels >= 2:
            print("Shape of left channel:", features[0][0].shape)  # Размерность левого канала
            print("Shape of right channel:", features[0][1].shape)  # Размерность правого канала
            print("Mean value (left channel):", np.mean(features[0][0]))  # Среднее значение левого канала
            print("Std deviation (left channel):", np.std(features[0][0]))  # Стандартное отклонение левого канала
            print("Mean value (right channel):", np.mean(features[0][1]))  # Среднее значение правого канала
            print("Std deviation (right channel):", np.std(features[0][1]))  # Стандартное отклонение правого канала
        else:
            print(
                f"Warning: Only {num_channels} channel(s) found. Expected at least 2 channels for left and right channel statistics.")

        # Статистика для каждого канала
        for i in range(num_channels):
            channel = features[0, i, :]
            print(f"Фича {i + 1}: min={np.min(channel):.4f}, max={np.max(channel):.4f}, mean={np.mean(channel):.4f}")

    except Exception as e:
        print(f"Error while calculating statistics: {str(e)}")

# Функция для нахождения линейных фильтров (LFCC)
def create_linear_filters_PyTorch(n_filters: int,
                          n_fft: int,
                          fmin: float,
                          fmax: float,
                          freqs: np.ndarray
) -> np.ndarray:
    filters = np.zeros((n_filters, len(freqs)))
    filter_points = np.linspace(fmin, fmax, n_filters + 2)  # Границы фильтров

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

    return filters

# Основная функция вычисления LFCC метрики для библиотеки Librosa
def compute_lfcc_PyTorch(y: np.ndarray | None,
                 sr: float,
                 n_fft: int,
                 win_length: int | None,
                 hop_length: int,
                 n_filters: int,
                 n_lfcc: int,
                 fmin: float = 0.0,
                 fmax: float = None
) -> float:
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
    freqs = np.linspace(fmin, fmax or sr / 2, int(n_fft // 2 + 1))

    # Создаём линейные фильтры
    filters = create_linear_filters_PyTorch(n_filters, n_fft, fmin, fmax or sr / 2, freqs)

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
    return lfcc