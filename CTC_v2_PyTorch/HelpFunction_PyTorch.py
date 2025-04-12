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
            waveform = waveform[0:1]  # Берём левый канал
        elif waveform.dim() == 1:
            print("Аудио уже МОНО")
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
            print("Аудио уже СТЕРЕО")
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
        print("Ia yest float")

    return waveform

"""
Функция по выдаче статистики по каждому из каналов.
"""
def print_statistic_data_PyTorch(features: Tensor) -> None:
    print("Shape:", features.shape)  # Размерность (например, [1, 64, T] для batch=1)
    print("Data type:", features.dtype)  # Тип данных (np.float32)
    print("Min value:", np.min(features))  # Минимальное значение
    print("Max value:", np.max(features))  # Максимальное значение
    print("Mean value:", np.mean(features))  # Среднее значение
    print("Std deviation:", np.std(features))  # Стандартное отклонение

    for i in range(features.shape[1]):  # для каждого из 64 признаков
      channel = features[0, i, :]
      print(f"Фича {i+1}: min={np.min(channel):.4f}, max={np.max(channel):.4f}, mean={np.mean(channel):.4f}")

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
        filters[i, left_mask] = (freqs[left_mask] - left) / (center - left)

        # Правый склон
        right_mask = (freqs > center) & (freqs <= right)
        filters[i, right_mask] = (right - freqs[right_mask]) / (right - center)

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