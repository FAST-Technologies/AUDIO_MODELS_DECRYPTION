import os
from subprocess import CalledProcessError, run
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import librosa

from Constants import Constants
MY_CONSTANTS = Constants()

# Функция загрузки аудио
def load_audio_new_V2_0(
    audio_path: str,
    sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
    return_format: str = "float",
    load_type: str = "mono"
) -> np.ndarray:
    # Загружаем аудио
    try:
        original_sample_rate, waveform = wavfile.read(audio_path)
        print(f"Original shape from wavfile: {waveform.shape}, sample rate: {original_sample_rate}")
    except Exception as e:
        raise ValueError(f"Ошибка при чтении аудиофайла: {e}")

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

    # Обработка каналов
    # if load_type == "mono":
    #     if waveform.ndim == 2:
    #         print("Аудио СТЕРЕО -> МОНО (левый канал)")
    #         waveform = waveform[0][np.newaxis, :]
    #     elif waveform.ndim == 1:
    #         print("Аудио уже МОНО")
    #         waveform = waveform[np.newaxis, :]
    #     else:
    #         raise ValueError(f"Неподдерживаемое количество каналов в аудио - ошибка МОНО: {waveform.shape}")
    # elif load_type == "stereo":
    #     if waveform.ndim == 1:
    #         print("Аудио МОНО -> СТЕРЕО (дублирование канала)")
    #         waveform = np.stack([waveform, waveform], axis=0)
    #     elif waveform.ndim == 2 and waveform.shape[0] == 1:
    #         print("Псевдо-МОНО -> СТЕРЕО (дублирование канала)")
    #         waveform = np.concatenate([waveform, waveform], axis=0)
    #     elif waveform.ndim == 2 and waveform.shape[0] == 2:
    #         print("Аудио уже СТЕРЕО")
    #         waveform = waveform
    #     elif waveform.ndim == 2 and waveform.shape[0] > 2:
    #         print(f"Аудио с {waveform.shape[0]} каналами -> СТЕРЕО (берём первые 2 канала)")
    #         waveform = waveform[:2, :]
    #     else:
    #         raise ValueError(f"Неподдерживаемое количество каналов в аудио - ошибка СТЕРЕО: {waveform.shape}")
    if load_type == "mono":
        if waveform.shape[0] > 1:
            print("Аудио СТЕРЕО -> МОНО (левый канал)")
            waveform = waveform[0:1, :]
        else:
            print("Аудио уже МОНО")
    elif load_type == "stereo":
        if waveform.shape[0] == 1:
            print("Аудио МОНО -> СТЕРЕО (дублирование канала)")
            waveform = np.concatenate([waveform, waveform], axis=0)
        elif waveform.shape[0] > 2:
            print(f"Аудио с {waveform.shape[0]} каналами -> СТЕРЕО (берём первые 2 канала)")
            waveform = waveform[:2, :]
        else:
            print("Аудио уже СТЕРЕО")
    print(f"Shape after channel processing: {waveform.shape}")

    # Ресемплинг
    if original_sample_rate != sample_rate:
        old_length = waveform.shape[-1]
        new_length = int(old_length * sample_rate / original_sample_rate)
        if new_length <= 0:
            raise ValueError(f"Ошибка ресемплинга: новая длина = {new_length}")
        old_t = np.linspace(0, old_length - 1, old_length)
        new_t = np.linspace(0, old_length - 1, new_length)
        interpolator = interp1d(old_t, waveform, axis=-1, kind='linear')
        waveform = interpolator(new_t)
        print(f"Shape after resampling: {waveform.shape}")

    # Нормализация
    if return_format == "float":
        waveform = waveform / 32768.0

    return waveform

# Первоначальная функция для преобразования аудио (PyTorch вариант)
def load_audio_prev(
    audio_path: str,
    sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
    return_format: str = "float"
) -> np.ndarray:
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts"))
    ffmpeg_path = os.path.join(venv_path, "ffmpeg.exe")

    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(
            f"ffmpeg.exe not found at {ffmpeg_path}. Please ensure it is installed in .venv/Scripts.")
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
        audio_bytes = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        print(f"Ошибка ffmpeg: {exc.stderr.decode()}")
        raise RuntimeError("Failed to load audio") from exc

    # Конвертируем байты в массив NumPy
    audio = np.frombuffer(audio_bytes, dtype=np.int16)

    if return_format == "float":
        return audio.astype(np.float32) / 32768.0
    return audio

"""
Функция по выдаче статистики по каждому из каналов.
"""
def print_statistic_data_V2_0(features: np.ndarray) -> None:
    print("Shape:", features.shape)  # Размерность (например, [1, 64, T] для batch=1)
    print("Data type:", features.dtype)  # Тип данных (np.float32)
    print("Min value:", np.min(features))  # Минимальное значение
    print("Max value:", np.max(features))  # Максимальное значение
    print("Mean value:", np.mean(features))  # Среднее значение
    print("Std deviation:", np.std(features))  # Стандартное отклонение

    for i in range(features.shape[1]):  # для каждого из 64 признаков
      channel = features[0, i, :]
      print(f"Фича {i+1}: min={np.min(channel):.4f}, max={np.max(channel):.4f}, mean={np.mean(channel):.4f}")


def create_linear_filters_V2_0(n_filters: int,
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

def compute_lfcc_V2_0(y: np.ndarray | None,
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

    return lfcc