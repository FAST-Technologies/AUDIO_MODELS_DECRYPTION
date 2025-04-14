from typing import Tuple
import numpy as np

from Constants import Constants
MY_CONSTANTS = Constants()  # Предполагается, что это объект с константами, но он не используется в коде напрямую

class FeatureExtractor_V2_0:
    """
    A module for extracting logarithmic mel spectrograms from raw audio signals.
    Uses the MelSpectrogram transformation (implemented manually) to extract features
    and applies logarithmic scaling.
    """
    def __init__(self,
                 sample_rate: int,  # Частота дискретизации аудио (например, 16000 Гц)
                 features: int      # Количество мел-фильтров (например, 40 или 80)
    ) -> None:
        """
        Initialization of the module for extracting mel spectrograms.

         Parameters
         ----------
         sample_rate : int
            Sampling rate of the input audio signal (for example, 16000 Hz).
         features : int
            Number of chalk filters (frequency dimension of features, for example, 40 or 80).

         Notes
         -----
         Calculates the parameters for STFT (n_fft, hop_length, win_length) and creates a bank of chalk filters.
         """
        # Инициализация параметров
        self.sample_rate = sample_rate  # Частота дискретизации входного сигнала
        self.features = features        # Количество мел-фильтров (размерность признаков по частоте)
        self.hop_length = sample_rate // 100  # Шаг между окнами (в сэмплах), например, 160 для 16000 Гц (10 мс)
        self.n_fft = sample_rate // 40        # Длина окна для БПФ, например, 400 для 16000 Гц (25 мс)
        self.win_length = sample_rate // 40   # Длина окна Ханна, совпадает с n_fft

        # Создание банка мел-фильтров (матрицы для преобразования спектра в мел-шкалу)
        self.mel_fb = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        """
         Creates a bank of chalk filters for converting the power spectrum into a chalk spectrogram.
         A linear approximation of triangular filters in a chalk scale is used.

         Returns
         -------
         np.ndarray
            The matrix of chalk filters.
            Shape: [n_mels, n_freqs], where n_mels = self.features, n_freqs = self.n_fft // 2 + 1

         Notes
        -----
        - Converts frequencies from Hertz to the chalk scale: mel = 1125 * ln(1 + f/700).
        - Creates triangular filters evenly distributed in the chalk scale.
        """
        # Количество частотных бинов в спектре (половина длины БПФ + 1 из-за симметрии)
        n_freqs = int(self.n_fft // 2 + 1)  # Например, 201 для n_fft=400
        f_min, f_max = 0.0, self.sample_rate / 2.0  # Диапазон частот: от 0 до половины частоты дискретизации (Nyquist)

        # Перевод частот в мел-шкалу по формуле: mel = 1125 * ln(1 + f/700)
        mel_min = 1125.0 * np.log1p(f_min / 700.0)  # Минимальная мел-частота (0 Гц -> 0 мел)
        mel_max = 1125.0 * np.log1p(f_max / 700.0)  # Максимальная мел-частота (например, 8000 Гц -> ~2835 мел)

        # Равномерное распределение точек в мел-шкале (features + 2 для границ фильтров)
        mel_points = np.linspace(mel_min, mel_max, self.features + 2)  # Например, 42 точки для 40 фильтров
        # Обратное преобразование из мел-шкалы в герцы: f = 700 * (exp(mel/1125) - 1)
        freq_points = 700.0 * (np.expm1(mel_points / 1125.0))  # Частоты границ фильтров в Гц

        # Инициализация матрицы фильтров: [кол-во фильтров, кол-во частотных бинов]
        fb = np.zeros((self.features, n_freqs))
        freqs = np.linspace(0, f_max, n_freqs)  # Линейная шкала частот для спектра (0 до f_max)

        # Построение треугольных фильтров
        for m in range(self.features):  # Для каждого фильтра
            f_left = freq_points[m]     # Левая граница треугольника
            f_center = freq_points[m + 1]  # Вершина треугольника
            f_right = freq_points[m + 2]   # Правая граница треугольника
            for f in range(n_freqs):  # Для каждой частоты в спектре
                if f_left <= freqs[f] <= f_center:  # Левая часть треугольника (рост)
                    fb[m, f] = (freqs[f] - f_left) / (f_center - f_left)  # Линейная интерполяция
                elif f_center < freqs[f] <= f_right:  # Правая часть треугольника (спад)
                    fb[m, f] = (f_right - freqs[f]) / (f_right - f_center)  # Линейная интерполяция
        return fb  # Матрица фильтров [n_mels, n_freqs]

    def forward(self,
                input_signal: np.ndarray,  # Входной сигнал: [B, channels, T] или [B, T]
                length: np.ndarray         # Длина сигнала для каждого батча: [B]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The main method for extracting features (logarithmic mel spectrograms).

         Parameters
         ----------
         input_signal : np.ndarray
            Audio input signal.
            Shape: [B, channels, T] or [B, T], where B is the batch size, channels is the number of channels, and T is the signal length.
         length : np.ndarray
            The length of the signal for each batch.
            Shape: [B], dtype: int

         Returns
         -------
         Tuple[np.ndarray, np.ndarray]
         - mel_spec : np.ndarray
                Logarithmic mel spectrogram.
                Shape: [B, channels, n_mels, T] or [B, n_mels, T] (if channels=1), where T is the number of frames.
         - out_lengths : np.ndarray
                The length of the output frames for each batch.
                Shape: [B], dtype: int64

         Notes
         -----
         - Performs STFT (Short-time Fourier Transform) with a Hanna window.
         - Converts the power spectrum into a mel spectrogram with using a bank of chalk filters.
         - Applies logarithmic scaling to compress the dynamic range.
         """
        # Если сигнал двумерный [B, T], добавляем размерность канала
        if input_signal.ndim == 2:
            input_signal = input_signal[:, np.newaxis, :]  # Преобразуем в [B, 1, T]

        # Извлечение размерностей входного сигнала
        batch_size, channels, time = input_signal.shape
        # Количество кадров после разбиения сигнала на окна
        num_frames = (time - self.n_fft) // self.hop_length + 1

        # Создание окна Ханна (Hanning window) для сглаживания краев кадров
        # Формула: w(n) = 0.5 * (1 - cos(2πn / (N-1))), где N - длина окна
        # Уменьшает утечку спектра, боковые лепестки на уровне -31.5 дБ
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.win_length) / (self.win_length - 1))

        # Вычисление коротковременного преобразования Фурье (STFT) вручную
        spectrogram = []
        for b in range(batch_size):  # Для каждого элемента батча
            for c in range(channels):  # Для каждого канала
                signal = input_signal[b, c]  # Извлекаем сигнал [T]
                frames = []
                for i in range(num_frames):  # Разбиваем сигнал на кадры
                    start = i * self.hop_length  # Начало текущего кадра
                    frame = signal[start:start + self.n_fft]  # Извлекаем кадр длиной n_fft
                    if len(frame) < self.n_fft:  # Если кадр короче (в конце сигнала)
                        frame = np.pad(frame, (0, self.n_fft - len(frame)))  # Дополняем нулями
                    frame = frame * window[:len(frame)]  # Применяем окно Ханна
                    fft = np.fft.rfft(frame, n=self.n_fft)  # Реальное БПФ (rfft для экономии)
                    power = np.abs(fft) ** 2  # Вычисляем спектр мощности (|FFT|^2)
                    frames.append(power)
                spectrogram.append(np.stack(frames, axis=0))  # Собираем кадры в [T, n_freqs]
        # Преобразуем список в массив [B * channels, T, n_freqs], затем в [B, channels, T, n_freqs]
        spectrogram = np.stack(spectrogram, axis=0).reshape(batch_size, channels, num_frames, -1)

        # Применение мел-фильтров: умножение спектра мощности на матрицу фильтров
        mel_spec = np.matmul(spectrogram, self.mel_fb.T)  # [B, channels, T, n_mels]
        mel_spec = mel_spec.transpose(0, 1, 3, 2)  # Переставляем оси: [B, channels, n_mels, T]

        # Логарифмическое масштабирование для сжатия динамического диапазона
        # Обрезаем значения снизу (1e-9) и сверху (1e9) для численной стабильности
        mel_spec = np.log(np.clip(mel_spec, 1e-9, 1e9))

        # Если сигнал моно (1 канал), убираем размерность канала
        if channels == 1:
            mel_spec = mel_spec.squeeze(1)  # [B, n_mels, T]

        # Возвращаем мел-спектрограмму и длину выходных кадров
        return mel_spec, self.out_len(length)

    def __call__(self,
                 input_signal: np.ndarray,  # Входной сигнал: [B, channels, T] или [B, T]
                 length: np.ndarray         # Длина сигнала для каждого батча: [B]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes the class callable by redirecting the call to the forward method.

         Parameters
         ----------
         input_signal : np.ndarray
            Audio input signal.
            Shape: [B, channels, T] or [B, T], where B is the batch size, channels is the number of channels, and T is the signal length.
         length : np.ndarray
            The length of the signal for each batch.
            Shape: [B], dtype: int

         Returns
         -------
         Tuple[np.ndarray, np.ndarray]
         - mel_spec : np.ndarray
                Logarithmic mel spectrogram.
                Shape: [B, channels, n_mels, T] or [B, n_mels, T] (if channels=1), where T is the number of frames.
         - out_lengths : np.ndarray
                The length of the output frames for each batch.
                Shape: [B], dtype: int64

         Notes
         -----
         Redirects the call to the forward method for ease of use.
         """
        return self.forward(input_signal, length)

    def out_len(self,
                input_lengths: np.ndarray  # Входные длины сигналов: [B]
    ) -> np.ndarray:
        """
        Calculates the length of the output data (number of frames) after feature extraction.

         Parameters
         ----------
         input_lengths : np.ndarray
            The length of the input signals for each batch.
            Shape: [B], dtype: int

         Returns
         -------
         np.ndarray
            The length of the output frames for each batch.
            Shape: [B], dtype: int64

         Notes
         -----
         Takes into account the hop_length step and the n_fft window length.
         Formula: (input_length - n_fft) // hop_length + 1
        """
        # Формула: (длина сигнала - длина окна) // шаг + 1
        # Input shape: [B], dtype: int
        # Output shape: [B], dtype: int64
        return (input_lengths // self.hop_length + 1).astype(np.int64)