from typing import Tuple, List, Dict
import numpy as np
import onnxruntime as rt

from MetricsClass import return_metrics

VOCAB = [
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
    "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"
]

class GigaamCtcASRNumPy:
    """
    A module for extracting logarithmic mel spectrograms from raw audio signals.
    Uses the MelSpectrogram transformation (implemented manually) to extract features
    and applies logarithmic scaling.
    """
    def __init__(self,
                 model_path: str,
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
        super().__init__()
        sample_rate = 16000
        self.sample_rate = sample_rate  # Частота дискретизации входного сигнала
        self.features = 64        # Количество мел-фильтров (размерность признаков по частоте)
        self.hop_length = sample_rate // 100  # Шаг между окнами (в сэмплах), например, 160 для 16000 Гц (10 мс)
        self.n_fft = sample_rate // 40        # Длина окна для БПФ, например, 400 для 16000 Гц (25 мс)
        self.win_length = sample_rate // 40   # Длина окна Ханна, совпадает с n_fft
        self._model = rt.InferenceSession(model_path,
                                          providers=["CPUExecutionProvider"])
        try:
            self.tensor_info(flag="i",
                             tensors=self._model.get_inputs())
            self.tensor_info(flag="o",
                             tensors=self._model.get_outputs())
        except Exception as e:
            print(f"Ошибка в представлении тензорной информации модели: {e}")

        # Создание банка мел-фильтров (матрицы для преобразования спектра в мел-шкалу)
        self.mel_fb = self._create_mel_filterbank()
        self.vocab = VOCAB
        self.blank_idx = 33
        self.max_vocab_idx = len(self.vocab) - 1

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
            input_signal = input_signal[:, np.newaxis, :]

        batch_size, channels, time = input_signal.shape
        num_frames = (time - self.n_fft) // self.hop_length + 1
        print(f"batch_size: {batch_size}, channels: {channels}, time: {time}")
        print(f"n_fft: {self.n_fft}, hop_length: {self.hop_length}, num_frames: {num_frames}")

        if num_frames <= 0:
            raise ValueError(
                f"Signal length ({time}) is too short for STFT with n_fft={self.n_fft} "
                f"and hop_length={self.hop_length}. Need at least {self.n_fft} samples."
            )

        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.win_length) / (self.win_length - 1))

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
                if not frames:
                    raise ValueError("No frames extracted. Check signal length and STFT parameters.")
                spectrogram.append(np.stack(frames, axis=0))

        spectrogram = np.stack(spectrogram, axis=0).reshape(batch_size, channels, num_frames, -1)
        mel_spec = np.matmul(spectrogram, self.mel_fb.T)
        mel_spec = mel_spec.transpose(0, 1, 3, 2)
        mel_spec = np.log(np.clip(mel_spec, 1e-9, 1e9))

        if channels == 1:
            mel_spec = mel_spec.squeeze(1)

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

    def tensor_info(self,
                    flag: str,
                    tensors: List[rt.NodeArg]
                    ) -> None:
        """
        Print information about the input or output tensors of an ONNX model.

        Parameters
        ----------
        flag : str
            Either "i" for inputs or "o" for outputs.
        tensors : List[rt.NodeArg]
            List of input or output nodes from the ONNX model.

        Raises
        ------
        ValueError
            If flag is neither 'i' nor 'o'.
        """
        if flag == "i":
            print("Inputs expected by the model:")
        elif flag == "o":
            print("Outputs of the model:")
        else:
            raise ValueError(f"Invalid flag: {flag}. Must be 'i' for inputs or 'o' for outputs.")

        if not tensors:
            print("No tensors found.")
            return

        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    def decode_ctc_greedy(self,
                          log_probs: np.ndarray,
                          vocab: List[str],
                          blank_idx: int,
                          max_vocab_idx: int,
                          ground_truth: str = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Perform greedy decoding on CTC log probabilities to produce a transcription.

        Parameters
        ----------
        log_probs : np.ndarray
            Log probabilities from the model.
            Shape: [batch_size, seq_len, num_classes], where batch_size must be 1.
        vocab : List[str]
            Vocabulary list mapping token indices to characters/tokens.
        blank_idx : int
            Index of the blank token in the vocabulary.
        max_vocab_idx : int
            Maximum valid token index (len(vocab) - 1).
        ground_truth : str, optional
            Ground truth transcription for computing metrics.

        Returns
        -------
        Tuple[str, Dict[str, float]]
            - transcription : str
                Decoded transcription.
            - metrics : Dict[str, float]
                Evaluation metrics (if ground_truth is provided).

        Raises
        ------
        ValueError
            If batch_size is not 1.
        """
        if log_probs.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")
        log_prob = log_probs[0]
        token_ids = log_probs.argmax(-1).squeeze().tolist()
        print("Log probs shape:", log_probs.shape)
        print("Predicted token indices:", token_ids)
        decoded_ids: List[int] = []
        total_log_prob = 0.0
        prev_tok = None
        for t, tok in enumerate(token_ids):
            total_log_prob += float(log_prob[t, tok])
            if tok > max_vocab_idx:
                print(f"Warning: Token {tok} exceeds VOCAB size ({max_vocab_idx}), skipping")
                continue
            if (tok != prev_tok or prev_tok == blank_idx) and tok != blank_idx:
                decoded_ids.append(tok)
            prev_tok = tok
        transcription = "".join(vocab[tok] for tok in decoded_ids)
        print(f"Decoded transcription (Greedy): {transcription}")
        print(f"Log probability (Greedy): {total_log_prob:.15f}")
        metrics = {}
        if ground_truth:
            metrics = return_metrics(transcription=transcription,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=total_log_prob,
                                     flag="greedy")
        return transcription, metrics

    def decode_ctc_beam_search(
            self,
            log_probs: np.ndarray,
            vocab: List[str],
            blank_idx: int,
            max_vocab_idx: int,
            beam_width: int = 3,
            length_penalty: float = 1.0,
            ground_truth: str = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Perform beam search decoding on CTC log probabilities to produce a transcription.

        Parameters
        ----------
        log_probs : np.ndarray
            Log probabilities from the model.
            Shape: [batch_size, seq_len, num_classes], where batch_size must be 1.
        vocab : List[str]
            Vocabulary list mapping token indices to characters/tokens.
        blank_idx : int
            Index of the blank token in the vocabulary.
        max_vocab_idx : int
            Maximum valid token index (len(vocab) - 1).
        beam_width : int, optional
            Number of beams to keep at each step. Defaults to 3.
        length_penalty : float, optional
            Length penalty to apply during beam selection. Defaults to 1.0.
        ground_truth : str, optional
            Ground truth transcription for computing metrics.

        Returns
        -------
        Tuple[str, Dict[str, float]]
            - transcription : str
                Decoded transcription.
            - metrics : Dict[str, float]
                Evaluation metrics (if ground_truth is provided).

        Raises
        ------
        TypeError
            If log_probs is not a numpy.ndarray.
        ValueError
            If batch_size is not 1, or if log_probs shape is incorrect, or if max_vocab_idx/blank_idx are invalid.
        """
        if not isinstance(log_probs, np.ndarray):
            raise TypeError(f"Expected log_probs to be a numpy.ndarray, got {type(log_probs)}")

        if log_probs.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")

        if len(log_probs.shape) != 3:
            raise ValueError(f"Expected log_probs shape [batch_size, seq_len, num_classes], got {log_probs.shape}")
        log_prob = log_probs[0]
        seq_len, num_classes = log_prob.shape

        if max_vocab_idx >= num_classes:
            raise ValueError(f"max_vocab_idx ({max_vocab_idx}) must be less than num_classes ({num_classes})")
        if blank_idx >= num_classes:
            raise ValueError(f"blank_idx ({blank_idx}) must be less than num_classes ({num_classes})")

        # Инициализация лучей: (sequence, prob_blank, prob_non_blank, seq_len)
        # prob_blank — вероятность последовательности, заканчивающейся на blank
        # prob_non_blank — вероятность последовательности, заканчивающейся на non-blank
        # Shape: Dict[Tuple[int, ...], Tuple[float, float, int]]
        beams: Dict[Tuple[int, ...], Tuple[float, float, int]] = {tuple(): (0.0, float('-inf'), 0)}
        for t in range(seq_len):
            new_beams: Dict[Tuple[int, ...], Tuple[float, float, int]] = {}
            for seq, (prob_blank, prob_non_blank, seq_len_so_far) in beams.items():
                prob_total = np.logaddexp(prob_blank, prob_non_blank)
                for token in range(num_classes):
                    if token > max_vocab_idx and token != blank_idx:
                        continue

                    token_log_prob = float(log_prob[t, token])
                    if token == blank_idx:
                        new_prob_blank = prob_total + token_log_prob
                        current = new_beams.get(tuple(seq), (float('-inf'), float('-inf'), seq_len_so_far))
                        new_beams[tuple(seq)] = (
                            np.logaddexp(current[0], new_prob_blank),
                            current[1],
                            seq_len_so_far
                        )
                    else:
                        new_seq = list(seq)
                        if new_seq and new_seq[-1] == token:
                            new_prob_non_blank = prob_non_blank + token_log_prob
                            current = new_beams.get(tuple(seq), (float('-inf'), float('-inf'), seq_len_so_far))
                            new_beams[tuple(seq)] = (
                                current[0],
                                np.logaddexp(current[1], new_prob_non_blank),
                                seq_len_so_far
                            )
                        else:
                            new_seq.append(token)
                            new_seq_len = seq_len_so_far + 1
                            new_prob_non_blank = prob_total + token_log_prob
                            current = new_beams.get(tuple(new_seq), (float('-inf'), float('-inf'), new_seq_len))
                            new_beams[tuple(new_seq)] = (
                                current[0],
                                np.logaddexp(current[1], new_prob_non_blank),
                                new_seq_len
                            )
            beams = {}
            for seq, (prob_blank, prob_non_blank, seq_len_so_far) in sorted(
                    new_beams.items(),
                    key=lambda x: np.logaddexp(x[1][0], x[1][1]) / (x[1][2] ** length_penalty if x[1][2] > 0 else 1.0),
                    reverse=True
            )[:beam_width]:
                beams[tuple(seq)] = (prob_blank, prob_non_blank, seq_len_so_far)

        best_seq, (best_prob_blank, best_prob_non_blank, _) = max(
            beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1])
        )
        best_log_prob = np.logaddexp(best_prob_blank, best_prob_non_blank)
        transcription = "".join(vocab[tok] for tok in best_seq)
        print(f"Transcription (Beam Search, beam_width={beam_width}, length_penalty={length_penalty}): {transcription}")
        print(f"Log probability (Beam Search): {best_log_prob + 1:.14f}")
        metrics = {}
        if ground_truth:
            metrics = return_metrics(transcription=transcription,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=best_log_prob + 1,
                                     beam_width=beam_width,
                                     length_penalty=length_penalty,
                                     flag="beam")

        return transcription, metrics

    def recognize(self,
                  waveforms: np.ndarray,
                  decode_flag: str = "greedy",
                  beam_width: int = 10,
                  length_penalty: float = 0.7,
                  ground_truth: str = None
                  ) -> Tuple[str, Dict[str, float]]:
        """
        Recognize speech from the input waveform and return the transcription.

        Parameters
        ----------
        waveforms : np.ndarray
            Input waveform, shape [channels, samples] or [samples]. Expected to be in float32 format.
        decode_flag : str, optional
            Decoding method: "greedy" for greedy decoding, "beam" for beam search. Defaults to "greedy".
        beam_width : int, optional
            Number of beams for beam search decoding. Defaults to 10.
        length_penalty : float, optional
            Length penalty for beam search decoding. Defaults to 0.7.
        ground_truth : str, optional
            Ground truth transcription for computing metrics. Defaults to None.

        Returns
        -------
        Tuple[str, Dict[str, float]]
            - Transcribed text.
            - Metrics dictionary (if ground_truth is provided).
        """
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")

        if waveforms.dtype != np.float32:
            print(
                f"Предупреждение: waveforms имеет тип {waveforms.dtype}, ожидается np.float32. Выполняется приведение.")
            waveforms = waveforms.astype(np.float32)

        # Обрабатываем входной массив
        if waveforms.ndim == 1:
            # [samples] -> [1, samples]
            waveforms = waveforms[np.newaxis, :]
        elif waveforms.ndim == 2:
            # Уже в формате [channels, samples]
            pass
        else:
            raise ValueError(f"Expected waveforms to have 1 or 2 dimensions, got shape {waveforms.shape}")

        # Если больше одного канала, усредняем
        if waveforms.shape[0] > 1:
            # waveforms = np.mean(waveforms, axis=0, keepdims=True)  # [channels, samples] -> [1, samples]
            waveforms = waveforms[0][np.newaxis, :]  # [2, time] -> [1, time]

        # Добавляем размерность батча: [channels, samples] -> [batch_size, channels, samples]
        audio_tensor = waveforms[np.newaxis, :]  # [1, channels, samples]
        audio_length = np.array([audio_tensor.shape[-1]], dtype=np.int64)

        features, lengths = self.forward(audio_tensor, audio_length)
        features = features.astype(np.float32)
        lengths = lengths.astype(np.int64)
        inputs = {
            "features": features,
            "feature_lengths": lengths
        }

        log_probs = self._model.run(["log_probs"], inputs)[0]
        print("Тип значений log_probs:", type(log_probs))
        print("Размерность значений log_probs:", log_probs.shape)

        if np.any(np.isnan(log_probs)):
            raise ValueError("log_probs contains NaN values")
        if np.any(np.isinf(log_probs)):
            raise ValueError("log_probs contains Inf values")

        if len(log_probs.shape) == 2:
            log_probs = log_probs[np.newaxis, :]  # [1, seq_len, num_classes]
        elif len(log_probs.shape) != 3:
            raise ValueError(f"Unexpected shape for log_probs: {log_probs.shape}")

        if decode_flag == "greedy":
            transcription, metrics = self.decode_ctc_greedy(
                log_probs=log_probs,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                ground_truth=ground_truth
            )
        elif decode_flag == "beam":
            transcription, metrics = self.decode_ctc_beam_search(
                log_probs=log_probs,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                beam_width=beam_width,
                length_penalty=length_penalty,
                ground_truth=ground_truth
            )
        else:
            raise ValueError(f"Invalid decode_flag: {decode_flag}. Must be 'greedy' or 'beam'.")

        return transcription, metrics
