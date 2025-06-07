from typing import List, Tuple, Dict
import re
import os
from datetime import datetime
import numpy as np
import onnxruntime as rt
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass

from MetricsClass import return_metrics

@dataclass(frozen=True)
class Constants:
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    features: int = 80
    preemph: float = 0.97
    hidden_size: int = 640
    log_zero_guard_value: float = 2 ** -24
    ru_vocab_length: int = 1025

graphics_dir = "Graphics"

class NumPyPreprocessor:
    def __init__(self,
                 graphics_create: bool = False,
                 window: str = "hann",
                 center: bool = True,
                 mel_filterbank_type: str = "complex"):
        """Initialize the NumPy-based audio preprocessor."""
        self.sample_rate = Constants.sample_rate
        self.n_fft = Constants.n_fft
        self.win_length = Constants.win_length
        self.hop_length = Constants.hop_length
        self.n_mels = Constants.n_mels
        self.preemph = Constants.preemph
        self.log_zero_guard_value = Constants.log_zero_guard_value
        self.graphics_create = graphics_create
        self.window = window
        self.center = center
        self.mel_filterbank_type = mel_filterbank_type

    def stft_numpy(self,
                   signal: np.ndarray,
                   window: str = "hann"
    ) -> np.ndarray:
        """Compute the Short-Time Fourier Transform (STFT) of an audio signal.

        Parameters
        ----------
        signal : np.ndarray
            Input audio signal, shape [time].
        n_fft : int, optional
            Number of FFT points (default: 400).
        hop_length : int, optional
            Hop length between frames in samples (default: 160).
        win_length : int, optional
            Window length in samples (default: 400).
        window : str, optional
            Window type, supports 'hann', 'hamming', 'blackman' (default: 'hann').
        center : bool, optional
            If True, pads the signal to center frames (default: True).

        Returns
        -------
        np.ndarray
            STFT matrix, shape [n_fft // 2 + 1, n_frames], dtype complex128.

        Notes
        -----
        - Supports 'hann', 'hamming', and 'blackman' window types.
        - Applies padding to center frames when center=True.
        - Returns only positive frequency components.
        """
        if self.window == 'hann':
            win = np.hanning(self.win_length)
        elif self.window == 'hamming':
            win = np.hamming(self.win_length)
        elif self.window == 'blackman':
            win = np.blackman(self.win_length)
        else:
            win = np.ones(self.win_length)

        # Padding
        if self.center:
            pad_amount = self.n_fft // 2
            signal = np.pad(signal,
                            pad_amount,
                            mode='reflect')

        # Количество фреймов
        n_frames = 1 + (len(signal) - self.n_fft) // self.hop_length

        # Инициализация результата
        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_frames),
                               dtype=np.complex128)

        # Вычисление STFT
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft

            if end <= len(signal):
                frame = signal[start:end]

                # Применяем окно
                if len(frame) == self.win_length:
                    frame = frame * win
                elif len(frame) == self.n_fft and self.win_length != self.n_fft:
                    # Дополняем или обрезаем окно
                    if self.win_length < self.n_fft:
                        win_padded = np.pad(win,
                                            (0, self.n_fft - self.win_length),
                                            mode='constant')
                    else:
                        win_padded = win[:self.n_fft]
                    frame = frame * win_padded
                else:
                    frame = frame * win

                # FFT
                fft_frame = np.fft.fft(frame,
                                       n=self.n_fft)
                stft_matrix[:, i] = fft_frame[:self.n_fft // 2 + 1]

        return stft_matrix

    def create_mel_filterbank_torch_compatible(self) -> np.ndarray:
        """Create a Mel filterbank compatible with torchaudio.MelScale.
        Returns
        -------
        np.ndarray
            Mel filterbank matrix, shape [n_mels, n_fft // 2 + 1], dtype float32.

        Notes
        -----
        - Uses the HTK Mel scale formula as in torchaudio.
        - Normalizes filters using the 'slaney' method for area normalization.
        """
        fmax = self.sample_rate / 2.0
        n_freqs = self.n_fft // 2 + 1

        # Mel scale conversion (HTK=False, как в torchaudio по умолчанию)
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        # Создаем точки в mel шкале
        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Частоты для FFT bins
        fft_freqs = np.linspace(0, fmax, n_freqs)

        # Создаем фильтр банк
        filterbank = np.zeros((self.n_mels, n_freqs))
        for m in range(self.n_mels):
            left = hz_points[m]
            center = hz_points[m + 1]
            right = hz_points[m + 2]
            for k in range(n_freqs):
                freq = fft_freqs[k]
                if left <= freq <= center and center != left:
                    filterbank[m, k] = (freq - left) / (center - left)
                elif center < freq <= right and right != center:
                    filterbank[m, k] = (right - freq) / (right - center)

        # Нормализация
        # Нормализуем каждый фильтр по его площади
        enorm = 2.0 / (hz_points[2:self.n_mels + 2] - hz_points[:self.n_mels])
        filterbank *= enorm[:, np.newaxis]
        return filterbank.astype(np.float32)

    def _create_mel_filterbank_simple(self) -> np.ndarray:
        """Create a Mel filterbank compatible with torchaudio.MelScale.

        Returns
        -------
        np.ndarray
            Mel filterbank matrix, shape [n_mels, n_fft // 2 + 1], dtype float32.

        Notes
        -----
        - Uses a simplified Mel scale formula with linear spacing.
        - Normalizes each filter to sum to 1 for consistency.
        """
        n_freqs = int(Constants.n_fft // 2 + 1)
        f_min, f_max = 0.0, Constants.sample_rate / 2.0

        # Используем формулу mel scale как в torchaudio
        mel_min = 1125.0 * np.log1p(f_min / 700.0)
        mel_max = 1125.0 * np.log1p(f_max / 700.0)
        mel_points = np.linspace(mel_min, mel_max, Constants.n_mels + 2)
        freq_points = 700.0 * (np.expm1(mel_points / 1125.0))

        # Создаем банк фильтров
        fb = np.zeros((Constants.n_mels, n_freqs))
        freqs = np.linspace(0, f_max, n_freqs)

        for m in range(Constants.n_mels):
            f_left = freq_points[m]
            f_center = freq_points[m + 1]
            f_right = freq_points[m + 2]

            for f in range(n_freqs):
                if f_left <= freqs[f] <= f_center and f_center != f_left:
                    fb[m, f] = (freqs[f] - f_left) / (f_center - f_left)
                elif f_center < freqs[f] <= f_right and f_right != f_center:
                    fb[m, f] = (f_right - freqs[f]) / (f_right - f_center)

        fb = fb / (np.sum(fb,
                          axis=1,
                          keepdims=True) + 1e-10)

        return fb.astype(np.float32)

    def extract_features(self, input_signal: np.ndarray, length: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract log-mel spectrogram features from the input audio signal.

        Parameters
        ----------
        input_signal : np.ndarray
            Input audio signal, shape [batch, channels, time].
        length : np.ndarray
            Lengths of the input audio signals, shape [batch].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Log-mel spectrogram features, shape [batch, n_mels, time_frames].
            - Output lengths after feature extraction, shape [batch].
        """
        global mel_fb
        print(f"input_signal shape: {input_signal.shape}, min: {np.min(input_signal)}, max: {np.max(input_signal)}")
        if np.isnan(input_signal).any() or np.isinf(input_signal).any():
            print("Warning: input_signal contains NaN or Inf values!")

        mean = np.mean(input_signal)
        std = np.std(input_signal)
        if std == 0:
            std = 1e-6
        print(f"Mean before CMVN: {mean}, Std before CMVN: {std}")
        input_signal = (input_signal - mean) / (std + 1e-6)

        if Constants.preemph != 0.0:
            batch_size, channels, time = input_signal.shape
            for b in range(batch_size):
                for c in range(channels):
                    signal = input_signal[b, c]
                    preemph_signal = np.concatenate([signal[:1], signal[1:] - Constants.preemph * signal[:-1]])
                    input_signal[b, c] = preemph_signal

        batch_size, channels, time = input_signal.shape
        num_frames = int(np.floor(time / Constants.hop_length) + 1)
        print(f"batch_size: {batch_size}, channels: {channels}, time: {time}, num_frames: {num_frames}")

        if num_frames <= 0:
            raise ValueError(
                f"Signal length ({time}) is too short for STFT with n_fft={Constants.n_fft} and hop_length={Constants.hop_length}. "
                f"Need at least {Constants.n_fft} samples."
            )

        if self.mel_filterbank_type == "complex":
            mel_fb = self.create_mel_filterbank_torch_compatible()
        elif self.mel_filterbank_type == "simple":
            mel_fb = self._create_mel_filterbank_simple()
        mel_specs = []

        for b in range(batch_size):
            for c in range(channels):
                signal = input_signal[b, c]
                # Выбираем окно в зависимости от self.window
                if self.window == 'hamming':
                    stft_result = self.stft_numpy(signal,
                                                  window=self.window)
                elif self.window == 'blackman':
                    stft_result = self.stft_numpy(signal,
                                                  window=self.window)
                else:
                    stft_result = self.stft_numpy(signal,
                                                  window=self.window)

                power_spec = np.abs(stft_result) ** 2
                if power_spec.shape[1] > num_frames:
                    power_spec = power_spec[:, :num_frames]
                elif power_spec.shape[1] < num_frames:
                    power_spec = np.pad(power_spec,
                                        ((0, 0),
                                         (0, num_frames - power_spec.shape[1])),
                                        mode='constant',
                                        constant_values=0)
                mel_spec = np.dot(mel_fb, power_spec)
                mel_specs.append(mel_spec)

        mel_spectrogram = np.stack(mel_specs, axis=0).reshape(batch_size,
                                                              channels,
                                                              Constants.n_mels,
                                                              num_frames)
        print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
        print(f"Melspec NumPy: min={np.min(mel_spectrogram)}, max={np.max(mel_spectrogram)}")

        log_mel_spec = np.log(mel_spectrogram + Constants.log_zero_guard_value)
        print(f"log_mel_spec NumPy: min={np.min(log_mel_spec)}, max={np.max(log_mel_spec)}")
        np.save("mel_spec_numpy_raw.npy", log_mel_spec)

        mean = np.mean(log_mel_spec,
                       axis=3,
                       keepdims=True)
        std = np.std(log_mel_spec,
                     axis=3,
                     keepdims=True)
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
        print(f"After CMVN: mean={np.mean(log_mel_spec)}, std={np.std(log_mel_spec)}")
        np.save("mel_spec_numpy_cmvn.npy", log_mel_spec)

        if channels == 1:
            log_mel_spec = log_mel_spec.squeeze(1)

        features = log_mel_spec.astype(np.float32)
        features_len = np.array([num_frames],
                                dtype=np.int64)

        if self.graphics_create:
            if not os.path.exists(graphics_dir):
                os.makedirs(graphics_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{graphics_dir}/mel_spectrogram_NumPy_{timestamp}.png"
            plt.figure(figsize=(10, 4))
            plt.imshow(features[0],
                       aspect="auto",
                       origin="lower",
                       interpolation="nearest")
            plt.colorbar(label="Normalized Log Mel Energy")
            plt.title("Mel-Spectrogram (After CMVN)")
            plt.xlabel("Time Frames")
            plt.ylabel("Mel Frequency Bins")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        return features, features_len

def load_vocab(vocab_path: str) -> List[str]:
    """Load vocabulary from a file into a list of tokens.

    Parameters
    ----------
    vocab_path : str
        Path to the vocabulary file containing token-index pairs or single tokens.

    Returns
    -------
    List[str]
        List of vocabulary tokens, padded with "<pad>" up to 1025 entries.

    Notes
    -----
    - Supports files with either 'token index' pairs or single tokens per line.
    - Pads the vocabulary with "<pad>" if the length is less than 1025.
    """
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                token, idx = parts
                idx = int(idx)
                while len(vocab) <= idx:
                    vocab.append("")
                vocab[idx] = token
            elif len(parts) == 1:
                vocab.append(parts[0])
    while len(vocab) < Constants.ru_vocab_length:
        vocab.append("<pad>")
    return vocab

class RnntASRNumPy:
    def __init__(self,
                 encoder_path: str,
                 decoder_joint_path: str,
                 vocab_path: str,
                 graphics_create: bool = False,
                 window: str = "hann",
                 center: bool = True,
                 mel_filterbank_type: str = "complex"
    ) -> None:
        """Initialize the RNN-T ASR model with ONNX encoder and decoder-joint components.

        Parameters
        ----------
        encoder_path : str
            Path to the ONNX encoder model file.
        decoder_joint_path : str
            Path to the ONNX decoder-joint model file.

        Notes
        -----
        - Sets up model inputs and outputs based on ONNX session metadata.
        - Loads vocabulary and initializes token indices for special tokens.
        """
        self.features = Constants.features
        self.hidden_size = Constants.hidden_size
        self._encoder = rt.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(decoder_joint_path, providers=["CPUExecutionProvider"])
        self.vocab = load_vocab(vocab_path)
        self._setup_token_indices()
        self._print_model_info()
        self.graphics_create = graphics_create
        self.window = window
        self.center = center
        self.mel_filterbank_type = mel_filterbank_type

        self._encoder_input_name = self._encoder.get_inputs()[0].name
        self._encoder_length_name = self._encoder.get_inputs()[1].name
        self._decoder_input_name = self._decoder_joint.get_inputs()[0].name
        self._decoder_prev_token_name = self._decoder_joint.get_inputs()[1].name
        self._decoder_state_name = self._decoder_joint.get_inputs()[2].name
        self._decoder_output_name = self._decoder_joint.get_outputs()[0].name
        self._decoder_state_out_name = self._decoder_joint.get_outputs()[1].name

        print("Decoder joint inputs:")
        for inp in self._decoder_joint.get_inputs():
            print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

    def _setup_token_indices(self) -> None:
        """Set up indices for special tokens in the vocabulary.

        Notes
        -----
        - Identifies indices for blank, unknown, and padding tokens.
        - Populates a set of tokens to filter during decoding.
        """
        self._blank_idx = self.vocab.index("<blk>") if "<blk>" in self.vocab else Constants.ru_vocab_length - 1
        self._unk_idx = self.vocab.index("<unk>") if "<unk>" in self.vocab else Constants.ru_vocab_length - 1
        self._blk_idx = self._blank_idx
        self._pad_idx = self.vocab.index("<pad>") if "<pad>" in self.vocab else Constants.ru_vocab_length - 1
        self._max_vocab_idx = len(self.vocab) - 1
        self._tokens_to_filter = {self._blank_idx, self._unk_idx, self._blk_idx, self._pad_idx}
        for i in range(self._max_vocab_idx + 1, len(self.vocab)):
            self._tokens_to_filter.add(i)

    def _print_model_info(self) -> None:
        """Print detailed information about the loaded vocabulary and token indices.

        Notes
        -----
        - Displays vocabulary size and sample tokens for debugging.
        """
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
        print(f"First 10 tokens: {self.vocab[:10]}")
        print(f"Last 10 tokens: {self.vocab[-10:]}")
        print(f"Blank index: {self._blank_idx} ('{self.vocab[self._blank_idx]}')")
        print(f"UNK index: {self._unk_idx} ('{self.vocab[self._unk_idx]}')")
        print(f"Pad index: {self._pad_idx} ('{self.vocab[self._pad_idx]}')")
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _print_tensor_info(self,
                           title: str,
                           tensors: List[rt.NodeArg]
    ) -> None:
        """Print tensor metadata for debugging purposes.

        Parameters
        ----------
        title : str
            Title to describe the tensor group.
        tensors : List[rt.NodeArg]
            List of ONNX tensor metadata objects.

        Notes
        -----
        - Displays name, shape, and type for each tensor.
        """
        print(f"{title}:")
        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    def out_len(self,
                input_lengths: np.ndarray
    ) -> np.ndarray:
        """Calculate output length after feature extraction based on input audio length.

        Parameters
        ----------
        input_lengths : np.ndarray
            Input audio lengths, shape [batch].

        Returns
        -------
        np.ndarray
            Output lengths after applying hop length, shape [batch].

        Notes
        -----
        - Uses floor division with hop_length (160) to compute frame count.
        """
        return np.floor_divide(input_lengths, Constants.hop_length) + 1

    def extract_features(self,
                         input_signal: np.ndarray,
                         length: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract log-mel spectrogram features using the NumPyPreprocessor.

        Parameters
        ----------
        input_signal : np.ndarray
            Input audio signal, shape [batch, channels, time].
        length : np.ndarray
            Lengths of the input audio signals, shape [batch].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Log-mel spectrogram features, shape [batch, n_mels, time_frames].
            - Output lengths after feature extraction, shape [batch].
        """
        preprocessor = NumPyPreprocessor(self.graphics_create,
                                         self.window,
                                         self.center,
                                         self.mel_filterbank_type)
        return preprocessor.extract_features(input_signal, length)

    def _encode(self,
                features: np.ndarray,
                features_lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input features using the ONNX encoder model.

        Parameters
        ----------
        features : np.ndarray
            Log-mel spectrogram features, shape [batch, features, time_frames].
        features_lens : np.ndarray
            Lengths of the feature sequences, shape [batch].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - Encoded output, shape [batch, hidden_size, time_frames].
            - Output lengths after encoding, shape [batch].

        Notes
        -----
        - Runs inference using the ONNX encoder session.
        """
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def _decode(self,
                prev_tokens: List[int],
                prev_state: Tuple[np.ndarray, np.ndarray],
                encoder_out: np.ndarray
    ) -> Tuple[np.ndarray, int, Tuple[np.ndarray, np.ndarray]]:
        """Decode a single step using the ONNX decoder-joint model.

        Parameters
        ----------
        prev_tokens : List[int]
            Previous decoded token sequence.
        prev_state : Tuple[np.ndarray, np.ndarray]
            Previous decoder states, shape [(1, 1, hidden_size), (1, 1, hidden_size)].
        encoder_out : np.ndarray
            Encoded output for the current time step, shape [batch, hidden_size, 1].

        Returns
        -------
        Tuple[np.ndarray, int, Tuple[np.ndarray, np.ndarray]]
            A tuple containing:
            - Logits for the current step, shape [vocab_size].
            - Placeholder return value (currently -1).
            - Updated decoder states.

        Notes
        -----
        - Uses the previous token or blank token as input.
        - Runs inference using the ONNX decoder-joint session.
        - Prints debug information for logits.
        """
        prev_token = self._blank_idx if not prev_tokens else prev_tokens[-1]
        inputs = {
            "encoder_outputs": encoder_out.astype(np.float32),
            "targets": np.array([[prev_token]], dtype=np.int32),
            "target_length": np.array([1], dtype=np.int32),
            "input_states_1": prev_state[0],
            "input_states_2": prev_state[1],
        }
        outputs = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            inputs
        )
        logits = np.squeeze(outputs[0])
        print(f"Debug: logits shape={logits.shape}, min={logits.min()}, max={logits.max()}")
        return logits, -1, (outputs[1], outputs[2])

    def decode_rnnt_greedy_improved(self,
                                    encoder_output: np.ndarray,
                                    ground_truth: str = None,
                                    state_init: str = "zero",
                                    clean_transcription: bool = True
    ) -> Tuple[str, Dict[str, float], List[int]]:
        """Perform greedy decoding to transcribe encoded audio output.

        Parameters
        ----------
        encoder_output : np.ndarray
            Encoded output from the encoder, shape [batch, hidden_size, time_frames].
        ground_truth : str, optional
            Ground truth transcription for metric computation.
        state_init : str, optional
            Initialization method for decoder states ('zero' or 'random').

        Returns
        -------
        Tuple[str, Dict[str, float], List[int]]
            A tuple containing:
            - Transcribed text.
            - Metrics dictionary if ground truth is provided.
            - List of timestamps corresponding to decoded tokens.

        Notes
        -----
        - Applies blank and repeat penalties during decoding.
        - Uses `_postprocess_uncleaned` for final text cleanup.
        """
        max_len = encoder_output.shape[2]
        print(f"Starting improved greedy decoding with {max_len} time frames")

        if state_init == "random":
            state = (np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32),
                     np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32))
        else:
            state = (np.zeros((1, 1, self.hidden_size), dtype=np.float32),
                     np.zeros((1, 1, self.hidden_size), dtype=np.float32))

        hyp = []
        blank_penalty = -0.5
        repeat_penalty = 0.1

        for t in range(max_len):
            current_encoder_out = encoder_output[:, :, t:t + 1]
            logits, _, state = self._decode(hyp, state, current_encoder_out)
            logits = logits.copy()

            # Применяем штрафы как в PyTorch версии
            if self._blank_idx < len(logits):
                logits[self._blank_idx] += blank_penalty
            if self._pad_idx < len(logits):
                logits[self._pad_idx] -= 5.0

            # Нормализация логитов
            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-12)

            # Штраф за повторы
            if hyp and hyp[-1] != self._blank_idx:
                probs[hyp[-1]] *= repeat_penalty

            next_token = np.argmax(probs).item()
            if next_token != self._blank_idx and (not hyp or next_token != hyp[-1]):
                hyp.append(next_token)

            if t % 5 == 0:
                token_str = self.vocab[next_token] if next_token < len(self.vocab) else 'OUT_OF_VOCAB'
                print(f"Step {t}: token={next_token}('{token_str}'), prob={probs[next_token]:.3f}")

        if clean_transcription == False:
            text = self._postprocess_uncleaned(hyp, "greedy")
        else:
            text = self._postprocess_cleaned(hyp, "greedy")
        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(transcription=text,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=0.0,
                                     flag="greedy")
        return text, metrics, [t for t in range(len(hyp))]

    def decode_rnnt_beam_search_fixed(self,
                                      encoder_output: np.ndarray,
                                      vocab: List[str],
                                      blank_idx: int,
                                      beam_width: int = 8,
                                      length_penalty: float = 0.7,
                                      ground_truth: str = None,
                                      max_steps: int = 1000,
                                      min_tokens: int = 18,
                                      state_init: str = "zero",
                                      clean_transcription: bool = True
    ) -> Tuple[str, Dict[str, float], List[int]]:
        """Perform fixed beam search decoding to transcribe encoded audio output.

        Parameters
        ----------
        encoder_output : np.ndarray
            Encoded output from the encoder, shape [batch, hidden_size, time_frames].
        vocab : List[str]
            Vocabulary list for token mapping.
        blank_idx : int
            Index of the blank token in the vocabulary.
        beam_width : int, optional
            Number of beams to maintain during search (default: 8).
        length_penalty : float, optional
            Penalty factor for sequence length (default: 0.7).
        ground_truth : str, optional
            Ground truth transcription for metric computation.
        max_steps : int, optional
            Maximum number of decoding steps (default: 1000).
        min_tokens : int, optional
            Minimum number of tokens to continue search (default: 18).
        state_init : str, optional
            Initialization method for decoder states ('zero' or 'random').

        Returns
        -------
        Tuple[str, Dict[str, float], List[int]]
            A tuple containing:
            - Transcribed text.
            - Metrics dictionary if ground truth is provided.
            - List of timestamps corresponding to decoded tokens.

        Notes
        -----
        - Implements beam search with temperature scaling and penalties.
        - Prunes beams based on unique sequences and scores.
        - Uses `_postprocess_uncleaned` for final text cleanup.
        """
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        print(f"Starting fixed beam search: beam_width={beam_width}, time_frames={time_frames}")

        if state_init == "random":
            state = (np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32),
                     np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32))
        else:
            state = (np.zeros((1, 1, self.hidden_size), dtype=np.float32),
                     np.zeros((1, 1, self.hidden_size), dtype=np.float32))

        beams = [(tuple(), 0.0, state, [], 0, 0, [])]
        temperature = 0.8
        blank_penalty = -2.0
        repeat_penalty = 0.05
        step = 0
        max_iterations = min(time_frames * 4, max_steps)

        while step < max_iterations:
            new_beams = []

            for seq, score, (state1, state2), prev_tokens, curr_t, token_count, timestamps in beams:
                if curr_t >= time_frames:
                    new_beams.append((seq, score, (state1, state2), prev_tokens, curr_t, token_count, timestamps))
                    continue

                current_encoder_out = encoder_output[:, :, curr_t:curr_t + 1]
                if current_encoder_out.shape[2] == 0:
                    print(f"Warning: encoder_output_t is empty at t={curr_t}, skipping this beam")
                    new_beams.append((seq, score, (state1, state2), prev_tokens, curr_t + 1, token_count, timestamps))
                    continue

                logits, _, (new_state1, new_state2) = self._decode(prev_tokens, (state1, state2), current_encoder_out)
                logits_t = logits.copy()

                if logits_t.size == 0:
                    print(f"Warning: logits are empty at t={curr_t}, skipping this beam")
                    new_beams.append((seq, score, (state1, state2), prev_tokens, curr_t + 1, token_count, timestamps))
                    continue

                print(f"DEBUG: t={curr_t}, raw logits_t for blank {self._blank_idx}: {logits_t[self._blank_idx]:.4f}")
                non_blank_logits = np.delete(logits_t,
                                             self._blank_idx)
                non_blank_vocab_indices = np.delete(np.arange(len(self.vocab)), self._blank_idx)
                k_val = min(5, len(non_blank_logits))
                top_5_raw_indices_in_non_blank_array = np.argsort(non_blank_logits)[-k_val:]
                top_5_raw_indices = non_blank_vocab_indices[top_5_raw_indices_in_non_blank_array]
                print("DEBUG: t={curr_t}, top 5 raw non-blank logits: " +
                      ", ".join([f"{self.vocab[idx]}:{logits_t[idx]:.4f}" for idx in top_5_raw_indices]))

                if blank_idx < len(logits_t):
                    logits_t[blank_idx] += blank_penalty
                if self._pad_idx < len(logits_t):
                    logits_t[self._pad_idx] -= 5.0

                if temperature != 1.0:
                    logits_t = logits_t / temperature
                logits_t = logits_t - np.max(logits_t)
                exp_logits = np.exp(logits_t)
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)

                if len(seq) > 0:
                    last_token = seq[-1]
                    if last_token != blank_idx:
                        probs[last_token] *= repeat_penalty

                top_indices = np.argsort(probs)[-beam_width * 2:][::-1]
                for token in top_indices:
                    prob = probs[token]
                    new_score = score + np.log(prob + 1e-12)
                    new_seq = seq + (token,)
                    new_timestamps = timestamps + [curr_t]
                    new_count = token_count + (1 if token != blank_idx else 0)
                    new_prev_tokens = prev_tokens + [token]

                    new_beams.append((new_seq, new_score, (new_state1, new_state2), new_prev_tokens, curr_t + 1,
                                      new_count, new_timestamps))

            unique_beams = {}
            for beam in new_beams:
                key = (beam[0], tuple(beam[3]))
                if key not in unique_beams or beam[1] > unique_beams[key][1]:
                    unique_beams[key] = beam

            beams = sorted(list(unique_beams.values()), key=lambda x: x[1], reverse=True)[:beam_width]
            step += 1

            if all(beam[4] >= time_frames for beam in beams) and step > min_tokens:
                print(f"All beams completed at step {step}")
                break

        if not beams:
            return "", {}, []

        best_beam = beams[0]
        best_seq, best_score, _, _, _, token_count, timestamps = best_beam

        print(f"Best raw sequence (token IDs): {best_seq}")
        print(f"Mapped raw sequence: {[vocab[idx] for idx in best_seq if idx < len(vocab)]}")

        if clean_transcription == False:
            text = self._postprocess_uncleaned(list(best_seq), "beam")
        else:
            text = self._postprocess_cleaned(list(best_seq), "beam")

        print(f"Fixed Beam Search completed:")
        print(f"  Transcription: '{text}'")
        print(f"  Log probability: {best_score:.6f}")
        normalized_score = best_score / (max(1, token_count) ** length_penalty) if token_count > 0 else best_score
        print(f"  Normalized score: {normalized_score:.6f}")
        print(f"  Total tokens: {token_count}")
        print(f"  Steps: {step}")

        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(
                transcription=text,
                ground_truth=ground_truth,
                metrics=metrics,
                total_log_prob=best_score,
                beam_width=beam_width,
                length_penalty=length_penalty,
                flag="beam"
            )

        return text, metrics, timestamps

    def _postprocess_uncleaned(self,
                              decoded_ids: List[int],
                              flag: str = "greedy"
    ) -> str:
        """Postprocess decoded token IDs into a cleaned text string.

        Parameters
        ----------
        decoded_ids : List[int]
            List of decoded token IDs from the model.
        flag : str, optional
            Flag to indicate decoding method ('greedy' or beam') for logging.

        Returns
        -------
        str
            Cleaned and formatted transcription text.

        Notes
        -----
        - Filters out special tokens and applies word deduplication.
        - Normalizes whitespace and preserves double characters.
        - Capitalizes the first letter for readability.
        """
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if
                        tok_id < len(self.vocab) and tok_id not in {self._blank_idx, self._pad_idx}]
        text = "".join(valid_tokens).replace("▁", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Сохраняем двойные символы, как в PyTorch

        words = text.split()
        cleaned_words = []
        last_word = None
        for word in words:
            if word and (not last_word or word.lower() != last_word.lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word

        text = " ".join(cleaned_words).strip()
        text = re.sub(r'(?<=\s)[.,!?]+$', '', text).strip()
        text = text[0].upper() + text[1:] if text else text

        if flag == "greedy":
            print(f"Final transcription (Improved Greedy Decoding): '{text}'")
        elif flag == "beam":
            print(f"Final transcription (Improved Beam Search): '{text}'")
        return text

    def _postprocess_cleaned(self,
                                decoded_ids: List[int],
                                flag: str = "greedy"
    ) -> str:
        """Alternative postprocessing method for decoded token IDs.

        Parameters
        ----------
        decoded_ids : List[int]
            List of decoded token IDs from the model.
        flag : str, optional
            Flag to indicate decoding method ('greedy' or 'beam') for logging.

        Returns
        -------
        str
            Cleaned and formatted transcription text.

        Notes
        -----
        - Similar to `_postprocess_uncleaned` but with stricter punctuation removal.
        - Preserves double characters and normalizes whitespace.
        """
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if
                        tok_id < len(self.vocab) and tok_id not in {self._blank_idx, self._pad_idx}]
        text = "".join(valid_tokens).replace("▁", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        words = text.split()
        cleaned_words = []
        last_word = None
        for word in words:
            if word and (not last_word or word.lower() != last_word.lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word

        text = " ".join(cleaned_words).strip()
        text = re.sub(r'[.,!?]$', '', text).strip()

        if flag == "greedy":
            print(f"Final transcription (Improved Greedy Decoding): '{text}'")
        elif flag == "beam":
            print(f"Final transcription (Improved Beam Search): '{text}'")
        return text

    def _get_log_probs(self,
                       full_encoder_out: np.ndarray,
                       t: int,
                       prev_token: int,
                       state: tuple
    ) -> Tuple[np.ndarray, tuple]:
        """Compute log probabilities for a single decoding step.

        Parameters
        ----------
        full_encoder_out : np.ndarray
            Full encoded output, shape [batch, hidden_size, time_frames].
        t : int
            Current time step index.
        prev_token : int
            Previous token ID.
        state : tuple
            Previous decoder states, shape [(1, 1, hidden_size), (1, 1, hidden_size)].

        Returns
        -------
        Tuple[np.ndarray, tuple]
            A tuple containing:
            - Log probabilities, shape [vocab_size].
            - Updated decoder states.

        Notes
        -----
        - Slices encoder output for the current time step.
        - Runs inference using the ONNX decoder-joint session.
        - Normalizes logits to compute log probabilities safely.
        """
        decoder_input = np.array([[prev_token]],
                                 dtype=np.int32)

        inputs = {
            "encoder_outputs": full_encoder_out[:, :, t:t + 1].astype(np.float32),
            "targets": decoder_input,
            "target_length": np.array([1],
                                      dtype=np.int32),
            "input_states_1": state[0],
            "input_states_2": state[1],
        }

        outputs = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            inputs
        )

        logits = np.squeeze(outputs[0])
        print(f"Debug: _get_log_probs t={t}, logits shape={logits.shape}, min={logits.min()}, max={logits.max()}")

        new_state = (outputs[1], outputs[2])

        exp_logits = np.exp(logits - np.max(logits))
        eps = 1e-10
        log_probs = np.log(exp_logits / (np.sum(exp_logits) + eps))

        return log_probs, new_state

    def decode_rnnt_beam_search_advanced(
            self,
            encoder_output: np.ndarray,
            beam_width: int = 5,
            length_penalty: float = 0.6,
            ground_truth: str = None
    ) -> Tuple[str, Dict, List[int]]:
        """Perform advanced beam search decoding to transcribe encoded audio output.

        Parameters
        ----------
        encoder_output : np.ndarray
            Encoded output from the encoder, shape [batch, hidden_size, time_frames].
        beam_width : int, optional
            Number of beams to maintain during search (default: 5).
        length_penalty : float, optional
            Penalty factor for sequence length (default: 0.6).
        ground_truth : str, optional
            Ground truth transcription for metric computation.

        Returns
        -------
        Tuple[str, Dict, List[int]]
            A tuple containing:
            - Transcribed text.
            - Metrics dictionary (currently empty).
            - List of timestamps corresponding to decoded tokens.

        Notes
        -----
        - Implements a prefix-based beam search with temperature scaling.
        - Applies blank penalty to encourage non-blank tokens.
        - Uses `_postprocess_final` for final text cleanup.
        """
        T = encoder_output.shape[2]  # Time dimension from encoder output

        # Initial state
        initial_state = (
            np.zeros((1, 1, self.hidden_size), dtype=np.float32),
            np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        )

        # B - dictionary of {prefix: (log_prob, state, non_blank_count)}
        B = {tuple(): (0.0, initial_state, 0)}

        temperature = 0.9  # Add temperature scaling to prevent underflow
        blank_penalty = -1.5  # Less aggressive penalty to encourage non-blank tokens

        for t in range(T):
            A = defaultdict(lambda: (-np.inf, None, 0))

            sorted_beams = sorted(B.items(), key=lambda x: x[1][0], reverse=True)

            for prefix, (log_prob, state, non_blank_count) in sorted_beams[:beam_width]:
                log_probs, new_state = self._get_log_probs(
                    encoder_output,
                    t,
                    prefix[-1] if prefix else self._blank_idx,
                    state
                )

                # Apply temperature scaling to logits
                log_probs = log_probs / temperature

                # Safe log probability computation
                exp_logits = np.exp(log_probs - np.max(log_probs))
                eps = 1e-10
                norm_factor = np.sum(exp_logits) + eps
                log_probs = np.log(exp_logits / norm_factor)

                # Apply blank penalty
                log_probs[self._blank_idx] += blank_penalty

                # Blank extension
                blank_log_prob = log_probs[self._blank_idx]
                p_blank = log_prob + blank_log_prob
                if p_blank > A[prefix][0]:
                    A[prefix] = (p_blank, state, non_blank_count)

                # Non-blank extensions
                non_blank_indices = [i for i in range(len(log_probs)) if i != self._blank_idx]
                non_blank_log_probs = log_probs[non_blank_indices]
                top_k_indices = np.argsort(non_blank_log_probs)[-beam_width:] if non_blank_log_probs.size > 0 else []

                for k in top_k_indices:
                    original_k = non_blank_indices[k]
                    if original_k >= len(self.vocab):
                        continue
                    new_prefix = prefix + (original_k,)
                    p_non_blank = log_prob + log_probs[original_k]
                    new_non_blank_count = non_blank_count + 1
                    if p_non_blank > B.get(new_prefix, (-np.inf, None, 0))[0]:
                        B[new_prefix] = (p_non_blank, new_state, new_non_blank_count)

            # Merge with blank extensions
            for prefix, (p, s, nbc) in A.items():
                if p > B.get(prefix, (-np.inf, None, 0))[0]:
                    B[prefix] = (p, s, nbc)

            # Pruning with length penalty and diversity
            sorted_B = sorted(B.items(),
                              key=lambda x: (x[1][0] / (max(1, len(x[0])) ** length_penalty + 1e-6), x[1][2]),
                              reverse=True)
            B = dict(sorted_B[:beam_width])

        # Select best hypothesis
        if not B:
            return "", {}, []
        best_seq_tuple, (best_score, _, non_blank_count) = max(B.items(),
                                                               key=lambda x: x[1][0] / (
                                                                       max(1, len(x[0])) ** length_penalty + 1e-6))
        best_seq = list(best_seq_tuple)
        text = self._postprocess_final(best_seq, "BS_ADVANCED")

        print(f"Advanced Beam Search completed:")
        print(f"  Transcription: '{text}'")
        print(f"  Log probability: {best_score:.6f}")
        print(f"  Non-blank tokens: {non_blank_count}")

        metrics = {}
        timestamps = list(range(len(best_seq)))  # Simplified timestamps

        return text, metrics, timestamps

    def _postprocess_final(self,
                           decoded_ids: List[int],
                           flag: str = "Final"
    ) -> str:
        """Final postprocessing of decoded token IDs into a cleaned text string.

        Parameters
        ----------
        decoded_ids : List[int]
            List of decoded token IDs from the model.
        flag : str, optional
            Flag to indicate decoding method for logging (default: 'Final').

        Returns
        -------
        str
            Cleaned and formatted transcription text.

        Notes
        -----
        - Removes special tokens and applies strict deduplication.
        - Normalizes whitespace and punctuation.
        - Capitalizes the first letter for readability.
        """
        tokens = [self.vocab[tok_id] for tok_id in decoded_ids if
                  tok_id not in {self._blank_idx, self._pad_idx, self._unk_idx}]
        text = "".join(tokens).replace(" ", " ").strip()
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        if not words: return ""
        cleaned_words = [words[0]]
        for i in range(1, len(words)):
            if words[i].lower() != words[i - 1].lower():
                cleaned_words.append(words[i])
        text = " ".join(cleaned_words)
        if text: text = text[0].upper() + text[1:]
        print(f"Final transcription ({flag}): '{text}'")
        return text

    def recognize(self,
                  waveforms: np.ndarray,
                  decode_flag: str = "greedy",
                  ground_truth: str = None,
                  max_steps: int = 3000,
                  min_tokens: int = 15,
                  state_init: str = "zero",
                  beam_width: int = 8,
                  length_penalty: float = 0.7,
                  clean_transcription: bool = True
    ) -> Tuple[str, List[int]]:
        """Recognize speech from raw audio waveforms using the specified decoding method.

        Parameters
        ----------
        waveforms : np.ndarray
            Raw audio waveforms, expected shape [time] or [channels, time].
        decode_flag : str, optional
            Decoding method ('greedy', 'beam', or 'BS_ADVANCED') (default: 'greedy').
        ground_truth : str, optional
            Ground truth transcription for metric computation.
        max_steps : int, optional
            Maximum number of decoding steps for beam search (default: 3000).
        min_tokens : int, optional
            Minimum number of tokens for beam search to continue (default: 15).
        state_init : str, optional
            Initialization method for decoder states ('zero' or 'random') (default: 'zero').
        beam_width : int, optional
            Number of beams for beam search (default: 8).
        length_penalty : float, optional
            Penalty factor for sequence length in beam search (default: 0.7).

        Returns
        -------
        Tuple[str, List[int]]
            A tuple containing:
            - Transcribed text.
            - List of timestamps corresponding to decoded tokens.

        Notes
        -----
        - Preprocesses waveforms into features using `extract_features`.
        - Supports greedy and beam search decoding methods.
        - Visualizes the spectrogram for inspection.
        """
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")
        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        if waveforms.ndim == 2 and waveforms.shape[0] > 1:
            waveforms = np.mean(waveforms, axis=0)
        elif waveforms.ndim == 2:
            waveforms = waveforms[0]
        waveforms = waveforms[np.newaxis, np.newaxis, :]  # [1, 1, samples]

        audio_length = np.array([waveforms.shape[-1]], dtype=np.int64)
        print(
            f"waveforms shape: {waveforms.shape}, audio_length: {audio_length}, min: {np.min(waveforms)}, max: {np.max(waveforms)}")
        if np.isnan(waveforms).any() or np.isinf(waveforms).any():
            print("Warning: waveforms contains NaN or Inf values!")

        predicted_out_len = self.out_len(audio_length)
        print(f"Predicted output length: {predicted_out_len}")

        features, features_len = self.extract_features(waveforms, audio_length)
        print(f"Features shape: {features.shape}, features_len: {features_len}")
        if np.isnan(features).any() or np.isinf(features).any():
            print("Warning: features contain NaN or Inf values!")

        if self.graphics_create:
            if not os.path.exists(graphics_dir):
                os.makedirs(graphics_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{graphics_dir}/mel_spectrogram_CMVN_NumPy_{timestamp}.png"
            plt.figure(figsize=(10, 4))
            plt.imshow(features[0],
                       aspect="auto",
                       origin="lower",
                       interpolation="nearest")
            plt.colorbar(label="Normalized Log Mel Energy")
            plt.title("Mel-Spectrogram (After CMVN)")
            plt.xlabel("Time Frames")
            plt.ylabel("Mel Frequency Bins")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        features = features.astype(np.float32)
        features_len = features_len.astype(np.int64)

        encoder_out_data, encoder_out_lengths = self._encode(features, features_len)
        print(f"Encoder output data shape: {encoder_out_data.shape}, lengths: {encoder_out_lengths}")

        if decode_flag == "greedy":
            transcription, _, timestamps = self.decode_rnnt_greedy_improved(encoder_out_data,
                                                                            ground_truth,
                                                                            state_init,
                                                                            clean_transcription)
        elif decode_flag == "beam":
            transcription, _, timestamps = self.decode_rnnt_beam_search_fixed(encoder_out_data,
                                                                              self.vocab,
                                                                              self._blank_idx,
                                                                              beam_width,
                                                                              length_penalty,
                                                                              ground_truth,
                                                                              max_steps,
                                                                              min_tokens,
                                                                              state_init,
                                                                              clean_transcription)

        elif decode_flag == "BS_ADVANCED":
            transcription, _, timestamps = self.decode_rnnt_beam_search_advanced(
                encoder_out_data,
                beam_width=beam_width,
                length_penalty=length_penalty,
                ground_truth=ground_truth,
            )
        else:
            raise ValueError("decode_flag must be 'greedy', 'beam' or 'BS_ADVANCED'")

        return transcription, timestamps