from typing import Tuple, List, Dict
import numpy as np
import onnxruntime as rt

from MetricsClass import return_metrics

# Загружаем словарь из файла
def load_vocab(vocab_path: str) -> List[str]:
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token, idx = line.strip().split()
            idx = int(idx)
            while len(vocab) <= idx:
                vocab.append("")
            vocab[idx] = token
    return vocab

VOCAB = load_vocab("onnx_models/vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt")
BLANK_IDX = VOCAB.index("<blk>")
MAX_VOCAB_IDX = len(VOCAB) - 1

class RnntASRNumPy:
    """
    A module for RNN-T based Automatic Speech Recognition (ASR) using NumPy and ONNX models.

    This module processes raw audio signals, extracts logarithmic Mel spectrogram features,
    and performs transcription using an RNN-T model (encoder + decoder/joint) with
    greedy decoding or beam search.
    """
    def __init__(self,
                 encoder_path: str,
                 decoder_joint_path: str,
                 sample_rate: int = 16000,
                 features: int = 64) -> None:
        """
        Initialize the GigaamRnntASRNumPy with ONNX encoder and decoder/joint models.

        Parameters
        ----------
        encoder_path : str
            Path to the ONNX encoder model file.
        decoder_joint_path : str
            Path to the ONNX decoder/joint model file.
        sample_rate : int, optional
            Sample rate of the audio. Defaults to 16000 Hz.
        features : int, optional
            Number of Mel features. Defaults to 64.
        """
        self.sample_rate = sample_rate
        self.features = features
        self.hop_length = sample_rate // 100  # 160
        self.n_fft = sample_rate // 40  # 400
        self.win_length = sample_rate // 40  # 400

        # Загружаем ONNX модели
        self._encoder = rt.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(decoder_joint_path, providers=["CPUExecutionProvider"])

        # Проверяем входы и выходы моделей
        self.tensor_info("Encoder Inputs", self._encoder.get_inputs())
        self.tensor_info("Encoder Outputs", self._encoder.get_outputs())
        self.tensor_info("Decoder/Joint Inputs", self._decoder_joint.get_inputs())
        self.tensor_info("Decoder/Joint Outputs", self._decoder_joint.get_outputs())

        # Имена входов и выходов
        self._encoder_input_name = self._encoder.get_inputs()[0].name
        self._encoder_length_name = self._encoder.get_inputs()[1].name
        self._decoder_input_name = self._decoder_joint.get_inputs()[0].name
        self._decoder_prev_token_name = self._decoder_joint.get_inputs()[1].name
        self._decoder_state_name = self._decoder_joint.get_inputs()[2].name
        self._decoder_output_name = self._decoder_joint.get_outputs()[0].name
        self._decoder_state_out_name = self._decoder_joint.get_outputs()[1].name

        # Размер скрытого состояния (предполагаем; уточните из модели)
        self.hidden_size = 320  # Обычно для FastConformer моделей

        # Создание банка Mel-фильтров
        self.mel_fb = self._create_mel_filterbank()

        self.vocab = VOCAB
        self.blank_idx = BLANK_IDX
        self.max_vocab_idx = MAX_VOCAB_IDX

    def _create_mel_filterbank(self) -> np.ndarray:
        """
        Creates a bank of Mel filters for converting the power spectrum into a Mel spectrogram.

        Returns
        -------
        np.ndarray
            Matrix of Mel filters, shape [n_mels, n_freqs], where n_mels = self.features,
            n_freqs = self.n_fft // 2 + 1.
        """
        n_freqs = int(self.n_fft // 2 + 1)
        f_min, f_max = 0.0, self.sample_rate / 2.0

        # Перевод частот в Mel-шкалу: mel = 1125 * ln(1 + f/700)
        mel_min = 1125.0 * np.log1p(f_min / 700.0)
        mel_max = 1125.0 * np.log1p(f_max / 700.0)

        mel_points = np.linspace(mel_min, mel_max, self.features + 2)
        freq_points = 700.0 * (np.expm1(mel_points / 1125.0))

        fb = np.zeros((self.features, n_freqs))
        freqs = np.linspace(0, f_max, n_freqs)

        for m in range(self.features):
            f_left = freq_points[m]
            f_center = freq_points[m + 1]
            f_right = freq_points[m + 2]
            for f in range(n_freqs):
                if f_left <= freqs[f] <= f_center:
                    fb[m, f] = (freqs[f] - f_left) / (f_center - f_left)
                elif f_center < freqs[f] <= f_right:
                    fb[m, f] = (f_right - freqs[f]) / (f_right - f_center)
        return fb

    def forward(self,
                input_signal: np.ndarray,
                length: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract logarithmic Mel spectrogram features from the input audio signal.

        Parameters
        ----------
        input_signal : np.ndarray
            Audio input signal, shape [batch_size, channels, time] or [batch_size, time].
        length : np.ndarray
            Lengths of the input signals, shape [batch_size].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Mel spectrogram, shape [batch_size, n_mels, time_frames] or [batch_size, channels, n_mels, time_frames].
            - Output lengths, shape [batch_size].
        """
        if input_signal.ndim == 2:
            input_signal = input_signal[:, np.newaxis, :]

        batch_size, channels, time = input_signal.shape
        num_frames = (time - self.n_fft) // self.hop_length + 1
        if num_frames <= 0:
            raise ValueError(f"Signal length ({time}) is too short for STFT with n_fft={self.n_fft} "
                             f"and hop_length={self.hop_length}.")

        # Окно Ханна
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
                        frame = np.pad(frame, (0, self.n_fft - len(frame)), mode='constant')
                    frame = frame * window[:len(frame)]
                    fft = np.fft.rfft(frame, n=self.n_fft)
                    power = np.abs(fft) ** 2
                    frames.append(power)
                if not frames:
                    raise ValueError("No frames extracted.")
                spectrogram.append(np.stack(frames, axis=0))

        spectrogram = np.stack(spectrogram, axis=0).reshape(batch_size, channels, num_frames, -1)
        mel_spec = np.matmul(spectrogram, self.mel_fb.T)
        mel_spec = mel_spec.transpose(0, 1, 3, 2)
        mel_spec = np.log(np.clip(mel_spec, 1e-9, 1e9))

        if channels == 1:
            mel_spec = mel_spec.squeeze(1)

        return mel_spec, self.out_len(length)

    def out_len(self, input_lengths: np.ndarray) -> np.ndarray:
        """
        Calculate the output length after feature extraction.

        Parameters
        ----------
        input_lengths : np.ndarray
            Input lengths, shape [batch_size].

        Returns
        -------
        np.ndarray
            Output lengths, shape [batch_size].
        """
        return (input_lengths // self.hop_length + 1).astype(np.int64)

    def tensor_info(self, title: str, tensors: List[rt.NodeArg]) -> None:
        """
        Print tensor information for ONNX model inputs/outputs.

        Parameters
        ----------
        title : str
            Title for the tensor info (e.g., "Encoder Inputs").
        tensors : List[rt.NodeArg]
            List of tensor nodes.
        """
        print(f"{title}:")
        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    def encode(self, features: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Run the encoder on the extracted features to get hidden states.

        Parameters
        ----------
        features : np.ndarray
            Mel spectrogram features, shape [batch_size, n_mels, time_frames].
        lengths : np.ndarray
            Lengths of the features, shape [batch_size].

        Returns
        -------
        np.ndarray
            Encoder outputs (hidden states), shape [batch_size, time_frames, hidden_size].
        """
        inputs = {
            self._encoder_input_name: features,
            self._encoder_length_name: lengths
        }
        encoder_output = self._encoder.run([self._encoder.get_outputs()[0].name], inputs)[0]
        return encoder_output

    def decode_step(self,
                    encoder_output: np.ndarray,
                    time_step: int,
                    prev_token: np.ndarray,
                    state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one decoding step using the decoder/joint network.

        Parameters
        ----------
        encoder_output : np.ndarray
            Encoder outputs, shape [batch_size, time_frames, hidden_size].
        time_step : int
            Current time step.
        prev_token : np.ndarray
            Previous token, shape [batch_size, 1].
        state : np.ndarray
            Decoder state, shape [batch_size, 1, hidden_size].

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Logits, shape [batch_size, 1, vocab_size].
            - New state, shape [batch_size, 1, hidden_size].
        """
        encoder_frame = encoder_output[:, time_step:time_step+1, :]
        inputs = {
            self._decoder_input_name: encoder_frame,
            self._decoder_prev_token_name: prev_token,
            self._decoder_state_name: state
        }
        logits, new_state = self._decoder_joint.run(
            [self._decoder_output_name, self._decoder_state_out_name],
            inputs
        )
        return logits, new_state

    def decode_rnnt_greedy(self,
                           encoder_output: np.ndarray,
                           vocab: List[str],
                           blank_idx: int,
                           max_vocab_idx: int,
                           ground_truth: str = None) -> Tuple[str, Dict[str, float]]:
        """
        Perform greedy decoding on RNN-T encoder outputs.

        Parameters
        ----------
        encoder_output : np.ndarray
            Encoder outputs, shape [batch_size, time_frames, hidden_size].
        vocab : List[str]
            Vocabulary list.
        blank_idx : int
            Blank token index.
        max_vocab_idx : int
            Maximum valid token index.
        ground_truth : str, optional
            Ground truth for metrics.

        Returns
        -------
        Tuple[str, Dict[str, float]]
            - Transcription.
            - Metrics (if ground_truth is provided).
        """
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        T = encoder_output.shape[1]
        state = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        prev_token = np.array([[0]], dtype=np.int64)
        decoded_ids = []
        total_log_prob = 0.0

        for t in range(T):
            logits, state = self.decode_step(encoder_output, t, prev_token, state)
            logits = logits[0, 0]
            token_idx = np.argmax(logits, axis=-1).item()
            log_prob = logits[token_idx]
            total_log_prob += float(log_prob)

            if token_idx != blank_idx and token_idx <= max_vocab_idx:
                decoded_ids.append(token_idx)
            prev_token = np.array([[token_idx]], dtype=np.int64)

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

    def decode_rnnt_beam_search(self,
                                encoder_output: np.ndarray,
                                vocab: List[str],
                                blank_idx: int,
                                max_vocab_idx: int,
                                beam_width: int = 3,
                                length_penalty: float = 1.0,
                                ground_truth: str = None) -> Tuple[str, Dict[str, float]]:
        """
        Perform beam search decoding on RNN-T encoder outputs.

        Parameters
        ----------
        encoder_output : np.ndarray
            Encoder outputs, shape [batch_size, time_frames, hidden_size].
        vocab : List[str]
            Vocabulary list.
        blank_idx : int
            Blank token index.
        max_vocab_idx : int
            Maximum valid token index.
        beam_width : int, optional
            Number of beams. Defaults to 3.
        length_penalty : float, optional
            Length penalty. Defaults to 1.0.
        ground_truth : str, optional
            Ground truth for metrics.

        Returns
        -------
        Tuple[str, Dict[str, float]]
            - Transcription.
            - Metrics (if ground_truth is provided).
        """
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        T = encoder_output.shape[1]
        state = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        beams = [(tuple(), 0.0, state, np.array([[0]], dtype=np.int64))]
        num_classes = max_vocab_idx + 1

        for t in range(T):
            new_beams = {}
            for seq, score, state, prev_token in beams:
                logits, new_state = self.decode_step(encoder_output, t, prev_token, state)
                logits = logits[0, 0]
                logits = logits - np.max(logits)
                log_probs = logits - np.log(np.sum(np.exp(logits)))

                for token in range(num_classes):
                    if token > max_vocab_idx and token != blank_idx:
                        continue
                    token_log_prob = float(log_probs[token])
                    new_score = score + token_log_prob
                    new_seq = list(seq)

                    if token != blank_idx:
                        new_seq.append(token)
                    new_seq = tuple(new_seq)
                    if new_seq in new_beams:
                        existing_score = new_beams[new_seq][1]
                        new_beams[new_seq] = (
                            new_seq,
                            np.logaddexp(existing_score, new_score),
                            new_state,
                            np.array([[token]], dtype=np.int64)
                        )
                    else:
                        new_beams[new_seq] = (new_seq, new_score, new_state, np.array([[token]], dtype=np.int64))

            beams = sorted(new_beams.values(),
                           key=lambda x: x[1] / (len(x[0]) ** length_penalty if len(x[0]) > 0 else 1.0),
                           reverse=True)[:beam_width]

        best_seq, best_score, _, _ = beams[0]
        transcription = "".join(vocab[tok] for tok in best_seq)
        print(f"Transcription (Beam Search, beam_width={beam_width}, length_penalty={length_penalty}): {transcription}")
        print(f"Log probability (Beam Search): {best_score:.14f}")
        metrics = {}
        if ground_truth:
            metrics = return_metrics(transcription=transcription,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=best_score,
                                     beam_width=beam_width,
                                     length_penalty=length_penalty,
                                     flag="beam")
        return transcription, metrics

    def recognize(self,
                  waveforms: np.ndarray,
                  decode_flag: str = "GD",
                  beam_width: int = 10,
                  length_penalty: float = 0.7,
                  ground_truth: str = None) -> Tuple[str, Dict[str, float]]:
        """
        Recognize speech from the input waveform.

        Parameters
        ----------
        waveforms : np.ndarray
            Input waveform, shape [channels, samples] or [samples].
        decode_flag : str, optional
            Decoding method: "GD" or "BS". Defaults to "GD".
        beam_width : int, optional
            Number of beams. Defaults to 10.
        length_penalty : float, optional
            Length penalty. Defaults to 0.7.
        ground_truth : str, optional
            Ground truth for metrics.

        Returns
        -------
        Tuple[str, Dict[str, float]]
            - Transcription.
            - Metrics (if ground_truth is provided).
        """
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")

        if waveforms.dtype != np.float32:
            print(f"Warning: waveforms dtype is {waveforms.dtype}, expected np.float32. Casting.")
            waveforms = waveforms.astype(np.float32)

        if waveforms.ndim == 1:
            waveforms = waveforms[np.newaxis, :]
        elif waveforms.ndim != 2:
            raise ValueError(f"Expected waveforms to have 1 or 2 dimensions, got shape {waveforms.shape}")

        if waveforms.shape[0] > 1:
            waveforms = waveforms[0][np.newaxis, :]

        audio_tensor = waveforms[np.newaxis, :]
        audio_length = np.array([audio_tensor.shape[-1]], dtype=np.int64)

        features, lengths = self.forward(audio_tensor, audio_length)
        features = features.astype(np.float32)
        lengths = lengths.astype(np.int64)

        encoder_output = self.encode(features, lengths)
        print("Encoder output shape:", encoder_output.shape)

        if decode_flag == "GD":
            transcription, metrics = self.decode_rnnt_greedy(
                encoder_output=encoder_output,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                ground_truth=ground_truth
            )
        elif decode_flag == "BS":
            transcription, metrics = self.decode_rnnt_beam_search(
                encoder_output=encoder_output,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                beam_width=beam_width,
                length_penalty=length_penalty,
                ground_truth=ground_truth
            )
        else:
            raise ValueError(f"Invalid decode_flag: {decode_flag}. Must be 'GD' or 'BS'.")

        return transcription, metrics