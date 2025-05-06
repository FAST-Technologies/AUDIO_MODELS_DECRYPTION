from typing import List, Tuple, Dict
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio

from MetricsClass import return_metrics

VOCAB = [
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
    "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"
]

class GigaamCtcASRPyTorch(nn.Module):
    """
        Module for extracting Log-mel spectrogram features from raw audio signals.

        This module uses Torchaudio's MelSpectrogram transform to extract features
        and applies logarithmic scaling. It also visualizes Mel filter banks using
        both Torchaudio and Librosa implementations for comparison.
    """

    @torch.inference_mode()
    def __init__(self,
                 model_path: str
     ) -> None:
        """
        Initialize the GigaamCtcASR model with an ONNX model.

        Parameters
        ----------
        model_path : str
            Path to the ONNX model file.
        """
        super().__init__()
        sample_rate = 16000
        features = 64
        self.sample_rate = sample_rate
        self.features = features
        self.hop_length = sample_rate // 100
        self.n_fft = sample_rate // 40
        self.win_length = sample_rate // 40
        self._model = rt.InferenceSession(model_path,
                                          providers=["CPUExecutionProvider"])
        try:
            self.tensor_info(flag="i",
                        tensors=self._model.get_inputs())
            self.tensor_info(flag="o",
                        tensors=self._model.get_outputs())
        except Exception as e:
            print(f"Ошибка в представлении тензорной информации модели: {e}")

        self.register_buffer(
            "mel_fb",
            torchaudio.functional.melscale_fbanks(
                n_freqs=int(self.n_fft // 2 + 1),
                f_min=0.0,
                f_max=self.sample_rate / 2.0,
                n_mels=self.features,
                sample_rate=self.sample_rate,
                norm="slaney"
            ))
        self.vocab = VOCAB
        self.blank_idx = 33
        self.max_vocab_idx = len(self.vocab) - 1

    @torch.inference_mode()
    def out_len(self,
                input_lengths: Tensor
    ) -> Tensor:
        """
        Calculate the output length after the feature extraction process.

        Parameters
        ----------
        input_lengths : Tensor
            Input lengths of the audio signals, expected shape [batch].

        Returns
        -------
        Tensor
            Output lengths after feature extraction, shape [batch].

        Notes
        -----
        The output length is calculated by dividing the input length by the hop length
        and adding 1 to account for the final frame.
        """
        return input_lengths.div(self.hop_length,
                                 rounding_mode="floor").add(1).long()

    @torch.inference_mode()
    def forward(self,
                input_signal: Tensor,
                length: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.

        Parameters
        ----------
        input_signal : Tensor
            Raw audio signal tensor, expected shape [batch, time].
        length : Tensor
            Lengths of the input audio signals, expected shape [batch].

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple containing:
            - Log-mel spectrogram features, shape [batch, features, time_frames].
            - Output lengths after feature extraction, shape [batch].

        Notes
        -----
        - The spectrogram is computed using Torchaudio's `spectrogram` function with a Hann window.
        - The Mel filter bank is applied to the spectrogram, followed by logarithmic scaling.
        - Values are clamped to the range [MY_CONSTANTS.MIN_LOG_TOLERANCE, MY_CONSTANTS.MAX_LOG_TOLERANCE]
          before applying the logarithm to avoid numerical issues.
        """
        spectrogram = torchaudio.functional.spectrogram(
            waveform=input_signal,
            pad=0,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.win_length,
                                     device=input_signal.device),
            power=2.0,
            normalized=False,
            onesided=True,
            center=True,
            pad_mode="reflect"
        )
        mel_spec = torch.matmul(spectrogram.transpose(-2, -1),
                                self.mel_fb).transpose(-2, -1)
        mel_spec = torch.log(mel_spec.clamp_(1e-9,1e9))
        mel_spec = mel_spec.unsqueeze(0)
        return mel_spec, self.out_len(length)

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
        # prev_tok = BLANK_IDX
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
                  waveforms: np.ndarray[np.float32],
                  decode_flag: str = "GD",
                  beam_width: int = 10,
                  length_penalty: float = 0.7,
                  ground_truth: str = None
    ) -> str:
        """
        Recognize speech from the input waveform and return the transcription.

        Parameters
        ----------
        waveforms : npt.NDArray[np.float32]
            Input waveform, shape [samples, channels]. Expected to be in float32 format.
        decode_flag : str, optional
            Decoding method: "GD" for greedy decoding, "BS" for beam search. Defaults to "BS".
        beam_width : int, optional
            Number of beams for beam search decoding. Defaults to 10.
        length_penalty : float, optional
            Length penalty for beam search decoding. Defaults to 0.7.
        ground_truth : str, optional
            Ground truth transcription for computing metrics. Defaults to None.

        Returns
        -------
        str
            Transcribed text.
        """
        global transcription
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")

        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        if waveforms.ndim == 2 and waveforms.shape[1] > 1:
            waveforms = np.mean(waveforms, axis=1)
            # waveforms = waveforms[0][np.newaxis, :]  # [2, time] -> [1, time]
        waveforms = waveforms.flatten()
        audio_tensor = torch.from_numpy(waveforms).float()
        audio_length = torch.tensor([len(waveforms)], dtype=torch.long)
        features, lengths = self.forward(audio_tensor.unsqueeze(0),
                                                               audio_length)
        features = features.detach().cpu().numpy().astype(np.float32)
        lengths = lengths.detach().cpu().numpy().astype(np.int64)
        features_for_onnx = features.squeeze(1) # (1, 1, 64, T) -> (1, 64, T)
        inputs = {
            "features": features_for_onnx,
            "feature_lengths": lengths
        }

        log_probs = self._model.run(["log_probs"], inputs)[0]  # [1, time_frames, vocab_size]
        print("Тип значений log_probs_PyMONO:", type(log_probs))
        print("Размерность значений log_probs_PyMONO:",
              log_probs.shape if isinstance(log_probs, np.ndarray) else "Не является numpy массивом")
        if np.any(np.isnan(log_probs)):
            raise ValueError("log_probs contains NaN values")
        if np.any(np.isinf(log_probs)):
            raise ValueError("log probs contains Inf values")

        # Декодирование с помощью beam search
        if decode_flag == "GD":
            transcription, _ = self.decode_ctc_greedy(
                log_probs=log_probs,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                ground_truth=ground_truth
            )
        elif decode_flag == "BS":
            transcription, _ = self.decode_ctc_beam_search(
                log_probs=log_probs,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                beam_width=beam_width,
                length_penalty=length_penalty,
                ground_truth=ground_truth
            )

        return transcription

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