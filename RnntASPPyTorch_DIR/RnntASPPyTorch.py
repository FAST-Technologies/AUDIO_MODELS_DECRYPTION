from typing import List, Tuple, Dict, Any
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio

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

class RnntASRPyTorch(nn.Module):
    """
    Module for RNN-T based Automatic Speech Recognition (ASR) using PyTorch and ONNX models.
    Processes raw audio signals, extracts Log-mel spectrogram features, and performs transcription
    using an RNN-T model with greedy decoding or beam search.
    """

    @torch.inference_mode()
    def __init__(self,
                 encoder_path: str,
                 decoder_joint_path: str,
                 sample_rate: int = 16000,
                 features: int = 80) -> None:
        """
        Initialize the RNNT ASR model with ONNX encoder and decoder/joint models.

        Parameters
        ----------
        encoder_path : str
            Path to the ONNX encoder model file.
        decoder_joint_path : str
            Path to the ONNX decoder/joint model file.
        sample_rate : int, optional
            Sample rate of the audio. Defaults to 16000 Hz.
        features : int, optional
            Number of mel features. Defaults to 80.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.features = features
        self.hop_length = sample_rate // 100  # 160
        self.n_fft = sample_rate // 40  # 400
        self.win_length = sample_rate // 40  # 400

        # Загружаем ONNX модели
        self._encoder = rt.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(decoder_joint_path, providers=["CPUExecutionProvider"])

        # Проверяем входы и выходы моделей
        self._print_tensor_info("Encoder Inputs", self._encoder.get_inputs())
        self._print_tensor_info("Encoder Outputs", self._encoder.get_outputs())
        self._print_tensor_info("Decoder/Joint Inputs", self._decoder_joint.get_inputs())
        self._print_tensor_info("Decoder/Joint Outputs", self._decoder_joint.get_outputs())

        # Имена входов и выходов
        self._encoder_input_name = self._encoder.get_inputs()[0].name
        self._encoder_length_name = self._encoder.get_inputs()[1].name
        self._decoder_input_name = self._decoder_joint.get_inputs()[0].name
        self._decoder_prev_token_name = self._decoder_joint.get_inputs()[1].name
        self._decoder_state_name = self._decoder_joint.get_inputs()[2].name
        self._decoder_output_name = self._decoder_joint.get_outputs()[0].name
        self._decoder_state_out_name = self._decoder_joint.get_outputs()[1].name

        # Размер скрытого состояния (обновлено до 640 на основе входов декодера)
        self.hidden_size = 640

        # Mel filter bank
        self.register_buffer(
            "mel_fb",
            torchaudio.functional.melscale_fbanks(
                n_freqs=int(self.n_fft // 2 + 1),
                f_min=0.0,
                f_max=self.sample_rate / 2.0,
                n_mels=80,
                sample_rate=self.sample_rate,
                norm="slaney"
            ))

        self.vocab = VOCAB
        self.blank_idx = BLANK_IDX
        self.max_vocab_idx = MAX_VOCAB_IDX

    def _print_tensor_info(self, title: str, tensors: List[rt.NodeArg]) -> None:
        """Print tensor information (inputs/outputs)."""
        print(f"{title}:")
        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    @torch.inference_mode()
    def out_len(self, input_lengths: Tensor) -> Tensor:
        """Calculate output length after feature extraction."""
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    @torch.inference_mode()
    def extract_features(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        spectrogram = torchaudio.functional.spectrogram(
            waveform=input_signal,
            pad=0,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.win_length, device=input_signal.device),
            power=2.0,
            normalized=False,
            onesided=True,
            center=True,
            pad_mode="reflect"
        )
        print(f"Spectrogram shape: {spectrogram.shape}")
        mel_spec = torch.matmul(spectrogram.transpose(-2, -1), self.mel_fb).transpose(-2, -1)
        print(f"Mel spectrogram shape (before log): {mel_spec.shape}")
        mel_spec = torch.log(mel_spec.clamp_(1e-9, 1e9))
        print(f"Mel spectrogram shape (after log): {mel_spec.shape}")
        return mel_spec, self.out_len(length)

    @torch.inference_mode()
    def encode(self, features: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Run the encoder on extracted features to get hidden states."""
        inputs = {
            self._encoder_input_name: features,
            self._encoder_length_name: lengths.astype(np.int64)
        }
        encoder_output = self._encoder.run([self._encoder.get_outputs()[0].name], inputs)[0]
        return encoder_output

    @torch.inference_mode()
    def decode_step(self,
                    encoder_output: np.ndarray,
                    prev_token: np.ndarray,
                    state: Tuple[np.ndarray, np.ndarray],
                    t: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        state1, state2 = state
        target_length = np.array([1], dtype=np.int32)
        encoder_output_t = encoder_output[:, :, t:t + 1]
        print(f"Decode step t={t}: encoder_output_t shape = {encoder_output_t.shape}, prev_token = {prev_token}")

        # Check if encoder_output_t is empty
        if encoder_output_t.shape[2] == 0:
            print(f"Error: encoder_output_t is empty at t={t}, returning dummy logits")
            dummy_logits = np.zeros((1, 1, 1, 1025), dtype=np.float32)
            return dummy_logits, (state1, state2)

        inputs = {
            self._decoder_input_name: encoder_output_t,
            self._decoder_prev_token_name: prev_token.astype(np.int32),
            self._decoder_joint.get_inputs()[2].name: target_length,
            self._decoder_joint.get_inputs()[3].name: state1,
            self._decoder_joint.get_inputs()[4].name: state2
        }
        outputs = self._decoder_joint.run(
            [self._decoder_output_name,
             self._decoder_joint.get_outputs()[2].name,
             self._decoder_joint.get_outputs()[3].name],
            inputs
        )
        logits = outputs[0]
        print(f"Decode step t={t}: logits shape = {logits.shape}, logits[0, 0, 0] = {logits[0, 0, 0]}")
        return logits, (outputs[1], outputs[2])

    def decode_rnnt_greedy(self,
                           encoder_output: np.ndarray,
                           vocab: List[str],
                           blank_idx: int,
                           max_vocab_idx: int,
                           ground_truth: str = None,
                           max_steps: int = 1000,
                           min_tokens: int = 10) -> Tuple[str, Dict[str, float]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        prev_token = np.array([[0]], dtype=np.int32)
        decoded_ids = []
        total_log_prob = 0.0
        step = 0
        generated_tokens = 0

        while step < max_steps and generated_tokens < min_tokens:
            logits, (state1, state2) = self.decode_step(encoder_output, prev_token, (state1, state2),
                                                        step % time_frames)
            logits_t = logits[0, 0, 0]
            probs = np.exp(logits_t - np.max(logits_t)) / np.sum(np.exp(logits_t - np.max(logits_t)), axis=-1)

            # Игнорируем <blk> на ранних шагах
            probs[blank_idx] = 0
            token_idx = np.argmax(probs)
            top2_indices = np.argsort(probs)[-2:]
            top2_probs = probs[top2_indices]
            log_prob = np.log(probs[token_idx] + 1e-10)
            total_log_prob += log_prob

            print(
                f"Step {step}, Time frame {step % time_frames}: Token idx = {token_idx}, Log prob = {log_prob}, Top probs = {top2_probs}")
            if token_idx != blank_idx and token_idx <= max_vocab_idx:
                decoded_ids.append(token_idx)
                prev_token = np.array([[token_idx]], dtype=np.int32)
                generated_tokens += 1
            else:
                prev_token = np.array([[blank_idx]], dtype=np.int32)

            step += 1

        transcription = "".join(vocab[tok] for tok in decoded_ids)
        print(f"Decoded transcription (Greedy): {transcription}")
        print(f"Log probability (Greedy): {total_log_prob:.15f}")
        print(f"Total tokens generated: {generated_tokens}")
        metrics = {}
        if ground_truth and transcription:
            metrics = return_metrics(transcription=transcription,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=total_log_prob,
                                     flag="greedy")
        else:
            print("Skipping metrics computation: either ground_truth or transcription is empty.")
        return transcription, metrics

    def decode_rnnt_beam_search(self,
                                encoder_output: np.ndarray,
                                vocab: List[str],
                                blank_idx: int,
                                max_vocab_idx: int,
                                beam_width: int = 3,
                                length_penalty: float = 1.0,
                                ground_truth: str = None,
                                max_steps: int = 1000,
                                min_tokens: int = 10) -> Tuple[str, Dict[str, float]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        beams = [(tuple(), 0.0, (state1, state2), np.array([[0]]), 0, 0)]

        step = 0
        while step < max_steps:
            new_beams = {}
            for seq, score, (state1, state2), prev_token, t, token_count in beams:
                if t >= time_frames and token_count >= min_tokens:
                    new_beams[seq] = (seq, score, (state1, state2), prev_token, t, token_count)
                    continue

                logits, (new_state1, new_state2) = self.decode_step(encoder_output, prev_token, (state1, state2),
                                                                    t % time_frames)
                probs = np.exp(logits[0, 0, 0] - np.max(logits[0, 0, 0])) / np.sum(
                    np.exp(logits[0, 0, 0] - np.max(logits[0, 0, 0])))

                # Игнорируем <blk> на ранних шагах
                probs[blank_idx] = 0
                top_indices = np.argpartition(probs, -beam_width)[-beam_width:]
                top_probs = probs[top_indices]

                for token, prob in zip(top_indices, top_probs):
                    new_score = score + np.log(prob + 1e-10)
                    # Увеличиваем штраф за выбор <blk>
                    if token == blank_idx:
                        new_score -= 1.0  # Увеличили штраф до 1.0
                    # Штрафуем повторяющиеся токены
                    if len(seq) > 1 and seq[-1] == token and token != blank_idx:
                        new_score -= 0.3  # Штраф за повторение
                    new_seq = list(seq) + ([token] if token != blank_idx else [])
                    new_count = token_count + (1 if token != blank_idx else 0)
                    new_beams[tuple(new_seq)] = (
                    tuple(new_seq), new_score, (new_state1, new_state2), np.array([[token]]), t + 1, new_count)

            beams = sorted(new_beams.values(), key=lambda x: x[1] / (max(1, x[5]) ** length_penalty), reverse=True)[
                    :beam_width]
            step += 1
            if all(t >= time_frames and count >= min_tokens for _, _, _, _, t, count in beams):
                break

        best_seq, best_score, _, _, _, _ = beams[0]
        transcription = "".join(vocab[tok] for tok in best_seq if tok != blank_idx)
        print(f"Transcription (Beam Search, beam_width={beam_width}, length_penalty={length_penalty}): {transcription}")
        print(f"Log probability (Beam Search): {best_score:.15f}")
        print(f"Total tokens in best sequence: {len(best_seq)}")
        metrics = {}
        if ground_truth and transcription:
            metrics = return_metrics(transcription=transcription,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=best_score,
                                     beam_width=beam_width,
                                     length_penalty=length_penalty,
                                     flag="beam")
        else:
            print("Skipping metrics computation: either ground_truth or transcription is empty.")
        return transcription, metrics

    def recognize(self,
                  waveforms: np.ndarray,
                  decode_flag: str = "GD",
                  beam_width: int = 10,
                  length_penalty: float = 0.7,
                  ground_truth: str = None,
                  max_steps: int = 1000,
                  min_tokens: int = 10) -> str:
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")

        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        if waveforms.ndim == 2 and waveforms.shape[1] > 1:
            waveforms = np.mean(waveforms, axis=1)
        waveforms = waveforms.flatten()

        audio_length_samples = len(waveforms)
        expected_time_frames = audio_length_samples // self.hop_length + 1
        print(f"Input audio length: {audio_length_samples} samples, expected time frames: {expected_time_frames}")

        audio_tensor = torch.from_numpy(waveforms).float()
        audio_length = torch.tensor([len(waveforms)], dtype=torch.long)

        features, lengths = self.extract_features(audio_tensor.unsqueeze(0), audio_length)
        features = features.detach().cpu().numpy().astype(np.float32)
        lengths = lengths.detach().cpu().numpy().astype(np.int64)

        encoder_output = self.encode(features, lengths)
        time_frames = encoder_output.shape[2]
        print(f"Encoder output shape: {encoder_output.shape}, time_frames: {time_frames}")

        # Ограничиваем max_steps количеством time_frames
        max_steps = min(max_steps, time_frames * 10)  # Даем возможность пройти больше шагов

        if decode_flag == "GD":
            transcription, _ = self.decode_rnnt_greedy(
                encoder_output=encoder_output,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                ground_truth=ground_truth,
                max_steps=max_steps,
                min_tokens=min_tokens
            )
        elif decode_flag == "BS":
            transcription, _ = self.decode_rnnt_beam_search(
                encoder_output=encoder_output,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                beam_width=beam_width,
                length_penalty=length_penalty,
                ground_truth=ground_truth,
                max_steps=max_steps,
                min_tokens=min_tokens
            )
        else:
            raise ValueError("decode_flag must be 'GD' или 'BS'")

        return transcription