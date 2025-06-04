from typing import List, Tuple, Dict, Any
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
from torch import Tensor
import re
import torchaudio
import matplotlib.pyplot as plt
import nemo.collections.asr.modules.audio_preprocessing as nemo_preproc

from MetricsClass import return_metrics

# Загружаем словарь из файла
def load_vocab(vocab_path: str) -> List[str]:
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
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
    # Pad to 1025 tokens as specified
    while len(vocab) < 1025:
        vocab.append("<pad>")
    return vocab

VOCAB = load_vocab("onnx_models/vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt")
BLANK_IDX = VOCAB.index("<unk>")
BLK_IDX = VOCAB.index("<blk>") if "<blk>" in VOCAB else 1395  # Adjust to 1024 since vocab size is 1025
PAD_IDX = VOCAB.index("<pad>") if "<pad>" in VOCAB else 1024
MAX_VOCAB_IDX = len(VOCAB) - 1

# Проверка словаря
print(f"Vocab size: {len(VOCAB)}")
print(f"Sample vocab tokens: {VOCAB[:10]}")
print(f"BLANK_IDX: {BLANK_IDX}, BLK_IDX: {BLK_IDX}, PAD_IDX: {PAD_IDX}, MAX_VOCAB_IDX: {MAX_VOCAB_IDX}")

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
        super().__init__()
        self.sample_rate = sample_rate
        self.features = features
        self.hidden_size = 640  # Consistent with original
        self.vocab = VOCAB
        self.blank_idx = BLANK_IDX
        self.blk_idx = BLK_IDX
        self.pad_idx = PAD_IDX
        self.max_vocab_idx = MAX_VOCAB_IDX

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

        # Use NeMo's AudioToMelSpectrogramPreprocessor for feature extraction
        self.audio_to_mel = nemo_preproc.AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            window="hann",
            # normalize="per_feature",
            n_fft=512,
            normalize=False,
            # n_fft=None,
            preemph=0.97,
            log_zero_guard_value=5.960464477539063e-08,
            dither=1e-05,
            features=features,
        )

    def _print_tensor_info(self, title: str, tensors: List[rt.NodeArg]) -> None:
        print(f"{title}:")
        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    def normalize_features(self, features: Tensor) -> Tensor:
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True)
        if std.eq(0).any():
            print("Warning: zero standard deviation in normalize_features, adding epsilon")
            std = std + 1e-6
        return (features - mean) / std

    @torch.inference_mode()
    def out_len(self, input_lengths: Tensor) -> Tensor:
        return input_lengths.div(160, rounding_mode="floor").add(1).long()

    @torch.inference_mode()
    def extract_features(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        print(f"input_signal shape: {input_signal.shape}, min: {input_signal.min().item()}, max: {input_signal.max().item()}")
        if torch.isnan(input_signal).any() or torch.isinf(input_signal).any():
            print("Warning: input_signal contains NaN or Inf values!")

        mel_spec_output = self.audio_to_mel(input_signal=input_signal, length=length)
        print(f"audio_to_mel output type: {type(mel_spec_output)}")
        mel_spec, _ = mel_spec_output
        print(f"mel_spec shape: {mel_spec.shape}, dtype: {mel_spec.dtype}, min: {mel_spec.min().item()}, max: {mel_spec.max().item()}")
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            print("Warning: mel_spec contains NaN or Inf values!")

        mel_spec = torch.clamp(mel_spec, min=1e-8)
        log_mel = torch.log(mel_spec)
        print(f"log_mel shape: {log_mel.shape}, min: {log_mel.min().item()}, max: {log_mel.max().item()}")
        if torch.isnan(log_mel).any() or torch.isinf(log_mel).any():
            print("Warning: log_mel contains NaN or Inf values!")

        plt.figure(figsize=(10, 4))
        plt.imshow(log_mel[0].cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
        plt.colorbar(label="Log Mel Energy")
        plt.title("Log Mel-Spectrogram (Before Normalization)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()
        plt.show()

        norm_mel = self.normalize_features(log_mel)
        print(f"norm_mel shape: {norm_mel.shape}, min: {norm_mel.min().item()}, max: {norm_mel.max().item()}")
        if torch.isnan(norm_mel).any() or torch.isinf(norm_mel).any():
            print("Warning: norm_mel contains NaN or Inf values!")

        return norm_mel, self.out_len(length)

    @torch.inference_mode()
    def encode(self,
               features: np.ndarray,
               lengths: np.ndarray) -> np.ndarray:
        inputs = {
            self._encoder_input_name: features,
            self._encoder_length_name: lengths.astype(np.int64)
        }
        encoder_output = self._encoder.run([self._encoder.get_outputs()[0].name], inputs)[0]
        print(f"Encoder output shape: {encoder_output.shape}")
        print(f"Encoder output min/max: {encoder_output.min()}, {encoder_output.max()}")
        if np.isnan(encoder_output).any() or np.isinf(encoder_output).any():
            print("Warning: encoder_output contains NaN or Inf values!")
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

        if encoder_output_t.shape[2] == 0:
            print(f"Error: encoder_output_t is empty at t={t}, returning dummy logits")
            dummy_logits = np.zeros((1, 1, 1, len(self.vocab)), dtype=np.float32)  # Adjusted to 1025
            return dummy_logits, (state1, state2)

        # encoder_output_t = (encoder_output_t - encoder_output_t.mean()) / (encoder_output_t.std() + 1e-9)

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
        print(f"Raw logits min: {logits.min()}, max: {logits.max()}")
        # logits = logits - np.max(logits)  # Стабилизация логитов
        return logits, (outputs[1], outputs[2])

    def decode_rnnt_greedy(self,
                           encoder_output: np.ndarray,
                           vocab: List[str],
                           blank_idx: int,
                           max_vocab_idx: int,
                           ground_truth: str = None,
                           max_steps: int = 1000,
                           min_tokens: int = 10,
                           max_tokens_per_step: int = 10) -> Tuple[str, Dict[str, float], List[int]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        # state1 = np.random.normal(0, 0.1, (1, 1, self.hidden_size)).astype(np.float32)
        # state2 = np.random.normal(0, 0.1, (1, 1, self.hidden_size)).astype(np.float32)
        start_token = self.vocab.index("▁") if "▁" in self.vocab else 1
        prev_token = np.array([[start_token]], dtype=np.int32)

        decoded_ids = []
        timestamps = []  # Track time steps for each emitted token
        step = 0
        repeat_count = 0
        last_token = None
        emitted_tokens = 0  # Track consecutive emissions

        while step < time_frames and step < max_steps:
            logits, (state1, state2) = self.decode_step(
                encoder_output,
                prev_token,
                (state1, state2),
                step
            )

            temperature = 0.5
            probs = np.exp(logits[0, 0, 0]/temperature - np.max(logits[0, 0, 0]/temperature))  # Remove temperature (set to 1.0)
            probs /= np.sum(probs)
            print(f"Step {step}, Top 5 probs: {np.sort(probs)[-5:]}")
            print(f"Top 5 indices: {np.argsort(probs)[-5:]}")

            probs[blank_idx] = 0.0
            probs[self.blk_idx] = 0.0
            probs[self.pad_idx] = 0.0
            token_idx = np.argmax(probs)
            print(
                f"Selected token_idx: {token_idx}, token: {self.vocab[token_idx] if token_idx < len(self.vocab) else 'INVALID'}")

            if token_idx != self.blank_idx and token_idx != self.blk_idx and token_idx != self.pad_idx and token_idx <= self.max_vocab_idx:
                if token_idx == last_token:
                    repeat_count += 1
                    if repeat_count > 5:
                        print("Breaking due to repeated token")
                        break
                else:
                    repeat_count = 0
                    last_token = token_idx
                    decoded_ids.append(token_idx)
                    timestamps.append(step)  # Record the time step
                    emitted_tokens += 1
                    prev_token = np.array([[token_idx]], dtype=np.int32)
            else:
                prev_token = np.array([[self.blank_idx]], dtype=np.int32)
                emitted_tokens = 0  # Reset on blank

            # Advance time step
            if emitted_tokens >= max_tokens_per_step:
                emitted_tokens = 0
            step += 1

        # Post-process transcription with space cleanup
        tokens = [self.vocab[tok] for tok in decoded_ids if
                  tok < len(self.vocab) and tok != self.blk_idx and tok != self.pad_idx]
        text = "".join(tokens)
        # Replace ▁ with space and clean up spacing (similar to example code)
        text = text.replace("▁", " ")
        text = re.sub(r"\A\s|\s\B|(\s)\b", lambda x: " " if x.group(1) else "", text)

        print(f"Decoded transcription (Greedy): {text}")
        print(f"Total tokens generated: {len(decoded_ids)}")
        print(f"Decoded IDs: {decoded_ids}")
        print(f"Decoded tokens: {tokens}")
        print(f"Timestamps: {timestamps}")

        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(transcription=text,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=0.0,
                                     flag="greedy")

        return text, metrics, timestamps

    def decode_rnnt_beam_search(self,
                                encoder_output: np.ndarray,
                                vocab: List[str],
                                blank_idx: int,
                                max_vocab_idx: int,
                                beam_width: int = 3,
                                length_penalty: float = 1.0,
                                ground_truth: str = None,
                                max_steps: int = 1000,
                                min_tokens: int = 10,
                                max_tokens_per_step: int = 10) -> Tuple[str, Dict[str, float], List[int]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        # state1 = np.random.normal(0, 0.1, (1, 1, self.hidden_size)).astype(
        #     np.float32)  # Changed to noisy initialization
        # state2 = np.random.normal(0, 0.1, (1, 1, self.hidden_size)).astype(np.float32)
        beams = [(tuple(), 0.0, (state1, state2), np.array([[0]]), 0, 0, [])]  # Added timestamps list
        cache = {}

        t = 0
        emitted_tokens = 0
        while t < time_frames and t < max_steps:
            new_beams = {}
            for seq, score, (state1, state2), prev_token, curr_t, token_count, timestamps in beams:
                if curr_t >= time_frames and token_count >= min_tokens:
                    new_beams[seq] = (seq, score, (state1, state2), prev_token, curr_t, token_count, timestamps)
                    continue

                cache_key = (tuple(seq), tuple(prev_token.flatten()), curr_t)
                if cache_key in cache:
                    logits, (new_state1, new_state2) = cache[cache_key]
                else:
                    logits, (new_state1, new_state2) = self.decode_step(encoder_output, prev_token, (state1, state2),
                                                                        curr_t)
                    cache[cache_key] = (logits, (new_state1, new_state2))

                logits_t = logits[0, 0, 0]  # (V+1,)
                top_k_indices = np.argsort(logits_t)[-100:]
                top_k_indices = np.concatenate([top_k_indices, np.array([blank_idx])])
                logits_t_sampled = logits_t[top_k_indices]
                top_indices = np.argpartition(logits_t_sampled, -beam_width)[-beam_width:]
                top_probs = logits_t_sampled[top_indices]
                top_indices = top_k_indices[top_indices]

                for token, prob in zip(top_indices, top_probs):
                    new_score = score + prob
                    if token == blank_idx:
                        new_score -= 3.0
                    if len(seq) > 1 and seq[-1] == token and token != blank_idx:
                        new_score -= 1.0
                    if len(seq) > 2 and seq[-1] == token and seq[-2] == token and token != blank_idx:
                        continue
                    new_seq = list(seq) + ([token] if token != blank_idx else [])
                    new_timestamps = timestamps + ([curr_t] if token != blank_idx else [])
                    new_count = token_count + (1 if token != blank_idx else 0)
                    new_beams[tuple(new_seq)] = (
                        tuple(new_seq), new_score, (new_state1, new_state2), np.array([[token]]), curr_t + 1, new_count,
                        new_timestamps
                    )

            beams = sorted(new_beams.values(), key=lambda x: x[1] / (max(1, x[5]) ** length_penalty), reverse=True)[
                    :beam_width]
            t += 1
            if t % 50 == 0:
                cache.clear()
            if all(curr_t >= time_frames and count >= min_tokens for _, _, _, _, curr_t, count, _ in beams):
                break

        best_seq, best_score, _, _, _, token_count, timestamps = beams[0]
        # Post-process transcription with space cleanup
        tokens = [vocab[tok] for tok in best_seq if tok != blank_idx and tok != self.blk_idx and tok != self.pad_idx]
        text = "".join(tokens)
        text = text.replace("▁", " ")
        text = re.sub(r"\A\s|\s\B|(\s)\b", lambda x: " " if x.group(1) else "", text)

        print(f"Transcription (Beam Search, beam_width={beam_width}, length_penalty={length_penalty}): {text}")
        print(f"Log probability (Beam Search): {best_score:.15f}")
        print(f"Total tokens in best sequence: {token_count}")
        print(f"Timestamps: {timestamps}")

        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(transcription=text,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=best_score,
                                     beam_width=beam_width,
                                     length_penalty=length_penalty,
                                     flag="beam")
        return text, metrics, timestamps

    def recognize(self,
                  waveforms: np.ndarray,
                  decode_flag: str = "GD",
                  beam_width: int = 10,
                  length_penalty: float = 0.7,
                  ground_truth: str = None,
                  max_steps: int = 1000,
                  min_tokens: int = 10) -> Tuple[str, List[int]]:  # Updated return type
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")

        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        if waveforms.ndim == 2 and waveforms.shape[0] > 1:
            waveforms = np.mean(waveforms, axis=0)
        elif waveforms.ndim == 2:
            waveforms = waveforms[0]
        waveforms = waveforms.flatten()

        audio_length_samples = len(waveforms)
        expected_time_frames = audio_length_samples // 160 + 1
        print(f"Input audio length: {audio_length_samples} samples, expected time frames: {expected_time_frames}")

        audio_tensor = torch.from_numpy(waveforms).float().unsqueeze(0)
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
        print(
            f"audio_tensor shape: {audio_tensor.shape}, audio_length: {audio_length}, min: {audio_tensor.min().item()}, max: {audio_tensor.max().item()}")
        if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
            print("Warning: audio_tensor contains NaN or Inf values!")

        mean = audio_tensor.mean()
        std = audio_tensor.std()
        if std == 0:
            print("Warning: audio_tensor has zero standard deviation, adding epsilon")
            std = 1e-6
        audio_tensor = (audio_tensor - mean) / (std + 1e-6)

        features, lengths = self.extract_features(audio_tensor, audio_length)
        print(f"features shape: {features.shape}, lengths: {lengths}")
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Warning: features contain NaN or Inf values!")

        plt.figure(figsize=(10, 4))
        plt.imshow(features[0].cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
        plt.colorbar(label="Normalized Log Mel Energy")
        plt.title("Mel-Spectrogram (After Normalization)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()
        plt.show()

        features = features.detach().cpu().numpy().astype(np.float32)
        lengths = lengths.detach().cpu().numpy().astype(np.int64)

        encoder_output = self.encode(features, lengths)
        time_frames = encoder_output.shape[2]
        print(f"Encoder output shape: {encoder_output.shape}, time_frames: {time_frames}")

        max_steps = min(max_steps, time_frames * 10)

        if decode_flag == "GD":
            transcription, _, timestamps = self.decode_rnnt_greedy(  # Updated to capture timestamps
                encoder_output=encoder_output,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                ground_truth=ground_truth,
                max_steps=max_steps,
                min_tokens=min_tokens
            )
        elif decode_flag == "BS":
            transcription, _, timestamps = self.decode_rnnt_beam_search(
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

        return transcription, timestamps