from typing import List, Tuple, Dict, Any
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
from torch import Tensor
import re
import torchaudio
import matplotlib.pyplot as plt

from MetricsClass import return_metrics

# Preprocessing parameters (aligned with test_nemo_preprocessor.py)
sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
preemph = 0.97
log_zero_guard_value = float(2**-24)

melscale_fbanks80 = torchaudio.functional.melscale_fbanks(
    n_fft // 2 + 1, 0, sample_rate // 2, 80, sample_rate, "slaney", "slaney"
)

def normalize(x: Tensor, lens: Tensor) -> Tensor:
    batch_size, M, T = x.shape
    lens_3d = lens.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
    mask = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(1) < lens_3d  # [batch_size, 1, T]
    lens_3d = lens_3d.float()
    mean = (torch.where(mask, x, 0).sum(dim=-1, keepdim=True) / lens_3d).detach()
    var = (torch.where(mask, x - mean, 0).pow(2).sum(dim=-1, keepdim=True) / (lens_3d - 1)).detach()
    return torch.where(mask, (x - mean) / (torch.sqrt(var + 1e-5)), 0)

def nemo_preprocessor(waveforms: Tensor, waveforms_lens: Tensor) -> Tuple[Tensor, Tensor]:
    if preemph != 0.0:
        waveforms = torch.cat([waveforms[:, :1], waveforms[:, 1:] - preemph * waveforms[:, :-1]], dim=-1)

    waveforms = torch.nn.functional.pad(waveforms, (n_fft // 2, n_fft // 2), mode="reflect")
    hann_window = torch.nn.functional.pad(
        torch.hann_window(win_length, periodic=False, device=waveforms.device),
        (n_fft // 2 - win_length // 2, n_fft // 2 - win_length // 2)
    )
    stft = torch.stft(waveforms, n_fft=n_fft, hop_length=hop_length, window=hann_window, return_complex=True)
    spectrogram = stft.abs().pow(2)

    mel_spectrogram = torch.matmul(spectrogram.transpose(1, 2), melscale_fbanks80.to(spectrogram.device))
    log_mel_spectrogram = torch.log(mel_spectrogram + log_zero_guard_value)

    features = log_mel_spectrogram.transpose(1, 2)  # [batch_size, M, T] -> [batch_size, T, M]
    features_lens = (waveforms_lens / hop_length + 1).long()
    return normalize(features, features_lens), features_lens

# Vocab loading
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
    while len(vocab) < 1025:
        vocab.append("<pad>")
    return vocab

VOCAB = load_vocab("onnx_models/vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt")
BLANK_IDX = VOCAB.index("<unk>")
BLK_IDX = VOCAB.index("<blk>") if "<blk>" in VOCAB else 1024
PAD_IDX = VOCAB.index("<pad>") if "<pad>" in VOCAB else 1024
MAX_VOCAB_IDX = len(VOCAB) - 1

print(f"Vocab size: {len(VOCAB)}")
print(f"Sample vocab tokens: {VOCAB[:10]}")
print(f"BLANK_IDX: {BLANK_IDX}, BLK_IDX: {BLK_IDX}, PAD_IDX: {PAD_IDX}, MAX_VOCAB_IDX: {MAX_VOCAB_IDX}")

class RnntASRPyTorch(nn.Module):
    @torch.inference_mode()
    def __init__(self,
                 encoder_path: str,
                 decoder_joint_path: str,
                 sample_rate: int = 16000,
                 features: int = 80) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.features = features
        self.hidden_size = 640
        self.vocab = VOCAB
        self.blank_idx = BLANK_IDX
        self.blk_idx = BLK_IDX
        self.pad_idx = PAD_IDX
        self.max_vocab_idx = MAX_VOCAB_IDX

        self._encoder = rt.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(decoder_joint_path, providers=["CPUExecutionProvider"])

        self._print_tensor_info("Encoder Inputs", self._encoder.get_inputs())
        self._print_tensor_info("Encoder Outputs", self._encoder.get_outputs())
        self._print_tensor_info("Decoder/Joint Inputs", self._decoder_joint.get_inputs())
        self._print_tensor_info("Decoder/Joint Outputs", self._decoder_joint.get_outputs())

        self._encoder_input_name = self._encoder.get_inputs()[0].name
        self._encoder_length_name = self._encoder.get_inputs()[1].name
        self._decoder_input_name = self._decoder_joint.get_inputs()[0].name
        self._decoder_prev_token_name = self._decoder_joint.get_inputs()[1].name
        self._decoder_state_name = self._decoder_joint.get_inputs()[2].name
        self._decoder_output_name = self._decoder_joint.get_outputs()[0].name
        self._decoder_state_out_name = self._decoder_joint.get_outputs()[1].name

    def _print_tensor_info(self, title: str, tensors: List[rt.NodeArg]) -> None:
        print(f"{title}:")
        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    @torch.inference_mode()
    def out_len(self, input_lengths: Tensor) -> Tensor:
        return input_lengths.div(160, rounding_mode="floor").add(1).long()

    @torch.inference_mode()
    def extract_features(self, input_signal: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        print(f"input_signal shape: {input_signal.shape}, min: {input_signal.min().item()}, max: {input_signal.max().item()}")
        if torch.isnan(input_signal).any() or torch.isinf(input_signal).any():
            print("Warning: input_signal contains NaN or Inf values!")

        features, features_len = nemo_preprocessor(input_signal, length)

        plt.figure(figsize=(10, 4))
        plt.imshow(features[0].cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
        plt.colorbar(label="Normalized Log Mel Energy")
        plt.title("Mel-Spectrogram (After Normalization)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()
        plt.show()

        return features, features_len

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
            dummy_logits = np.zeros((1, 1, 1, len(self.vocab)), dtype=np.float32)
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
        print(f"Raw logits min: {logits.min()}, max: {logits.max()}")
        logits = logits - np.max(logits)  # Stabilize logits
        return logits, (outputs[1], outputs[2])

    def decode_rnnt_greedy(self,
                           encoder_output: np.ndarray,
                           vocab: List[str],
                           blank_idx: int,
                           max_vocab_idx: int,
                           ground_truth: str = None,
                           max_steps: int = 1000,
                           min_tokens: int = 10,
                           max_tokens_per_step: int = 10,
                           state_init: str = "zero") -> Tuple[str, Dict[str, float], List[int]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        if state_init == "random":
            state1 = np.random.normal(0, 0.05, (1, 1, self.hidden_size)).astype(np.float32)  # Меньше шума
            state2 = np.random.normal(0, 0.05, (1, 1, self.hidden_size)).astype(np.float32)
        else:
            state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
            state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)

        start_token = self.vocab.index("▁") if "▁" in self.vocab else 0
        prev_token = np.array([[start_token]], dtype=np.int32)

        decoded_ids = []
        timestamps = []
        step = 0
        consecutive_blanks = 0
        consecutive_repeats = 0
        last_emitted_token = None
        emitted_count = 0

        # Словарь для отслеживания частоты токенов
        token_frequency = {}

        # Настройки для более мягкого декодирования
        base_temperature = 1.2  # Увеличили базовую температуру
        min_temperature = 0.8  # Увеличили минимальную температуру
        base_blank_penalty = -1.5  # Более мягкий штраф за blank

        print(f"Starting greedy decoding with {time_frames} time frames")

        while step < time_frames and step < max_steps:
            logits, (state1, state2) = self.decode_step(encoder_output, prev_token, (state1, state2), step)

            if logits.size == 0:
                print(f"Empty logits at step {step}, breaking")
                break

            logits = logits[0, 0, 0]  # [vocab_size]

            # Динамическая температура - плавное снижение
            progress = step / min(time_frames, max_steps)
            temperature = max(min_temperature, base_temperature - progress * 0.4)

            # Мягкий штраф за blank токен
            blank_penalty = base_blank_penalty
            if consecutive_blanks > 3:
                blank_penalty -= 1.0  # Увеличиваем штраф при много blank подряд
            logits[blank_idx] += blank_penalty

            # Штрафы за специальные токены
            if self.blk_idx < len(logits):
                logits[self.blk_idx] -= 3.0
            if self.pad_idx < len(logits):
                logits[self.pad_idx] -= 3.0

            # Применяем температуру
            scaled_logits = logits / temperature

            # Стабилизация для предотвращения переполнения
            scaled_logits = scaled_logits - np.max(scaled_logits)

            # Вычисляем вероятности
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / (np.sum(exp_logits) + 1e-12)

            # Штрафы за повторения - более мягкие
            if last_emitted_token is not None and last_emitted_token < len(probs):
                repeat_penalty = -0.5 if consecutive_repeats == 0 else -1.0 - consecutive_repeats * 0.3
                probs[last_emitted_token] = probs[last_emitted_token] * np.exp(repeat_penalty)

            # Глобальные штрафы за частые токены - более мягкие
            for token_id, freq in token_frequency.items():
                if token_id < len(probs) and freq > 2:
                    frequency_penalty = -0.2 * (freq - 2)  # Мягче штрафуем
                    probs[token_id] = probs[token_id] * np.exp(frequency_penalty)

            # Нормализуем вероятности после штрафов
            probs = np.clip(probs, 1e-12, 1.0)
            probs = probs / np.sum(probs)

            # Выбираем токен - используем top-k sampling для разнообразия
            k = min(5, len(probs))  # top-5 для лучшего разнообразия
            top_k_indices = np.argpartition(probs, -k)[-k:]
            top_k_probs = probs[top_k_indices]

            # Нормализуем top-k вероятности
            if np.sum(top_k_probs) > 0:
                top_k_probs = top_k_probs / np.sum(top_k_probs)
                # С вероятностью 0.8 берем наиболее вероятный, с 0.2 - семплируем
                if np.random.random() < 0.8:
                    selected_idx = np.argmax(top_k_probs)
                else:
                    selected_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
                token_idx = top_k_indices[selected_idx]
            else:
                token_idx = np.argmax(probs)

            if step % 20 == 0:  # Логируем реже
                print(
                    f"Step {step}: token_idx={token_idx}, token='{self.vocab[token_idx] if token_idx < len(self.vocab) else 'UNK'}', prob={probs[token_idx]:.4f}, temp={temperature:.2f}")

            # Обработка выбранного токена
            if token_idx == blank_idx or token_idx == self.blk_idx or token_idx == self.pad_idx or token_idx > max_vocab_idx:
                # Blank токен
                consecutive_blanks += 1
                consecutive_repeats = 0
                prev_token = np.array([[blank_idx]], dtype=np.int32)

                # Если слишком много blank подряд, пытаемся форсировать не-blank
                if consecutive_blanks > 8:
                    print(f"Too many consecutive blanks ({consecutive_blanks}), trying to force non-blank")
                    # Исключаем blank и выбираем следующий лучший
                    probs_no_blank = probs.copy()
                    probs_no_blank[blank_idx] = 0
                    probs_no_blank[self.blk_idx] = 0
                    probs_no_blank[self.pad_idx] = 0
                    if np.sum(probs_no_blank) > 0:
                        probs_no_blank = probs_no_blank / np.sum(probs_no_blank)
                        token_idx = np.argmax(probs_no_blank)
                        if token_idx <= max_vocab_idx:
                            # Принудительно emit этот токен
                            decoded_ids.append(token_idx)
                            timestamps.append(step)
                            token_frequency[token_idx] = token_frequency.get(token_idx, 0) + 1
                            prev_token = np.array([[token_idx]], dtype=np.int32)
                            last_emitted_token = token_idx
                            consecutive_blanks = 0
                            consecutive_repeats = 0
                            emitted_count += 1
                            print(f"Forced emission: {self.vocab[token_idx]}")
            else:
                # Не-blank токен
                if token_idx == last_emitted_token:
                    consecutive_repeats += 1
                    # Позволяем некоторые повторения, но ограничиваем их
                    if consecutive_repeats >= 3:
                        print(f"Skipping repeated token {self.vocab[token_idx]} (repeat #{consecutive_repeats})")
                        prev_token = np.array([[blank_idx]], dtype=np.int32)
                    else:
                        decoded_ids.append(token_idx)
                        timestamps.append(step)
                        token_frequency[token_idx] = token_frequency.get(token_idx, 0) + 1
                        prev_token = np.array([[token_idx]], dtype=np.int32)
                        emitted_count += 1
                else:
                    # Новый токен
                    decoded_ids.append(token_idx)
                    timestamps.append(step)
                    token_frequency[token_idx] = token_frequency.get(token_idx, 0) + 1
                    prev_token = np.array([[token_idx]], dtype=np.int32)
                    last_emitted_token = token_idx
                    consecutive_repeats = 0
                    emitted_count += 1

                consecutive_blanks = 0

            step += 1

            # Проверка условий остановки - более мягкие
            if emitted_count >= min_tokens and step > time_frames * 0.8:
                print(f"Stopping: sufficient tokens ({emitted_count}) and processed most frames")
                break

            # Аварийный выход если генерируем слишком много токенов
            if emitted_count > time_frames * 2:
                print(f"Stopping: too many tokens generated ({emitted_count})")
                break

        # Постобработка токенов
        tokens = [self.vocab[tok] for tok in decoded_ids if
                  tok < len(self.vocab) and tok != self.blk_idx and tok != self.pad_idx]

        # Убираем trailing мусор более аккуратно
        while tokens and tokens[-1].strip() in ['', 'ь', 'Ь', '▁']:
            tokens.pop()

        # Собираем текст
        text = "".join(tokens)
        text = text.replace("▁", " ")

        # Постобработка текста
        text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
        text = re.sub(r'(\w)\1{2,}', r'\1', text)  # Повторяющиеся символы (3+ раза)
        text = re.sub(r'(\w+)\s+\1(\s|$)', r'\1\2', text)  # Повторяющиеся слова
        text = text.strip()

        print(f"Decoded transcription (Improved Greedy): {text}")
        print(f"Total tokens generated: {len(decoded_ids)}")
        print(f"Unique tokens: {len(set(decoded_ids))}")
        print(f"Token frequency distribution: {sorted(token_frequency.items(), key=lambda x: x[1], reverse=True)[:10]}")

        # Вычисляем метрики
        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(transcription=text,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=0.0,
                                     flag="improved_greedy")

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
                                max_tokens_per_step: int = 10,
                                state_init: str = "random") -> Tuple[str, Dict[str, float], List[int]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        if state_init == "random":
            state1 = np.random.normal(0, 0.1, (1, 1, self.hidden_size)).astype(np.float32)
            state2 = np.random.normal(0, 0.1, (1, 1, self.hidden_size)).astype(np.float32)
        else:
            state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
            state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)

        beams = [(tuple(), 0.0, (state1, state2), np.array([[0]]), 0, 0, [])]
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
                    logits, (new_state1, new_state2) = self.decode_step(encoder_output, prev_token, (state1, state2), curr_t)
                    cache[cache_key] = (logits, (new_state1, new_state2))

                logits_t = logits[0, 0, 0]
                logits_t[blank_idx] -= 3.0  # Apply blank penalty
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
                    new_beams[tuple(new_seq)] = (tuple(new_seq), new_score, (new_state1, new_state2), np.array([[token]]), curr_t + 1, new_count, new_timestamps)

            beams = sorted(new_beams.values(), key=lambda x: x[1] / (max(1, x[5]) ** length_penalty), reverse=True)[:beam_width]
            t += 1
            if t % 50 == 0:
                cache.clear()
            if all(curr_t >= time_frames and count >= min_tokens for _, _, _, _, curr_t, count, _ in beams):
                break

        best_seq, best_score, _, _, _, token_count, timestamps = beams[0]
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
                  min_tokens: int = 10,
                  state_init: str = "random") -> Tuple[str, List[int]]:
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
        print(f"audio_tensor shape: {audio_tensor.shape}, audio_length: {audio_length}, min: {audio_tensor.min().item()}, max: {audio_tensor.max().item()}")
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
            transcription, _, timestamps = self.decode_rnnt_greedy(
                encoder_output=encoder_output,
                vocab=self.vocab,
                blank_idx=self.blank_idx,
                max_vocab_idx=self.max_vocab_idx,
                ground_truth=ground_truth,
                max_steps=max_steps,
                min_tokens=min_tokens,
                state_init=state_init
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
                min_tokens=min_tokens,
                state_init=state_init
            )
        else:
            raise ValueError("decode_flag must be 'GD' или 'BS'")

        return transcription, timestamps