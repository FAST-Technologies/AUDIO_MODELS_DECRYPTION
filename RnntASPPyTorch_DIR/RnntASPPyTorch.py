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
        self.true_blank_idx = VOCAB.index("<blk>")
        self.unk_idx = VOCAB.index("<unk>")

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

    def decode_rnnt_beam_search(self,
                                encoder_output: np.ndarray,
                                vocab: List[str],
                                blank_idx: int,
                                max_vocab_idx: int,
                                beam_width: int = 5,
                                length_penalty: float = 2.0,  # Увеличено для контроля длины
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
        temperature = 0.5  # Для сглаживания вероятностей

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

                logits_t = logits[0, 0, 0]
                logits_t[blank_idx] -= 3.0  # Blank penalty
                # Нормализация с температурой
                logits_t = logits_t / temperature
                logits_t = logits_t - np.max(logits_t)
                exp_logits = np.exp(logits_t)
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)

                top_k_indices = np.argsort(probs)[-100:]
                top_k_indices = np.concatenate([top_k_indices, np.array([blank_idx])])
                top_probs = probs[top_k_indices]
                top_indices = np.argpartition(top_probs, -beam_width)[-beam_width:]
                top_probs = top_probs[top_indices]
                top_indices = top_k_indices[top_indices]

                for token, prob in zip(top_indices, top_probs):
                    new_score = score + np.log(prob + 1e-12)
                    if token == blank_idx:
                        new_score -= 3.0

                    # Усиленный контроль повторений
                    if len(seq) > 0 and seq[-1] == token and token != blank_idx:
                        new_score -= 4.0  # Увеличенный штраф
                    if len(seq) > 1 and seq[-2] == token and seq[-1] == token and token != blank_idx:
                        continue
                    if len(seq) > 2 and seq[-3] == token and seq[-2] == seq[-1] and token != blank_idx:
                        continue
                    if len(seq) > 3 and seq[-4] == token and seq[-3] == seq[-2] and token != blank_idx:
                        continue

                    # Общий механизм завершения слов
                    word_started = any(not vocab[t].startswith("▁") for t in seq[-2:] if t != blank_idx)
                    if word_started and not vocab[token].startswith("▁") and prob > 0.1:
                        new_score += 1.0  # Поощрение для продолжения слова

                    new_seq = list(seq) + ([token] if token != blank_idx else [])
                    new_timestamps = timestamps + ([curr_t] if token != blank_idx else [])
                    new_count = token_count + (1 if token != blank_idx else 0)

                    # Строгий контроль длины
                    if new_count > 25:  # Ограничение на максимальное количество токенов
                        new_score -= 10.0 * (new_count - 25)

                    new_beams[tuple(new_seq)] = (
                    tuple(new_seq), new_score, (new_state1, new_state2), np.array([[token]]), curr_t + 1, new_count,
                    new_timestamps)

            beams = sorted(new_beams.values(), key=lambda x: x[1] / (max(1, x[5]) ** length_penalty), reverse=True)[
                    :beam_width]
            t += 1
            if t % 50 == 0:
                cache.clear()
            if all(curr_t >= time_frames and count >= min_tokens for _, _, _, _, curr_t, count, _ in beams):
                break

        best_seq, best_score, _, _, _, token_count, timestamps = beams[0]
        tokens = [vocab[tok] for tok in best_seq if tok != blank_idx and tok != self.blk_idx and tok != self.pad_idx]
        text = "".join(tokens)
        text = text.replace("▁", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[.,]+$", "", text).strip()  # Удаление лишних знаков препинания
        text = re.sub(r"\s*,\s*", " ", text).strip()  # Удаление запятых
        # Удаление последовательных дублирований слов
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            word = words[i]
            if i == 0 or word != words[i - 1]:
                cleaned_words.append(word)
            i += 1
        text = " ".join(cleaned_words)

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

    def decode_rnnt_greedy_improved(self,
                                    encoder_output: np.ndarray,
                                    vocab: List[str],
                                    blank_idx: int,  # Это будет self.true_blank_idx (e.g., 1024)
                                    max_vocab_idx: int,
                                    ground_truth: str = None,
                                    max_steps: int = 1000,  # Этот max_steps может быть не главным ограничителем
                                    min_tokens: int = 10,  # Не используется активно для останова
                                    max_tokens_per_step: int = 10,  # Не используется
                                    state_init: str = "zero") -> Tuple[str, Dict[str, float], List[int]]:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        print(f"Starting fixed greedy decoding with {time_frames} time frames, true_blank_idx={blank_idx}")

        if state_init == "random":
            state1 = np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32)
            state2 = np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32)
        else:
            state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
            state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)

        # Начальный токен - ИСПОЛЬЗУЕМ ИСТИННЫЙ BLANK_IDX
        prev_token = np.array([[blank_idx]], dtype=np.int32)

        decoded_ids = []
        timestamps = []
        step = 0
        consecutive_blanks = 0
        emitted_count = 0

        last_emitted_token_idx = -1  # Для отслеживания повторений эмитированных не-blank токенов
        repeat_count = 0

        temperature = 1.0  # Попробуйте 1.0 для "чистого" жадного или немного выше/ниже
        blank_penalty_val = -1.5

        # Цикл будет идти максимум time_frames шагов (соответствует каждому кадру энкодера)
        # max_steps из параметров функции может быть дополнительным ограничителем, если он меньше time_frames
        actual_max_steps = min(time_frames, max_steps)
        last_emitted_non_blank_token_for_penalty = -1

        while step < actual_max_steps:
            try:
                logits_orig, (state1, state2) = self.decode_step(encoder_output, prev_token, (state1, state2), step)

                if logits_orig.size == 0:
                    print(f"Empty logits at step {step}, moving to next step")
                    step += 1
                    continue

                logits = logits_orig[0, 0, 0].copy()
                logits[blank_idx] += blank_penalty_val

                if self.pad_idx != blank_idx and self.pad_idx < len(logits):
                    logits[self.pad_idx] -= 5.0
                if 0 != blank_idx and 0 < len(logits):
                    logits[0] -= 2.0

                for i in range(max_vocab_idx + 1, len(logits)):
                    logits[i] -= 5.0

                if temperature != 1.0:
                    scaled_logits = logits / temperature
                else:
                    scaled_logits = logits

                scaled_logits = scaled_logits - np.max(scaled_logits)
                exp_logits = np.exp(scaled_logits)
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)

                # Контроль повторений
                last_two_tokens = [last_emitted_token_idx] if last_emitted_token_idx != -1 else []
                if last_emitted_token_idx != -1 and token_idx == last_emitted_token_idx:
                    repeat_count += 1
                else:
                    repeat_count = 0

                if repeat_count >= 1:
                    print(f"Applying repetition penalty for token {self.vocab[token_idx]}")
                    probs[token_idx] *= 0.05
                    probs = probs / np.sum(probs)
                    token_idx = np.argmax(probs)

                token_idx = np.argmax(probs)
                if probs[token_idx] < 0.6:
                    print(
                        f"Low confidence for token {self.vocab[token_idx]} (prob={probs[token_idx]:.4f}), emitting blank")
                    token_idx = blank_idx

                if step % 50 == 0:
                    token_str = self.vocab[token_idx] if token_idx < len(self.vocab) else 'OUT_OF_VOCAB'
                    print(
                        f"Step {step}/{actual_max_steps}: chosen='{token_str}' ({token_idx}), prob={probs[token_idx]:.4f}, "
                        f"blanks={consecutive_blanks}, emitted={emitted_count}, repeat_count={repeat_count}")

                is_blank_equivalent = (token_idx == blank_idx or
                                       (token_idx == self.pad_idx and self.pad_idx != blank_idx) or
                                       token_idx > max_vocab_idx)

                if is_blank_equivalent:
                    consecutive_blanks += 1
                    prev_token = np.array([[blank_idx]], dtype=np.int32)

                    if consecutive_blanks > 8:
                        print(f"Too many consecutive blanks ({consecutive_blanks}), forcing non-blank emission")
                        non_blank_probs = probs.copy()
                        non_blank_probs[blank_idx] = 0
                        for recent_token in last_two_tokens:
                            if recent_token < len(non_blank_probs):
                                non_blank_probs[recent_token] = 0
                        if np.sum(non_blank_probs) > 0:
                            non_blank_probs = non_blank_probs / np.sum(non_blank_probs)
                            token_idx = np.argmax(non_blank_probs)
                            decoded_ids.append(token_idx)
                            timestamps.append(step)
                            prev_token = np.array([[token_idx]], dtype=np.int32)
                            emitted_count += 1
                            consecutive_blanks = 0
                            last_emitted_token_idx = token_idx
                            last_emitted_non_blank_token_for_penalty = token_idx
                else:
                    if token_idx == last_emitted_token_idx:
                        repeat_count += 1
                    else:
                        repeat_count = 0
                    last_emitted_token_idx = token_idx

                    if repeat_count >= 1:
                        print(f"Applying repetition penalty for token {self.vocab[token_idx]}")
                        probs[token_idx] *= 0.05
                        probs = probs / np.sum(probs)
                        token_idx = np.argmax(probs)

                    decoded_ids.append(token_idx)
                    timestamps.append(step)
                    prev_token = np.array([[token_idx]], dtype=np.int32)
                    emitted_count += 1
                    consecutive_blanks = 0
                    last_emitted_non_blank_token_for_penalty = token_idx

                step += 1

            except Exception as e:
                print(f"Error at step {step}: {e}")
                step += 1
                continue

        print(f"Decoding completed: processed {step}/{actual_max_steps} steps, {emitted_count} tokens emitted")

        # Постобработка (используйте вашу _postprocess_tokens_improved)
        # text = self._postprocess_tokens(decoded_ids) # или _postprocess_tokens_improved
        text = self._postprocess_tokens_improved(decoded_ids)

        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(transcription=text,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=0.0,
                                     flag="improved_greedy")

        return text, metrics, timestamps

    def _postprocess_tokens(self, decoded_ids: List[int]) -> str:
        """
        Постобработка токенов в текст
        """
        # Фильтруем токены
        valid_tokens = []
        for tok_id in decoded_ids:
            if (tok_id < len(self.vocab) and
                    tok_id != self.blk_idx and
                    tok_id != self.pad_idx and
                    tok_id != self.blank_idx):
                token = self.vocab[tok_id]
                # Пропускаем пустые токены
                if token.strip():
                    valid_tokens.append(token)

        # Убираем дублирующиеся токены подряд
        filtered_tokens = []
        last_token = None
        for token in valid_tokens:
            if token != last_token:
                filtered_tokens.append(token)
                last_token = token
            elif len(filtered_tokens) == 0:  # Первый токен всегда добавляем
                filtered_tokens.append(token)

        # Собираем текст
        text = "".join(filtered_tokens)
        text = text.replace("▁", " ")

        # Базовая очистка
        text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
        text = text.strip()

        # Убираем повторяющиеся символы (более 2 подряд)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Убираем повторяющиеся слова
        words = text.split()
        cleaned_words = []
        last_word = None

        for word in words:
            word = word.strip()
            if word and (word.lower() != (last_word or "").lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word

        text = " ".join(cleaned_words)

        # Финальная очистка
        text = re.sub(r'\s+', ' ', text).strip()

        print(f"Final transcription (Improved Greedy): '{text}'")
        print(f"Tokens processed: {len(decoded_ids)} -> {len(valid_tokens)} -> {len(filtered_tokens)}")

        return text

    def _postprocess_tokens_improved(self, decoded_ids: List[int]) -> str:
        valid_tokens = []
        for tok_id in decoded_ids:
            if (tok_id < len(self.vocab) and
                    tok_id != self.blk_idx and
                    tok_id != self.pad_idx and
                    tok_id != self.blank_idx):
                token = self.vocab[tok_id]
                if token.strip():
                    valid_tokens.append(token)

        print(f"Valid tokens: {len(valid_tokens)}")

        filtered_tokens = []
        i = 0
        while i < len(valid_tokens):
            token = valid_tokens[i]
            consecutive_count = 1
            while (i + consecutive_count < len(valid_tokens) and
                   valid_tokens[i + consecutive_count] == token):
                consecutive_count += 1
            add_count = min(consecutive_count, 1)
            if len(token.replace('▁', '').strip()) <= 2:
                add_count = 1
            for _ in range(add_count):
                filtered_tokens.append(token)
            i += consecutive_count

        print(f"After deduplication: {len(filtered_tokens)} tokens")

        text = "".join(filtered_tokens)
        text = text.replace("▁", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            word = words[i].strip()
            if not word:
                i += 1
                continue
            consecutive_word_count = 1
            while (i + consecutive_word_count < len(words) and
                   words[i + consecutive_word_count].strip().lower() == word.lower()):
                consecutive_word_count += 1
            if len(word) > 3:
                add_word_count = min(consecutive_word_count, 2)
            else:
                add_word_count = 1
            for _ in range(add_word_count):
                cleaned_words.append(word)
            i += consecutive_word_count

        text = " ".join(cleaned_words)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[.,!?]$', '', text).strip()  # Удаление точки в конце

        print(f"Final transcription (Fixed Greedy): '{text}'")
        print(f"Token processing: {len(decoded_ids)} -> {len(valid_tokens)} -> {len(filtered_tokens)}")
        print(f"Final word count: {len(cleaned_words)}")

        return text

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
            transcription, _, timestamps = self.decode_rnnt_greedy_improved(
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
            transcription, _, timestamps = self.decode_rnnt_beam_search_fixed(
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

    def decode_rnnt_beam_search_fixed(self,
                                      encoder_output: np.ndarray,
                                      vocab: List[str],
                                      blank_idx: int,
                                      max_vocab_idx: int,
                                      beam_width: int = 5,
                                      length_penalty: float = 0.5,  # Уменьшен штраф за длину
                                      ground_truth: str = None,
                                      max_steps: int = 1000,
                                      min_tokens: int = 10,
                                      max_tokens_per_step: int = 10,
                                      # Этот параметр не используется в текущей реализации
                                      state_init: str = "zero") -> Tuple[str, Dict[str, float], List[int]]:
        """
        Исправленный beam search с лучшим балансом между качеством и контролем повторений
        """
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        print(f"Starting fixed beam search: beam_width={beam_width}, time_frames={time_frames}")

        # Инициализация состояний
        if state_init == "random":
            state1 = np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32)
            state2 = np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32)
        else:
            state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
            state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)

        # Структура луча: (sequence, score, state, prev_token, time_step, token_count, timestamps)
        initial_prev_token = np.array([[blank_idx]], dtype=np.int32)
        beams = [(tuple(), 0.0, (state1, state2), initial_prev_token, 0, 0, [])]

        # Параметры для контроля качества
        temperature = 1.0  # Увеличим температуру для большей "разнородности"
        blank_penalty = -0.5  # Менее агрессивный штраф за бланк

        step = 0
        max_iterations = min(time_frames * 2, max_steps)  # Увеличиваем лимит итераций

        # Кэш для результатов decode_step, чтобы избежать повторных вычислений
        # Ключ: (t, prev_token_id, state1_hash, state2_hash)
        # Значение: (logits, new_state1, new_state2)
        # Note: Хэширование состояний (np.ndarray) может быть медленным и потреблять много памяти.
        # Для простоты, пока что кэшируем только по (t, prev_token_id). Если это станет узким местом,
        # можно пересмотреть стратегию кэширования или его отключение.
        cache = {}

        while step < max_iterations:
            new_beams = {}

            # Сортируем лучи по их нормализованной оценке (с учетом length_penalty)
            # Это гарантирует, что мы расширяем наиболее перспективные лучи
            beams = sorted(
                beams,
                key=lambda x: x[1] / (max(1, x[5]) ** length_penalty) if x[5] > 0 else x[1],
                reverse=True
            )[:beam_width]

            for seq, score, (state1, state2), prev_token, curr_t, token_count, timestamps in beams:
                # Проверяем, нужно ли продолжать этот луч
                if curr_t >= time_frames:
                    # Луч завершён, сохраняем его
                    new_beams[seq] = (seq, score, (state1, state2), prev_token, curr_t, token_count, timestamps)
                    continue

                # Ключ для кэша: текущий шаг по времени и предыдущий токен
                cache_key = (curr_t, prev_token[0, 0].item())

                if cache_key in cache:
                    logits, new_state1, new_state2 = cache[cache_key]
                else:
                    try:
                        logits, (new_state1, new_state2) = self.decode_step(
                            encoder_output, prev_token, (state1, state2), curr_t
                        )
                        cache[cache_key] = (logits, new_state1, new_state2)
                    except Exception as e:
                        print(f"Error in decode_step at step {step}, t={curr_t}: {e}")
                        continue

                if logits.size == 0:
                    continue

                logits_t = logits[0, 0, 0].copy()

                # Применяем штрафы более осторожно
                logits_t[blank_idx] += blank_penalty

                # Штраф за <blk> токен, если он отличается от blank_idx
                if self.blk_idx != blank_idx and self.blk_idx < len(logits_t):
                    logits_t[self.blk_idx] -= 0.5  # Мягкий штраф

                # Штраф за недопустимые токены (если есть)
                for i in range(max_vocab_idx + 1, len(logits_t)):
                    logits_t[i] -= 8.0  # Сильный штраф за токены вне словаря

                # Штраф за PAD токены
                if hasattr(self, 'pad_idx') and self.pad_idx < len(logits_t):
                    logits_t[self.pad_idx] -= 6.0  # Сильный штраф за PAD

                # Мягкий контроль повторений - только для очень частых повторений
                # Увеличим порог для штрафа за повторения
                if len(seq) >= 2:
                    last_token = seq[-1]
                    # Штраф за повторение одного и того же токена 2+ раза подряд
                    if last_token == token and token != blank_idx:
                        logits_t[token] -= 0.5  # Небольшой штраф
                    if len(seq) >= 3 and seq[-2] == last_token and last_token == token and token != blank_idx:
                        logits_t[token] -= 1.0  # Больший штраф за 3+ повторения

                # Применяем температуру
                logits_t = logits_t / temperature
                logits_t = logits_t - np.max(logits_t)  # Для численной стабильности
                exp_logits = np.exp(logits_t)
                probs = exp_logits / (np.sum(exp_logits) + 1e-12)

                # Выбираем топ кандидатов
                # Увеличим количество рассматриваемых кандидатов
                top_k = min(len(probs), beam_width * 3)  # Рассматриваем больше токенов
                top_indices = np.argsort(probs)[-top_k:]
                top_probs = probs[top_indices]

                # Сортируем по вероятности
                sorted_idx = np.argsort(top_probs)[::-1]
                top_indices = top_indices[sorted_idx]
                top_probs = top_probs[sorted_idx]

                # Обрабатываем каждый кандидат
                for i, (token, prob) in enumerate(zip(top_indices, top_probs)):
                    if prob < 1e-6:  # Игнорируем очень маловероятные
                        continue

                    new_score = score + np.log(prob + 1e-12)

                    if token == blank_idx:
                        # Blank токен - не эмитируем, но продвигаемся по времени
                        new_seq = seq
                        new_timestamps = timestamps
                        new_count = token_count
                        new_prev_token = prev_token  # prev_token остается бланк
                    else:
                        # Эмитируем токен
                        if token >= len(vocab) or token > max_vocab_idx:
                            continue  # Невалидный токен

                        new_seq = seq + (token,)
                        new_timestamps = timestamps + [curr_t]
                        new_count = token_count + 1
                        new_prev_token = np.array([[token]], dtype=np.int32)

                        # Добавим небольшой штраф за "бессмысленные" токены, если они не являются частью слова
                        # Это можно улучшить, используя знание о границах слов из словаря.
                        # Пока что, просто очень мягкий штраф, если токен не пробел и не часть предыдущего токена
                        if len(seq) > 0 and self.vocab[token].strip() and not self.vocab[seq[-1]].endswith(" ") and not \
                        self.vocab[token].startswith(" "):
                            new_score -= 0.05

                    # Штраф за слишком длинные последовательности (более мягкий)
                    if new_count > 50:  # Увеличим допустимую длину
                        new_score -= 0.1 * (new_count - 50)

                    # Сохраняем луч
                    beam_key = new_seq
                    if beam_key not in new_beams or new_beams[beam_key][1] < new_score:
                        new_beams[beam_key] = (
                            new_seq, new_score, (new_state1, new_state2),
                            new_prev_token, curr_t + 1, new_count, new_timestamps
                        )

            if not new_beams:
                print(f"No beams at step {step}")
                break

            # Очистка кэша после каждого шага времени, чтобы избежать слишком большого потребления памяти
            cache.clear()

            # Выбираем лучшие лучи с учётом length_penalty
            beams = sorted(
                new_beams.values(),
                key=lambda x: x[1] / (max(1, x[5]) ** length_penalty),
                reverse=True
            )[:beam_width]

            step += 1

            # Логирование прогресса
            if step % 50 == 0:
                if beams:
                    best_beam = beams[0]
                    # Избегаем деления на ноль, если токенов нет
                    score_divisor = (max(1, best_beam[5]) ** length_penalty)
                    best_score_norm = best_beam[1] / score_divisor if score_divisor != 0 else best_beam[1]
                    print(f"Step {step}: best_score={best_score_norm:.4f}, tokens={best_beam[5]}, t={best_beam[4]}")
                else:
                    print(f"Step {step}: No beams left.")

            # Проверка на завершение всех лучей (если все лучи достигли конца аудио)
            if all(beam[4] >= time_frames for beam in beams) and step > min_tokens:  # Дополнительное условие
                print(f"All beams completed at step {step}")
                break

        # Выбираем лучший результат
        if not beams:
            return "", {}, []

        best_beam = beams[0]
        best_seq, best_score, _, _, _, token_count, timestamps = best_beam

        # Постобработка
        text = self._postprocess_tokens_conservative(list(best_seq))

        print(f"Fixed Beam Search completed:")
        print(f"  Transcription: '{text}'")
        print(f"  Log probability: {best_score:.6f}")
        # Избегаем деления на ноль, если токенов нет
        score_divisor = (max(1, token_count) ** length_penalty)
        normalized_score = best_score / score_divisor if score_divisor != 0 else best_score
        print(f"  Normalized score: {normalized_score:.6f}")
        print(f"  Total tokens: {token_count}")
        print(f"  Steps: {step}")

        # Метрики
        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(
                transcription=text,
                ground_truth=ground_truth,
                metrics=metrics,
                total_log_prob=best_score,
                beam_width=beam_width,
                length_penalty=length_penalty,
                flag="fixed_beam"
            )

        return text, metrics, timestamps

    def _postprocess_tokens_conservative(self, decoded_ids: List[int]) -> str:
        """
        Консервативная постобработка для удаления артефактов без потери информации.
        """
        # Фильтруем только явно невалидные токены (бланк, пад, <blk>, токены вне словаря)
        # идущие подряд повторяющиеся токены.

        filtered_tokens = []
        for i, tok_id in enumerate(decoded_ids):
            # Проверяем, является ли токен валидным и не является ли он одним из бланков/пада
            if (tok_id < len(self.vocab) and
                    tok_id != self.blank_idx and
                    tok_id != self.true_blank_idx and  # Проверяем оба индекса бланка
                    tok_id != self.pad_idx):
                token = self.vocab[tok_id]

                # Исключаем пустые строки, которые могут возникнуть из vocab
                if not token.strip():
                    continue

                # Удаляем повторяющиеся пробелы (если токен - это только пробел)
                if token == " " and filtered_tokens and filtered_tokens[-1] == " ":
                    continue

                # Удаляем повторяющиеся символы (если токен - это символ)
                if len(token) == 1 and filtered_tokens and filtered_tokens[-1] == token:
                    continue

                # Удаляем повторяющиеся слова или части слов, которые являются полными токенами
                # Если предыдущий токен такой же, и он не является частью другого слова (например, " а" и "а")
                if filtered_tokens and filtered_tokens[-1] == token:
                    continue  # Пропускаем точное повторение токена

                filtered_tokens.append(token)

        if not filtered_tokens:
            return ""

        # Объединяем токены в строку
        text = "".join(filtered_tokens)

        # Заменяем все случаи "  " (два пробела) на один пробел
        text = text.replace("  ", " ").strip()

        # Теперь обрабатываем повторения на уровне слов, более консервативно
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            word = words[i].strip()
            if not word:
                i += 1
                continue

            # Добавляем слово, если это первое слово или оно отличается от предыдущего
            if not cleaned_words or cleaned_words[-1].lower() != word.lower():
                cleaned_words.append(word)
            else:
                # Если слово повторяется, но не более одного раза подряд (т.е., два слова подряд одинаковые)
                # оставляем только одно. Если три или более, оставляем одно.
                # Эта логика уже учтена при формировании `filtered_tokens`,
                # но можно добавить здесь дополнительную проверку для слов.
                pass  # Пропускаем, если слово точно такое же, как предыдущее

            i += 1

        text = " ".join(cleaned_words)

        # Удаление лишних знаков препинания только в самом конце строки
        text = re.sub(r'[.,!?;:"]+$', '', text).strip()

        # Дополнительная очистка от случайных одиночных букв, если они не являются словами.
        # Например, "а а а" становится "а". Это можно сделать, если словарь позволяет.
        # Но пока что, не будем это делать, чтобы не удалять значимые токены.

        # Удаление повторяющихся символов внутри слова (например, "ооочень" -> "очень")
        # Делаем это очень осторожно, только для 3 и более повторений, оставляя 2
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        return text

    # def decode_rnnt_beam_search_improved(self,
    #                                      encoder_output: np.ndarray,
    #                                      vocab: List[str],
    #                                      blank_idx: int,
    #                                      max_vocab_idx: int,
    #                                      beam_width: int = 5,
    #                                      length_penalty: float = 1.0,
    #                                      ground_truth: str = None,
    #                                      max_steps: int = 1000,
    #                                      min_tokens: int = 10,
    #                                      max_tokens_per_step: int = 10,
    #                                      state_init: str = "zero") -> Tuple[str, Dict[str, float], List[int]]:
    #     """
    #     Улучшенный beam search с лучшим контролем повторений и качества
    #     """
    #     if encoder_output.shape[0] != 1:
    #         raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")
    #
    #     batch_size, hidden_size, time_frames = encoder_output.shape
    #
    #     # Инициализация состояний
    #     if state_init == "random":
    #         state1 = np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32)
    #         state2 = np.random.normal(0, 0.01, (1, 1, self.hidden_size)).astype(np.float32)
    #     else:
    #         state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
    #         state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
    #
    #     # Структура луча: (sequence, score, state, prev_token, time_step, token_count, timestamps, last_tokens_history)
    #     beams = [(tuple(), 0.0, (state1, state2), np.array([[blank_idx]]), 0, 0, [], [])]
    #
    #     # Кэш для избежания повторных вычислений
    #     cache = {}
    #
    #     # Параметры контроля качества
    #     temperature = 0.8
    #     blank_penalty = -2.0
    #     repetition_penalty = 0.3
    #     max_consecutive_repeats = 2
    #     early_stop_threshold = 0.95  # Если лучший луч имеет очень высокую уверенность
    #
    #     print(f"Starting improved beam search: beam_width={beam_width}, time_frames={time_frames}")
    #
    #     step = 0
    #     consecutive_no_progress = 0
    #
    #     while step < min(time_frames, max_steps):
    #         new_beams = {}
    #         best_score = float('-inf')
    #
    #         for seq, score, (state1, state2), prev_token, curr_t, token_count, timestamps, history in beams:
    #             # Проверка завершения для данного луча
    #             if curr_t >= time_frames:
    #                 if token_count >= min_tokens:
    #                     # Финальный луч
    #                     final_score = score / max(1, token_count) ** length_penalty
    #                     new_beams[seq] = (
    #                     seq, score, (state1, state2), prev_token, curr_t, token_count, timestamps, history)
    #                 continue
    #
    #             # Создание ключа для кэша
    #             cache_key = (
    #             tuple(seq[-10:]), tuple(prev_token.flatten()), curr_t)  # Используем только последние 10 токенов
    #
    #             if cache_key in cache:
    #                 logits, (new_state1, new_state2) = cache[cache_key]
    #             else:
    #                 try:
    #                     logits, (new_state1, new_state2) = self.decode_step(encoder_output, prev_token,
    #                                                                         (state1, state2), curr_t)
    #                     cache[cache_key] = (logits, (new_state1, new_state2))
    #                 except Exception as e:
    #                     print(f"Error in decode_step at t={curr_t}: {e}")
    #                     continue
    #
    #             if logits.size == 0:
    #                 continue
    #
    #             logits_t = logits[0, 0, 0].copy()
    #
    #             # Применение штрафов и бонусов
    #             logits_t[blank_idx] += blank_penalty
    #
    #             # Штраф за превышение максимального индекса словаря
    #             for i in range(max_vocab_idx + 1, len(logits_t)):
    #                 logits_t[i] -= 10.0
    #
    #             # Штраф за PAD токены
    #             if hasattr(self, 'pad_idx') and self.pad_idx < len(logits_t):
    #                 logits_t[self.pad_idx] -= 8.0
    #
    #             # Анализ истории для штрафов за повторения
    #             if len(history) >= 3:
    #                 # Штраф за повторяющиеся токены
    #                 recent_tokens = history[-3:]
    #                 for token in set(recent_tokens):
    #                     if recent_tokens.count(token) >= max_consecutive_repeats and token < len(logits_t):
    #                         logits_t[token] *= repetition_penalty
    #
    #             # Штраф за повторение последнего эмитированного токена
    #             if len(seq) > 0:
    #                 last_token = seq[-1]
    #                 if last_token < len(logits_t):
    #                     logits_t[last_token] *= 0.7
    #
    #             # Штраф за слишком частые одинаковые токены в последовательности
    #             if len(seq) >= 2:
    #                 for token in set(seq[-5:]):  # Проверяем последние 5 токенов
    #                     token_count_recent = seq[-5:].count(token)
    #                     if token_count_recent >= 2 and token < len(logits_t):
    #                         logits_t[token] *= (0.5 ** token_count_recent)
    #
    #             # Нормализация с температурой
    #             logits_t = logits_t / temperature
    #             logits_t = logits_t - np.max(logits_t)  # Стабилизация
    #             exp_logits = np.exp(logits_t)
    #             probs = exp_logits / (np.sum(exp_logits) + 1e-12)
    #
    #             # Выбор топ-K кандидатов (более консервативный подход)
    #             top_k = min(beam_width * 3, 50)
    #             top_indices = np.argpartition(probs, -top_k)[-top_k:]
    #             top_probs = probs[top_indices]
    #
    #             # Сортировка и выбор лучших
    #             sorted_indices = np.argsort(top_probs)[-beam_width:]
    #             selected_indices = top_indices[sorted_indices]
    #             selected_probs = top_probs[sorted_indices]
    #
    #             for token, prob in zip(selected_indices, selected_probs):
    #                 if prob < 1e-10:  # Игнорируем слишком маловероятные токены
    #                     continue
    #
    #                 new_score = score + np.log(prob + 1e-12)
    #
    #                 # Дополнительные проверки для blank токена
    #                 if token == blank_idx:
    #                     new_seq = seq
    #                     new_timestamps = timestamps
    #                     new_count = token_count
    #                     new_history = history
    #                     new_prev_token = prev_token
    #                 else:
    #                     # Проверка на валидность токена
    #                     if token >= len(vocab) or token > max_vocab_idx:
    #                         continue
    #
    #                     # Дополнительная проверка повторений для не-blank токенов
    #                     if len(seq) >= 2 and seq[-1] == token and seq[-2] == token:
    #                         new_score -= 5.0  # Сильный штраф за тройное повторение
    #
    #                     new_seq = seq + (token,)
    #                     new_timestamps = timestamps + [curr_t]
    #                     new_count = token_count + 1
    #                     new_history = (history + [token])[-10:]  # Ограничиваем историю
    #                     new_prev_token = np.array([[token]], dtype=np.int32)
    #
    #                 # Ограничение максимальной длины
    #                 if new_count > 50:  # Жесткое ограничение
    #                     new_score -= 20.0 * (new_count - 50)
    #
    #                 # Бонус за разнообразие (если токен не повторяется часто)
    #                 if token != blank_idx and len(seq) > 5:
    #                     recent_diversity = len(set(seq[-5:])) / min(5, len(seq[-5:]))
    #                     if recent_diversity > 0.6:  # Хорошее разнообразие
    #                         new_score += 0.5
    #
    #                 # Обновление лучшего луча
    #                 sequence_key = new_seq
    #                 normalized_score = new_score / max(1, new_count) ** length_penalty
    #
    #                 if sequence_key not in new_beams or new_beams[sequence_key][1] < new_score:
    #                     new_beams[sequence_key] = (
    #                         new_seq, new_score, (new_state1, new_state2),
    #                         new_prev_token, curr_t + 1, new_count, new_timestamps, new_history
    #                     )
    #                     best_score = max(best_score, normalized_score)
    #
    #         if not new_beams:
    #             print(f"No valid beams at step {step}, breaking")
    #             break
    #
    #         # Выбор лучших лучей
    #         beams = sorted(new_beams.values(),
    #                        key=lambda x: x[1] / max(1, x[5]) ** length_penalty,
    #                        reverse=True)[:beam_width]
    #
    #         # Проверка прогресса
    #         current_best_score = beams[0][1] / max(1, beams[0][5]) ** length_penalty
    #         if step > 0 and abs(current_best_score - best_score) < 0.001:
    #             consecutive_no_progress += 1
    #         else:
    #             consecutive_no_progress = 0
    #
    #         # Ранняя остановка при отсутствии прогресса
    #         if consecutive_no_progress > 20:
    #             print(f"Early stopping due to no progress at step {step}")
    #             break
    #
    #         step += 1
    #
    #         # Периодическая очистка кэша
    #         if step % 100 == 0:
    #             cache.clear()
    #             print(f"Step {step}: Best score = {current_best_score:.4f}, tokens = {beams[0][5]}")
    #
    #     # Выбор лучшего результата
    #     if not beams:
    #         return "", {}, []
    #
    #     best_seq, best_score, _, _, _, token_count, timestamps, _ = beams[0]
    #
    #     # Постобработка результата
    #     text = self._postprocess_tokens_advanced(list(best_seq))
    #
    #     print(f"Improved Beam Search completed:")
    #     print(f"  Transcription: '{text}'")
    #     print(f"  Log probability: {best_score:.6f}")
    #     print(f"  Total tokens: {token_count}")
    #     print(f"  Steps processed: {step}")
    #
    #     # Вычисление метрик
    #     metrics = {}
    #     if ground_truth and text:
    #         metrics = return_metrics(transcription=text,
    #                                  ground_truth=ground_truth,
    #                                  metrics=metrics,
    #                                  total_log_prob=best_score,
    #                                  beam_width=beam_width,
    #                                  length_penalty=length_penalty,
    #                                  flag="improved_beam")
    #
    #     return text, metrics, timestamps
    #
    # def _postprocess_tokens_advanced(self, decoded_ids: List[int]) -> str:
    #     """
    #     Продвинутая постобработка токенов с лучшим контролем повторений
    #     """
    #     # Фильтрация валидных токенов
    #     valid_tokens = []
    #     for tok_id in decoded_ids:
    #         if (tok_id < len(self.vocab) and
    #                 tok_id != self.blk_idx and
    #                 tok_id != self.pad_idx and
    #                 tok_id != self.blank_idx and
    #                 tok_id != self.true_blank_idx):
    #             token = self.vocab[tok_id]
    #             if token.strip():
    #                 valid_tokens.append(token)
    #
    #     if not valid_tokens:
    #         return ""
    #
    #     # Удаление последовательных дубликатов токенов
    #     filtered_tokens = []
    #     i = 0
    #     while i < len(valid_tokens):
    #         token = valid_tokens[i]
    #
    #         # Подсчет последовательных повторений
    #         consecutive_count = 1
    #         while (i + consecutive_count < len(valid_tokens) and
    #                valid_tokens[i + consecutive_count] == token):
    #             consecutive_count += 1
    #
    #         # Решение о количестве токенов для добавления
    #         if len(token.replace('▁', '').strip()) <= 2:
    #             # Короткие токены - максимум 1 повторение
    #             add_count = 1
    #         elif consecutive_count >= 3:
    #             # Много повторений - оставляем максимум 2
    #             add_count = min(2, consecutive_count)
    #         else:
    #             # Нормальный случай
    #             add_count = consecutive_count
    #
    #         for _ in range(add_count):
    #             filtered_tokens.append(token)
    #
    #         i += consecutive_count
    #
    #     # Собираем текст
    #     text = "".join(filtered_tokens)
    #     text = text.replace("▁", " ")
    #
    #     # Очистка множественных пробелов
    #     text = re.sub(r'\s+', ' ', text).strip()
    #
    #     # Удаление последовательных повторяющихся символов (больше 2 подряд)
    #     text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    #
    #     # Обработка слов - удаление дубликатов
    #     words = text.split()
    #     cleaned_words = []
    #
    #     i = 0
    #     while i < len(words):
    #         word = words[i].strip().lower()
    #         if not word:
    #             i += 1
    #             continue
    #
    #         # Подсчет последовательных повторений слов
    #         consecutive_word_count = 1
    #         while (i + consecutive_word_count < len(words) and
    #                words[i + consecutive_word_count].strip().lower() == word):
    #             consecutive_word_count += 1
    #
    #         # Добавляем оригинальное слово (сохраняем регистр первого вхождения)
    #         original_word = words[i].strip()
    #
    #         if len(word) >= 4:
    #             # Длинные слова - возможно оставить 2 повторения
    #             add_word_count = min(2, consecutive_word_count)
    #         else:
    #             # Короткие слова - только одно
    #             add_word_count = 1
    #
    #         for _ in range(add_word_count):
    #             cleaned_words.append(original_word)
    #
    #         i += consecutive_word_count
    #
    #     # Финальная сборка
    #     text = " ".join(cleaned_words)
    #     text = re.sub(r'\s+', ' ', text).strip()
    #
    #     # Удаление лишних знаков препинания в конце
    #     text = re.sub(r'[.,!?;:]+$', '', text).strip()
    #
    #     # Удаление изолированных коротких повторяющихся слов
    #     words = text.split()
    #     final_words = []
    #     for i, word in enumerate(words):
    #         # Если слово короткое и повторяется с предыдущим, пропускаем
    #         if (i > 0 and len(word) <= 2 and
    #                 word.lower() == words[i - 1].lower()):
    #             continue
    #         final_words.append(word)
    #
    #     final_text = " ".join(final_words)
    #
    #     print(
    #         f"Advanced postprocessing: {len(decoded_ids)} -> {len(valid_tokens)} -> {len(filtered_tokens)} -> {len(final_words)} words")
    #
    #     return final_text
    #
    # def recognize_fixed_improved(self,
    #                              waveforms: np.ndarray,
    #                              decode_flag: str = "GD",
    #                              beam_width: int = 10,
    #                              length_penalty: float = 0.7,
    #                              ground_truth: str = None,
    #                              max_steps: int = 1000,
    #                              min_tokens: int = 10,
    #                              state_init: str = "zero") -> Tuple[str, List[int]]:
    #     """
    #     Улучшенный метод распознавания с продвинутым beam search
    #     """
    #     if not isinstance(waveforms, np.ndarray):
    #         raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")
    #
    #     # Предобработка аудио
    #     if waveforms.dtype != np.float32:
    #         waveforms = waveforms.astype(np.float32)
    #     if waveforms.ndim == 2 and waveforms.shape[0] > 1:
    #         waveforms = np.mean(waveforms, axis=0)
    #     elif waveforms.ndim == 2:
    #         waveforms = waveforms[0]
    #     waveforms = waveforms.flatten()
    #
    #     audio_length_samples = len(waveforms)
    #     expected_time_frames = audio_length_samples // 160 + 1
    #     print(f"Input audio length: {audio_length_samples} samples, "
    #           f"expected time frames: {expected_time_frames}")
    #
    #     # Преобразование в тензор PyTorch
    #     audio_tensor = torch.from_numpy(waveforms).float().unsqueeze(0)
    #     audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
    #
    #     # Улучшенная нормализация аудио
    #     if audio_tensor.std() < 1e-6:
    #         print("Warning: audio has very low variance, applying minimal normalization")
    #         audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-6)
    #     else:
    #         # Стандартная z-score нормализация
    #         mean = audio_tensor.mean()
    #         std = audio_tensor.std()
    #         audio_tensor = (audio_tensor - mean) / (std + 1e-6)
    #
    #         # Дополнительное ограничение амплитуды
    #         audio_tensor = torch.clamp(audio_tensor, -3.0, 3.0)
    #
    #     # Извлечение признаков
    #     features, lengths = self.extract_features(audio_tensor, audio_length)
    #     print(f"Features shape: {features.shape}, lengths: {lengths}")
    #
    #     # Проверка на NaN/Inf в признаках
    #     if torch.isnan(features).any() or torch.isinf(features).any():
    #         print("Warning: Features contain NaN or Inf, applying clipping")
    #         features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    #         features = torch.clamp(features, -10.0, 10.0)
    #
    #     # Конвертация в numpy
    #     features_np = features.detach().cpu().numpy().astype(np.float32)
    #     lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
    #
    #     # Кодирование
    #     encoder_output = self.encode(features_np, lengths_np)
    #     time_frames = encoder_output.shape[2]
    #     print(f"Encoder output shape: {encoder_output.shape}, time_frames: {time_frames}")
    #
    #     # Выбор метода декодирования
    #     if decode_flag == "GD":
    #         transcription, _, timestamps = self.decode_rnnt_greedy_improved(
    #             encoder_output=encoder_output,
    #             vocab=self.vocab,
    #             blank_idx=self.true_blank_idx,
    #             max_vocab_idx=self.max_vocab_idx,
    #             ground_truth=ground_truth,
    #             max_steps=max_steps,
    #             min_tokens=min_tokens,
    #             state_init=state_init
    #         )
    #     elif decode_flag == "BS":
    #         # Используем улучшенный beam search
    #         transcription, _, timestamps = self.decode_rnnt_beam_search_improved(
    #             encoder_output=encoder_output,
    #             vocab=self.vocab,
    #             blank_idx=self.blank_idx,
    #             max_vocab_idx=self.max_vocab_idx,
    #             beam_width=beam_width,
    #             length_penalty=length_penalty,
    #             ground_truth=ground_truth,
    #             max_steps=max_steps,
    #             min_tokens=min_tokens,
    #             state_init=state_init
    #         )
    #     elif decode_flag == "BS_OLD":
    #         # Старый beam search для сравнения
    #         transcription, _, timestamps = self.decode_rnnt_beam_search(
    #             encoder_output=encoder_output,
    #             vocab=self.vocab,
    #             blank_idx=self.blank_idx,
    #             max_vocab_idx=self.max_vocab_idx,
    #             beam_width=beam_width,
    #             length_penalty=length_penalty,
    #             ground_truth=ground_truth,
    #             max_steps=max_steps,
    #             min_tokens=min_tokens,
    #             state_init=state_init
    #         )
    #     else:
    #         raise ValueError("decode_flag must be 'GD', 'BS', or 'BS_OLD'")
    #
    #     return transcription, timestamps
