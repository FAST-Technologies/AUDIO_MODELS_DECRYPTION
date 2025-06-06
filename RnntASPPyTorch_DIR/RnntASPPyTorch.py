import torch
import numpy as np
import onnxruntime as rt
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import re
from MetricsClass import return_metrics

# Параметры предобработки
sample_rate = 16_000
n_fft = 400
win_length = 400
hop_length = 160
n_mels = 80
features = 80
preemph = 0.97
log_zero_guard_value = 2 ** -24
vocab_path = "onnx_models/vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt"

def preprocess_audio(audio_tensor: torch.Tensor, audio_len: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # Применяем нормализацию
    mean = audio_tensor.mean()
    std = audio_tensor.std()
    if std == 0:
        std = 1e-6
    print(f"Mean before CMVN: {mean.item()}, Std before CMVN: {std.item()}")
    audio_tensor = (audio_tensor - mean) / (std + 1e-6)

    # Преэмфазис
    if preemph != 0.0:
        audio_tensor = torch.cat([audio_tensor[:, :1], audio_tensor[:, 1:] - preemph * audio_tensor[:, :-1]], dim=-1)

    time = audio_len.item()
    num_frames = int(np.floor(time / hop_length) + 1)  # Align with NumPy: 1413 frames
    features_len = torch.tensor([num_frames], dtype=torch.long).numpy().astype(np.int64)

    # Создаем спектрограмму
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=2.0
    ).to(audio_tensor.device)
    spectrogram = spectrogram_transform(audio_tensor)  # [batch, freq, time]

    if spectrogram.shape[-1] != num_frames:
        spectrogram = spectrogram[:, :, :num_frames]

    # Создаем Mel-фильтры
    mel_transform = T.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0,
        f_max=sample_rate // 2,
        n_stft=n_fft // 2 + 1
    ).to(audio_tensor.device)
    mel_spec = mel_transform(spectrogram)  # [batch, n_mels, time]
    print(f"Melspec Torch: {mel_spec}")


    # Логарифмирование и CMVN
    log_mel_spec = torch.log(mel_spec + log_zero_guard_value)
    print(f"log_mel_spec Torch: {mel_spec}")
    np.save("mel_spec_torch_raw.npy", log_mel_spec.numpy())
    mean = log_mel_spec.mean(dim=2, keepdim=True)
    std = log_mel_spec.std(dim=2, keepdim=True)
    log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
    np.save("mel_spec_torch_cmvn.npy", log_mel_spec.numpy())
    print(f"After CMVN: mean={log_mel_spec.mean().item()}, std={log_mel_spec.std().item()}")
    # Передаем в формате [batch, n_mels, time]
    features = log_mel_spec.numpy().astype(np.float32)
    # features_len = (audio_len / hop_length + 1).long().numpy().astype(np.int64)
    return features, features_len

# Загрузка вокабуляра
def load_vocab(vocab_path: str) -> List[str]:
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
    while len(vocab) < 1025:
        vocab.append("<pad>")
    return vocab

class RnntASRPyTorch:
    def __init__(self,
                 encoder_path: str,
                 decoder_joint_path: str):
        self.features = features
        self.hidden_size = 640
        self._encoder = rt.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(decoder_joint_path, providers=["CPUExecutionProvider"])
        self.vocab = load_vocab(vocab_path)
        self._setup_token_indices()
        self._print_model_info()

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

    def _setup_token_indices(self):
        self._blank_idx = self.vocab.index("<blk>") if "<blk>" in self.vocab else 1024
        self._unk_idx = self.vocab.index("<unk>") if "<unk>" in self.vocab else 1024
        self._blk_idx = self._blank_idx
        self._pad_idx = self.vocab.index("<pad>") if "<pad>" in self.vocab else 1024
        self._max_vocab_idx = len(self.vocab) - 1
        self._tokens_to_filter = {self._blank_idx, self._unk_idx, self._blk_idx, self._pad_idx}
        for i in range(self._max_vocab_idx + 1, len(self.vocab)):
            self._tokens_to_filter.add(i)

    def _print_model_info(self):
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
        print(f"First 10 tokens: {self.vocab[:10]}")
        print(f"Last 10 tokens: {self.vocab[-10:]}")
        print(f"Blank index: {self._blank_idx} ('{self.vocab[self._blank_idx]}')")
        print(f"UNK index: {self._unk_idx} ('{self.vocab[self._unk_idx]}')")
        print(f"Pad index: {self._pad_idx} ('{self.vocab[self._pad_idx]}')")
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _print_tensor_info(self, title: str, tensors: List[rt.NodeArg]) -> None:
        print(f"{title}:")
        for tensor in tensors:
            print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    def out_len(self, input_lengths: np.ndarray) -> np.ndarray:
        return np.floor_divide(input_lengths, 160) + 1

    def extract_features(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"input_signal shape: {input_signal.shape}, min: {input_signal.min().item()}, max: {input_signal.max().item()}")
        if torch.isnan(input_signal).any() or torch.isinf(input_signal).any():
            print("Warning: input_signal contains NaN or Inf values!")

        features, features_len = preprocess_audio(input_signal, length)

        plt.figure(figsize=(10, 4))
        plt.imshow(features[0], aspect="auto", origin="lower", interpolation="nearest")
        plt.colorbar(label="Normalized Log Mel Energy")
        plt.title("Mel-Spectrogram (After Normalization)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()
        plt.show()

        return torch.from_numpy(features), torch.from_numpy(features_len)

    def _encode(self,
                features: np.ndarray,
                features_lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        return logits, -1, (outputs[1], outputs[2])

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
        # logits = logits - np.max(logits)

        logits_t = logits[0, 0, 0].copy()
        print(f"DEBUG: t={t}, raw logits_t for blank {self._blank_idx}: {logits_t[self._blank_idx]:.4f}")
        non_blank_logits = np.delete(logits_t, self._blank_idx)
        non_blank_vocab_indices = np.delete(np.arange(len(self.vocab)), self._blank_idx)
        k_val = min(5, len(non_blank_logits))
        top_5_raw_indices_in_non_blank_array = np.argsort(non_blank_logits)[-k_val:]
        top_5_raw_indices = non_blank_vocab_indices[top_5_raw_indices_in_non_blank_array]

        print("DEBUG: t={t}, top 5 raw non-blank logits: " +
              ", ".join([f"{self.vocab[idx]}:{logits_t[idx]:.4f}" for idx in top_5_raw_indices]))

        return logits, (outputs[1], outputs[2])

    def decode_rnnt_greedy_improved(self,
                                    encoder_output: np.ndarray,
                                    ground_truth: str = None,
                                    state_init: str = "zero"
    ) -> Tuple[str, Dict[str, float], List[int]]:
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
            if self._blank_idx < len(logits):
                logits[self._blank_idx] += blank_penalty
            if self._pad_idx < len(logits):
                logits[self._pad_idx] -= 5.0
            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)))

            if hyp and hyp[-1] != self._blank_idx:
                probs[hyp[-1]] *= repeat_penalty

            next_token = np.argmax(probs).item()
            if next_token != self._blank_idx and (not hyp or next_token != hyp[-1]):
                hyp.append(next_token)
            if t % 5 == 0:
                token_str = self.vocab[next_token] if next_token < len(self.vocab) else 'OUT_OF_VOCAB'
                print(f"Step {t}: token={next_token}('{token_str}'), prob={probs[next_token]:.3f}")

        text = self._postprocess_improved(hyp, "GD")
        metrics = {}
        if ground_truth and text:
            metrics = return_metrics(transcription=text,
                                     ground_truth=ground_truth,
                                     metrics=metrics,
                                     total_log_prob=0.0,
                                     flag="improved_greedy")
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
                                      state_init: str = "zero"
    ) -> Tuple[str, Dict[str, float], List[int]]:
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

        # Инициализируем beam с пустой последовательностью и начальным токеном
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
                non_blank_logits = np.delete(logits_t, self._blank_idx)
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

        text = self._postprocess_improved(list(best_seq), "BS")

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
                flag="BS"
            )

        return text, metrics, timestamps

    def _postprocess_improved(self,
                              decoded_ids: List[int],
                              flag: str = "GD"
    ) -> str:
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if tok_id < len(self.vocab) and tok_id not in {self._blank_idx, self._pad_idx}]
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
        if flag == "GD":
            print(f"Final transcription (Improved Greedy Decoding): '{text}'")
        elif flag == "BS":
            print(f"Final transcription (Improved Beam Search): '{text}'")
        return text

    def _postprocess_tokens_conservative(self,
                                         decoded_ids: List[int]
    ) -> str:
        filtered_tokens = []
        special_ids = {self._blank_idx, self._pad_idx}

        for tok_id in decoded_ids:
            if tok_id < len(self.vocab) and tok_id not in special_ids:
                token = self.vocab[tok_id].strip()
                if token and token not in filtered_tokens[-1:]:
                    filtered_tokens.append(token)

        text = "".join(filtered_tokens).replace("▁", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        words = text.split()
        cleaned_words = []
        for i in range(len(words)):
            if i == 0 or words[i] != words[i - 1]:
                cleaned_words.append(words[i])

        text = " ".join(cleaned_words).strip()
        print(f"Final transcription (Improved Beam): '{text}'")
        return text

    def recognize(self,
                  waveforms: np.ndarray,
                  decode_flag: str = "GD",
                  ground_truth: str = None,
                  max_steps: int = 3000,
                  min_tokens: int = 15,
                  state_init: str = "zero",
                  beam_width: int = 8,
                  length_penalty: float = 0.7) -> Tuple[str, List[int]]:
        if not isinstance(waveforms, np.ndarray):
            raise TypeError(f"Expected waveforms to be a numpy.ndarray, got {type(waveforms)}")
        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        if waveforms.ndim == 2 and waveforms.shape[0] > 1:
            waveforms = np.mean(waveforms, axis=0)
        elif waveforms.ndim == 2:
            waveforms = waveforms[0]
        waveforms = waveforms.flatten()

        audio_tensor = torch.from_numpy(waveforms).float().unsqueeze(0)
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
        print(
            f"audio_tensor shape: {audio_tensor.shape}, audio_length: {audio_length}, min: {audio_tensor.min().item()}, max: {audio_tensor.max().item()}")
        if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
            print("Warning: audio_tensor contains NaN or Inf values!")

        # Вычисляем предсказанную длину выхода
        predicted_out_len = self.out_len(audio_length.numpy())
        print(f"Predicted output length: {predicted_out_len}")

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

        encoder_out_data, encoder_out_lengths = self._encode(features, lengths)
        print(f"Encoder output data shape: {encoder_out_data.shape}, lengths: {encoder_out_lengths}")

        if decode_flag == "GD":
            transcription, _, timestamps = self.decode_rnnt_greedy_improved(encoder_out_data,
                                                                            ground_truth,
                                                                            state_init)
        elif decode_flag == "BS":
            transcription, _, timestamps = self.decode_rnnt_beam_search_fixed(encoder_out_data,
                                                                              self.vocab,
                                                                              self._blank_idx,
                                                                              beam_width,
                                                                              length_penalty,
                                                                              ground_truth,
                                                                              max_steps,
                                                                              min_tokens,
                                                                              state_init)
        else:
            raise ValueError("decode_flag must be 'GD' or 'BS'")

        return transcription, timestamps