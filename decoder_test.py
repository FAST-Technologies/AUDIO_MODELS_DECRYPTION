import librosa
import torch
import numpy as np
import onnxruntime as rt
from pathlib import Path
from typing import List, Tuple
import re
import torchaudio.transforms as T

# --- 1. Audio Loading and Preprocessing ---
audio, sr = librosa.load("audio_files/checking.wav", sr=16000)
audio_tensor = torch.tensor(audio).float().unsqueeze(0)
audio_len = torch.tensor([audio.shape[0]], dtype=torch.int64)

# Простая предобработка
def preprocess_audio(audio_tensor: torch.Tensor, audio_len: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # Применяем нормализацию
    mean = audio_tensor.mean()
    std = audio_tensor.std()
    if std == 0:
        std = 1e-6
    audio_tensor = (audio_tensor - mean) / (std + 1e-6)

    # Фиксированные параметры (настроены под модель)
    n_fft = 400
    win_length = 400
    hop_length = 160
    n_mels = 80
    preemph = 0.97
    log_zero_guard_value = 2 ** -24

    # Преэмфазис
    if preemph != 0.0:
        audio_tensor = torch.cat([audio_tensor[:, :1], audio_tensor[:, 1:] - preemph * audio_tensor[:, :-1]], dim=-1)

    # Создаем спектрограмму
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=2.0
    ).to(audio_tensor.device)
    spectrogram = spectrogram_transform(audio_tensor)  # [batch, freq, time]

    # Создаем Mel-фильтры
    mel_transform = T.MelScale(
        n_mels=n_mels,
        sample_rate=sr,
        f_min=0,
        f_max=sr // 2,
        n_stft=n_fft // 2 + 1
    ).to(audio_tensor.device)
    mel_spec = mel_transform(spectrogram)  # [batch, n_mels, time]

    # Логарифмирование и CMVN
    log_mel_spec = torch.log(mel_spec + log_zero_guard_value)
    mean = log_mel_spec.mean(dim=2, keepdim=True)
    std = log_mel_spec.std(dim=2, keepdim=True)
    log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

    # Передаем в формате [batch, n_mels, time]
    features = log_mel_spec.numpy().astype(np.float32)
    features_len = (audio_len / hop_length + 1).long().numpy().astype(np.int64)
    return features, features_len

features, features_len = preprocess_audio(audio_tensor, audio_len)
print(f"Features shape: {features.shape}")
print(f"Features length: {features_len}")

# --- 2. OnnxConformerRNNT Class ---
class OnnxConformerRNNT:
    def __init__(self, model_files: dict):
        self._encoder = rt.InferenceSession(model_files["encoder"], providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], providers=["CPUExecutionProvider"])
        self.vocab = self._load_vocab(model_files["vocab"])
        self._setup_token_indices()
        self._print_model_info()

    def _load_vocab(self, vocab_path: str) -> List[str]:
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
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _encode(self, features: np.ndarray, features_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def _decode(self, prev_tokens: List[int], prev_state: Tuple[np.ndarray, np.ndarray], encoder_out: np.ndarray) -> \
    Tuple[np.ndarray, int, Tuple[np.ndarray, np.ndarray]]:
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

    def greedy_search(self, encoder_out: np.ndarray, encoder_out_len: np.ndarray) -> str:
        max_len = encoder_out.shape[2]
        print(f"Starting greedy decoding with {max_len} time frames")
        state = (np.zeros((1, 1, 640), dtype=np.float32), np.zeros((1, 1, 640), dtype=np.float32))
        hyp = []
        blank_penalty = -0.5
        temperature = 1.0
        repeat_penalty = 0.1

        for t in range(max_len):
            current_encoder_out = encoder_out[:, :, t:t + 1]
            logits, _, state = self._decode(hyp, state, current_encoder_out)
            logits = logits.copy()
            if self._blank_idx < len(logits):
                logits[self._blank_idx] += blank_penalty
            if self._pad_idx < len(logits):
                logits[self._pad_idx] -= 5.0
            if temperature != 1.0:
                logits = logits / temperature
            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-12)

            if hyp and hyp[-1] != self._blank_idx:
                probs[hyp[-1]] *= repeat_penalty

            next_token = np.argmax(probs).item()
            if probs[next_token] < 0.4:
                next_token = self._blank_idx
            if next_token != self._blank_idx and (not hyp or next_token != hyp[-1]):
                hyp.append(next_token)
            if t % 5 == 0:
                token_str = self.vocab[next_token] if next_token < len(self.vocab) else 'OUT_OF_VOCAB'
                print(f"Step {t}: token={next_token}('{token_str}'), prob={probs[next_token]:.3f}")

        return self._postprocess(hyp)

    def _postprocess(self, decoded_ids: List[int]) -> str:
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if
                        tok_id not in self._tokens_to_filter and tok_id <= self._max_vocab_idx]
        text = "".join(valid_tokens).replace("▁", " ").strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        words = text.split()
        cleaned_words = []
        last_word = None
        for word in words:
            if word and (not last_word or word.lower() != last_word.lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word
        text = " ".join(cleaned_words).strip()
        return text

# --- 3. Model Initialization and Inference ---
model_files = {
    "encoder": "onnx_models/encoder-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
    "decoder_joint": "onnx_models/decoder_joint-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
    "vocab": "onnx_models/vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt"
}

onnx_model = OnnxConformerRNNT(model_files)
encoder_out, encoder_out_len = onnx_model._encode(features, features_len)
print(f"Encoder output shape: {encoder_out.shape}")
print(f"Encoder output length: {encoder_out_len}")

# === 6. Декодинг (greedy search) ===
transcription = onnx_model.greedy_search(encoder_out, encoder_out_len)
print("Transcription:", transcription)