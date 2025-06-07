# Импорт необходимых библиотек
import librosa
import torch
import numpy as np
import onnxruntime as rt
from pathlib import Path
from typing import List, Tuple
import re
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# Параметры предобработки
sample_rate = 16_000
n_fft = 512
win_length = 400
hop_length = 160
n_mels = 80
features = 80
preemph = 0.97
log_zero_guard_value = 2 ** -24
vocab_path = "vocab.txt"


def preprocess_audio(audio_tensor: torch.Tensor,
                     audio_len: torch.Tensor
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess raw audio tensor to extract log-mel spectrogram features.

    Parameters
    ----------
    audio_tensor : Tensor
        Raw audio signal tensor, expected shape [batch, time].
    audio_len : Tensor
        Lengths of the input audio signals, expected shape [batch].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Log-mel spectrogram features, shape [batch, n_mels, time_frames].
        - Output lengths after feature extraction, shape [batch].

    Notes
    -----
    - Applies normalization using mean and standard deviation.
    - Performs pre-emphasis to enhance higher frequencies.
    - Computes spectrogram using Torchaudio's `Spectrogram` with a Hann window.
    - Applies Mel-scale transformation followed by logarithmic scaling.
    - Clamps values with `log_zero_guard_value` to avoid numerical issues during log computation.
    """
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
    num_frames = int(np.floor(time / hop_length) + 1)
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

    # Логарифмирование и CMVN
    log_mel_spec = torch.log(mel_spec + log_zero_guard_value)
    np.save("mel_spec_raw.npy", log_mel_spec.numpy())
    mean = log_mel_spec.mean(dim=2, keepdim=True)
    std = log_mel_spec.std(dim=2, keepdim=True)
    log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)
    np.save("mel_spec_cmvn.npy", log_mel_spec.numpy())
    print(f"After CMVN: mean={log_mel_spec.mean().item()}, std={log_mel_spec.std().item()}")

    features = log_mel_spec.numpy().astype(np.float32)
    return features, features_len


# Загрузка вокабуляра
def load_vocab(vocab_path: str) -> List[str]:
    """Load vocabulary from a file into a list of tokens.

    Parameters
    ----------
    vocab_path : str
        Path to the vocabulary file containing token-index pairs or single tokens.

    Returns
    -------
    List[str]
        List of vocabulary tokens, padded with "<pad>" up to 2561 entries.

    Notes
    -----
    - Supports files with either 'token index' pairs or single tokens per line.
    - Pads the vocabulary with "<pad>" if the length is less than 2561.
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
    while len(vocab) < 2561:
        vocab.append("<pad>")
    return vocab


# --- 2. Класс OnnxConformerRNNT ---
class OnnxConformerRNNT:
    def __init__(self, model_files: dict):
        # Инициализация ONNX-сессий для энкодера и декодера
        self._encoder = rt.InferenceSession(model_files["encoder"], providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(model_files["decoder_joint"], providers=["CPUExecutionProvider"])

        # Загрузка словаря (вокабуляра)
        self.vocab = load_vocab(model_files["vocab"])

        # Настройка индексов специальных токенов
        self._setup_token_indices()

        # Вывод информации о модели
        self._print_model_info()

    def _setup_token_indices(self):
        # Индексы специальных токенов (пустой, неизвестный, паддинг и т.д.)
        self._blank_idx = self.vocab.index("<blk>") if "<blk>" in self.vocab else 2560
        self._unk_idx = self.vocab.index("<unk>") if "<unk>" in self.vocab else 0
        self._blk_idx = self._blank_idx
        self._pad_idx = self.vocab.index("<pad>") if "<pad>" in self.vocab else 2560
        self._max_vocab_idx = len(self.vocab) - 1
        # Токены, которые нужно фильтровать после декодирования
        self._tokens_to_filter = {self._blank_idx, self._unk_idx, self._blk_idx, self._pad_idx}
        for i in range(self._max_vocab_idx + 1, len(self.vocab)):
            self._tokens_to_filter.add(i)

    def _print_model_info(self):
        # Печать информации о загруженном словаре
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")
        print(f"First 10 tokens: {self.vocab[:10]}")
        print(f"Last 10 tokens: {self.vocab[-10:]}")
        print(f"Blank index: {self._blank_idx} ('{self.vocab[self._blank_idx]}')")
        print(f"UNK index: {self._unk_idx} ('{self.vocab[self._unk_idx]}')")
        print(f"Max vocab index: {self._max_vocab_idx}")

    def _encode(self, features: np.ndarray, features_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Запуск энкодера ONNX и получение выходных признаков
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens}
        )
        return encoder_out, encoder_out_lens

    def _decode(self, prev_tokens: List[int], prev_state: Tuple[np.ndarray, np.ndarray], encoder_out: np.ndarray) -> \
    Tuple[np.ndarray, int, Tuple[np.ndarray, np.ndarray]]:
        # Запуск декодера (joint network) с текущими входами и состояниями
        outputs, state1, state2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out.astype(np.float32),
                "targets": np.array([[self._blank_idx if not prev_tokens else prev_tokens[-1]]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            }
        )
        return np.squeeze(outputs), -1, (state1, state2)

    def greedy_search(self, encoder_out: np.ndarray, encoder_out_len: np.ndarray) -> str:
        # Простая стратегия декодирования (поиск наиболее вероятных токенов)
        max_len = encoder_out.shape[2]
        print(f"Starting greedy decoding with {max_len} time frames")

        # Начальное состояние RNN
        state = (np.zeros((1, 1, 640), dtype=np.float32), np.zeros((1, 1, 640), dtype=np.float32))
        hyp = []  # гипотеза: список выбранных токенов

        for t in range(max_len):
            current_encoder_out = encoder_out[:, :, t:t + 1]  # текущий временной шаг
            logits, _, state = self._decode(hyp, state, current_encoder_out)
            logits = logits.copy()

            next_token = np.argmax(logits).item()
            # Добавление токена, если он не пустой и не повтор предыдущего
            if next_token != self._blank_idx and (not hyp or next_token != hyp[-1]):
                hyp.append(next_token)

            # Печать отладочной информации каждые 5 шагов
            if t % 5 == 0:
                token_str = self.vocab[next_token] if next_token < len(self.vocab) else 'OUT_OF_VOCAB'
                # print(f"Step {t}: token={next_token}('{token_str}')")

        return self._postprocess(hyp)

    def _postprocess(self, decoded_ids: List[int]) -> str:
        # Удаление специальных токенов и преобразование индексов в текст
        valid_tokens = [self.vocab[tok_id] for tok_id in decoded_ids if
                        tok_id not in self._tokens_to_filter and tok_id <= self._max_vocab_idx]
        text = "".join(valid_tokens).replace("▁", " ").strip()

        # Очистка текста от лишних пробелов и повторов символов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)

        # Удаление повторяющихся слов, если они длинные
        words = text.split()
        cleaned_words = []
        last_word = None
        for word in words:
            if word and (word.lower() != (last_word or "").lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word
        text = " ".join(cleaned_words).strip()
        return text


# --- 3. Инициализация модели и выполнение инференса ---
# Загрузка аудио с частотой дискретизации 16000 Гц
audio, sr = librosa.load("audio_files/20250404_174500.wav", sr=16000)

# Преобразование аудио в тензор и добавление размерности батча
audio_tensor = torch.tensor(audio).unsqueeze(0)
audio_len = torch.tensor([audio.shape[0]], dtype=torch.int32)

# Извлечение признаков из аудио (мел-спектрограммы и их длина)
features, features_len = preprocess_audio(audio_tensor, audio_len)
print(f"Features shape: {features.shape}")
print(f"Features length: {features_len}")

model_files = {
    "encoder": "onnx_models/encoder-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
    "decoder_joint": "onnx_models/decoder_joint-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
    "vocab": "onnx_models/vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt"
}

# Создание экземпляра модели
onnx_model = OnnxConformerRNNT(model_files)

# Получение выходов энкодера
encoder_out, encoder_out_len = onnx_model._encode(features, features_len)
print(f"Encoder output shape: {encoder_out.shape}")
print(f"Encoder output length: {encoder_out_len}")

# Декодирование аудио в текст
transcription = onnx_model.greedy_search(encoder_out, encoder_out_len)
print("Transcription:", transcription)