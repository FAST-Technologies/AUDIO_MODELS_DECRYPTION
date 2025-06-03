import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
import torchaudio
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch import Tensor
import nemo.collections.asr.modules.audio_preprocessing as nemo_preproc

# Загрузка словаря
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
BLK_IDX = VOCAB.index("<blk>") if "<blk>" in VOCAB else 1395  # Default to 1395 if not found
PAD_IDX = VOCAB.index("<pad>") if "<pad>" in VOCAB else 1024  # Adjust to 1024 as last index
MAX_VOCAB_IDX = len(VOCAB) - 1

# Проверка словаря
print(f"Vocab size: {len(VOCAB)}")
print(f"Sample vocab tokens: {VOCAB[:10]}")
print(f"BLANK_IDX: {BLANK_IDX}, BLK_IDX: {BLK_IDX}, PAD_IDX: {PAD_IDX}, MAX_VOCAB_IDX: {MAX_VOCAB_IDX}")

class RnntASRPyTorch(nn.Module):
    def __init__(self, encoder_path: str, decoder_joint_path: str, sample_rate: int = 16000,
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

        self._encoder_input_name = self._encoder.get_inputs()[0].name
        self._encoder_length_name = self._encoder.get_inputs()[1].name
        self._decoder_input_name = self._decoder_joint.get_inputs()[0].name
        self._decoder_prev_token_name = self._decoder_joint.get_inputs()[1].name
        self._decoder_state_name = self._decoder_joint.get_inputs()[2].name
        self._decoder_output_name = self._decoder_joint.get_outputs()[0].name
        self._decoder_state_out_name = self._decoder_joint.get_outputs()[1].name

        # Adjust audio preprocessing to match expected features
        self.audio_to_mel = nemo_preproc.AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            window="hann",
            normalize=False,
            n_fft=None,
            features=features  # Use 64 features as specified
        )

    def normalize_features(self, features: Tensor) -> Tensor:
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True)
        if std.eq(0).any():
            print("Warning: zero standard deviation in normalize_features, adding epsilon")
            std = std + 1e-6
        return (features - mean) / std

    def out_len(self, input_lengths: Tensor) -> Tensor:
        return input_lengths.div(160, rounding_mode="floor").add(1).long()

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

    def encode(self, features: np.ndarray, lengths: np.ndarray) -> np.ndarray:
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

    def decode_step(self, encoder_output: np.ndarray, prev_token: np.ndarray,
                    state: Tuple[np.ndarray, np.ndarray], t: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        state1, state2 = state
        target_length = np.array([1], dtype=np.int32)
        encoder_output_t = encoder_output[:, :, t:t + 1]
        if encoder_output_t.shape[2] == 0:
            dummy_logits = np.zeros((1, 1, 1, len(self.vocab)), dtype=np.float32)  # Adjusted to 1025
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
        return logits, (outputs[1], outputs[2])

    def decode_rnnt_greedy(self, encoder_output: np.ndarray, max_steps: int = 1000) -> str:
        if encoder_output.shape[0] != 1:
            raise ValueError(f"Expected batch_size=1, got {encoder_output.shape[0]}")

        batch_size, hidden_size, time_frames = encoder_output.shape
        state1 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        state2 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        start_token = self.vocab.index("▁") if "▁" in self.vocab else 1
        prev_token = np.array([[start_token]], dtype=np.int32)

        decoded_ids = []
        step = 0
        repeat_count = 0
        last_token = None

        while step < time_frames:
            logits, (state1, state2) = self.decode_step(
                encoder_output,
                prev_token,
                (state1, state2),
                step
            )

            temperature = 0.5
            probs = np.exp(logits[0, 0, 0] / temperature - np.max(logits[0, 0, 0] / temperature))
            probs /= np.sum(probs)
            print(f"Step {step}, Top 5 probs: {np.sort(probs)[-5:]}")
            print(f"Top 5 indices: {np.argsort(probs)[-5:]}")

            probs[self.blank_idx] = 0.0
            probs[self.blk_idx] = 0.0
            probs[self.pad_idx] = 0.0
            token_idx = np.argmax(probs)
            print(f"Selected token_idx: {token_idx}, token: {self.vocab[token_idx] if token_idx < len(self.vocab) else 'INVALID'}")

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
                    prev_token = np.array([[token_idx]], dtype=np.int32)
            else:
                prev_token = np.array([[self.blank_idx]], dtype=np.int32)

            step += 1

        result = "".join(self.vocab[tok] for tok in decoded_ids if
                         tok < len(self.vocab) and tok != self.blk_idx and tok != self.pad_idx)
        print(f"Decoded IDs: {decoded_ids}")
        print(f"Decoded tokens: {[self.vocab[tok] for tok in decoded_ids]}")
        return result

    def recognize(self, waveforms: np.ndarray) -> str:
        if waveforms.dtype != np.float32:
            waveforms = waveforms.astype(np.float32)
        if waveforms.ndim == 2 and waveforms.shape[0] > 1:
            waveforms = np.mean(waveforms, axis=0)
        elif waveforms.ndim == 2:
            waveforms = waveforms[0]
        waveforms = waveforms.flatten()

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
        return self.decode_rnnt_greedy(encoder_output)

if __name__ == "__main__":
    asr = RnntASRPyTorch("onnx_models/encoder-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
                         "onnx_models/decoder_joint-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx")
    waveform, sr = torchaudio.load("audio_files/20250404_174500.wav")
    print(f"Waveform shape: {waveform.shape}, sample rate: {sr}")
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        print(f"Resampled waveform shape: {waveform.shape}")

    plt.figure(figsize=(10, 4))
    plt.plot(waveform[0].numpy())
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    text = asr.recognize(waveform.numpy())
    print("Распознанный текст:", text)