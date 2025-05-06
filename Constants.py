from typing import Type
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Инициализация глобальных параметров
@dataclass(frozen=True)
class Constants:
      _URL_DIR: str = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
      AUDIO1_PATH: str = "E:/Прога (вся)/NeuralSpecter/AUDIO_MODELS_DECRYPTION/audio_files/-ot-vinta-bass-bossted.wav"
      AUDIO2_PATH: str =  'E:/Прога (вся)/NeuralSpecter/AUDIO_MODELS_DECRYPTION/audio_files/-blin-zachem-ya-syuda-prishel.wav'
      AUDIO3_PATH: str = 'E:/Прога (вся)/NeuralSpecter/AUDIO_MODELS_DECRYPTION/audio_files/-opa-kogo-to-hlopnuli.wav'
      MODEL_TYPE: str = "v2_ctc"
      MODEL_NAME: str = "v2_ctc.onnx"
      # MODEL_TYPE_RNNT: str = "v2_rnnt_Nemo_FastConformer_Hybrid"
      # MODEL_NAME_RNNT: str = "v2_rnnt_Nemo_FastConformer_Hybrid.onnx"
      MODEL_TYPE_RNNT: str = "v2_rnnt"
      MODEL_NAME_RNNT: str = "v2_rnnt.onnx"
      DOWNLOAD_CACHE: str = "~\.cache\gigaam"
      DOWNLOAD_CACHE_NEMO: str = "~/.cache/nemo"
      DIRNAME: str = ".\onnx_models"
      BLANK_IDX: int = 33  # Blank token index for CTC
      D_MODEL: int = 768 # Model dimension (e.g., for encoder)
      DTYPE: Type[np.dtype] = np.float32
      MAX_LETTERS_PER_FRAME: int = 3
      FEAT_IN: int = 64 # Number of mel filters for feature extraction
      PRED_HIDDEN: int = 320
      SAMPLE_RATE: int = 16000 # Hz
      LONGFORM_THRESHOLD: int = 25 * SAMPLE_RATE
      FLOAT_DIVISOR: float = 32768.0
      # Параметры
      N_FFT_TEST: int = SAMPLE_RATE // 40 # Number of FFT points for testing
      N_MELS_TEST: int = 64 # Number of mel filters for testing
      N_MFCC_TEST: int = 64 # Number of MFCC coefficients for testing
      N_LFCC_TEST: int = 64 # Number of LFCC coefficients for testing
      HOP_LENGTH_TEST: int = SAMPLE_RATE // 100 # Hop length for testing (16 ms at 16 kHz)
      WIN_LENGTH_TEST: int = SAMPLE_RATE // 40 # Window length for testing (32 ms at 16 kHz)
      MIN_LOG_TOLERANCE: float = 1e-9 # Min Tolerance for log function
      MAX_LOG_TOLERANCE: float = 1e9 # Max Tolerance for log function

# List букв русского языка
VOCAB = [
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о",
    "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"
]
print("VOCAB size:", len(VOCAB))

# Список моделей (отсюда нам нужна v2_ctc)
_MODEL_NAMES = [
    "ctc",
    "rnnt",
    "ssl",
    "emo",
    "v1_ctc",
    "v1_rnnt",
    "v1_ssl",
    "v2_ctc",
    "v2_rnnt",
    "v2_ssl",
]
print("_MODEL_NAMES size:", len(_MODEL_NAMES))

VALID_CMAPS = plt.colormaps()
VALID_INTERPOLATIONS = ['nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
VALID_LANGUAGES = ['EN', 'RU']
FORMATS_IMG = ['png', 'jpg', 'jpeg', 'pdf', 'svg']
