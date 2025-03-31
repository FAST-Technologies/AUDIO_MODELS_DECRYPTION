from typing import Tuple, Dict
import numpy as np
import hydra

from Constants import Constants
from .HelpFunction_NumPy import load_audio_prev
from .FeatureExtractor_NumPy import FeatureExtractor_V2_0
MY_CONSTANTS = Constants()

cfg = {
    "sample_rate": MY_CONSTANTS.SAMPLE_RATE,
    "features": MY_CONSTANTS.FEAT_IN,
    "model_name": "v2_ctc",
}

class GigaAM_V2_0:
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """
    def __init__(self,
                 cfg: Dict
    ) -> None:
        """
        Initialize the GigaAM_V2_0 model.

        Args:
            cfg (Dict): Configuration dictionary for the model.
        """
        self.cfg = cfg
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)
        self.preprocessor = FeatureExtractor_V2_0(
            sample_rate=cfg.get("sample_rate", MY_CONSTANTS.SAMPLE_RATE),
            features=cfg.get("features", MY_CONSTANTS.FEAT_IN)
        )


    def forward(self,
                features: np.ndarray,
                feature_lengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of the GigaAM model.

        Args:
            features (np.ndarray): Input features, shape [batch_size, time].
            feature_lengths (np.ndarray): Lengths of the input sequences, shape [batch_size].

        Returns:
            Tuple[np.ndarray, np.ndarray]: Encoded features and their lengths.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        return self.encoder(features, feature_lengths)

    def prepare_wav(self,
                    wav_file: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare a WAV file for processing.

        Args:
            wav_file (str): Path to the WAV file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Audio waveform and its length.
        """
        wav = load_audio_prev(wav_file)  # Используем новую функцию load_audio
        wav = wav[np.newaxis, :]  # Добавляем batch-размерность: [1, T]
        length = np.array([wav.shape[-1]], dtype=np.int64)
        return wav, length

    def embed_audio(self,
                    wav_file: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed an audio file into a feature representation.

        Args:
            wav_file (str): Path to the WAV file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Encoded features and their lengths.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len