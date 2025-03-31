import numpy as np
import omegaconf

from .GigaAM_NumPy import GigaAM_V2_0
from .CTC_model_NumPy import CTCHead_V2_0, CTCGreedyDecoding_V2_0
from Constants import Constants, VOCAB
MY_CONSTANTS = Constants()

# Отсюда идёт numpy реализация
cfg = {
    "sample_rate": MY_CONSTANTS.SAMPLE_RATE,
    "features": MY_CONSTANTS.FEAT_IN,
    "model_name": "v2_ctc"
}

class GigaAMASR_V2_0(GigaAM_V2_0):
    """
    Giga Acoustic Model for Speech Recognition
    """
    # Снова Hydra utils
    def __init__(self,
                 cfg: omegaconf.DictConfig
    ) -> None:
        super().__init__(cfg)
        # self.head = hydra.utils.instantiate(self.cfg.head)  # CTCHead
        # self.decoding = hydra.utils.instantiate(self.cfg.decoding)  # CTCGreedyDecoding
        self.head = CTCHead_V2_0(
            feat_in=MY_CONSTANTS.FEAT_IN,
            num_classes=len(VOCAB) + 1  # +1 для blank
        )
        self.decoding = CTCGreedyDecoding_V2_0(vocabulary=VOCAB)

    def forward_for_export(self,
                           features: np.ndarray,
                           feature_lengths: np.ndarray
    ) -> np.ndarray:
        encoded, _ = self.encoder(features, feature_lengths)
        return self.head(encoded)