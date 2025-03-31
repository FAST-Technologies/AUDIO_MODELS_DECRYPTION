import omegaconf
import hydra
from torch import Tensor

from .GigaAM import GigaAM
from .HelpFunction import onnx_converter

class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """
    def __init__(self,
                 cfg: omegaconf.DictConfig
    ) -> None:
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)  # CTCHead
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)  # CTCGreedyDecoding

    def forward_for_export(self,
                           features: Tensor,
                           feature_lengths: Tensor
    ) -> Tensor:
        return self.head(self.encoder(features, feature_lengths)[0])

    # После записи в onnx это нужно переписать на PyTorch?
    def to_onnx(self,
                dir_path: str = "."
    ) -> None:
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export
            onnx_converter(
                model_name=self.cfg.model_name,
                out_dir=dir_path,
                module=self,
                inputs=self.encoder.input_example(),
                input_names=["features", "feature_lengths"],
                output_names=["log_probs"],
                dynamic_axes={
                    "features": {0: "batch_size", 2: "seq_len"},
                    "feature_lengths": {0: "batch_size"},
                    "log_probs": {0: "batch_size", 1: "seq_len"},
                },
            )
            self.forward = saved_forward