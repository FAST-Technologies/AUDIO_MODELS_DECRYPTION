from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import omegaconf
import hydra

from Constants import Constants
from .HelpFunction import load_audio, onnx_converter
MY_CONSTANTS = Constants()

class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """
    def __init__(self,
                 cfg: omegaconf.DictConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
    ) -> Tensor:
        """
        Perform forward pass through the preprocessor and encoder.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def prepare_wav(self,
                    wav_file: str
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)
        return wav, length

    def embed_audio(self,
                    wav_file: str
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self,
                dir_path: str = "."
    ) -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
        )


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