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
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks.

    This class implements a self-supervised acoustic model for speech-related tasks,
    consisting of a preprocessor and an encoder.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing settings for the preprocessor and encoder.
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

        Parameters
        ----------
        features : Tensor
            Input features tensor, expected shape [batch, channels, seq_len].
        feature_lengths : Tensor
            Lengths of the input sequences, expected shape [batch].

        Returns
        -------
        Tensor
            Output tensor after passing through the preprocessor and encoder.

        Notes
        -----
        If the model is on a CUDA device, the forward pass uses FP16 precision via `torch.autocast`
        to optimize performance.
        """
        features, feature_lengths = self.preprocessor(features,
                                                      feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features,
                                feature_lengths)
        with torch.autocast(device_type=self._device.type,
                            dtype=torch.float16):
            return self.encoder(features,
                                feature_lengths)

    @property
    def _device(self) -> torch.device:
        """
        Get the device on which the model parameters reside.

        Returns
        -------
        torch.device
            The device (e.g., "cuda" or "cpu") where the model's parameters are located.
        """
        return next(self.parameters()).device

    @property
    def _dtype(self) -> torch.dtype:
        """
        Get the data type of the model parameters.

        Returns
        -------
        torch.dtype
            The data type (e.g., torch.float32) of the model's parameters.
        """
        return next(self.parameters()).dtype

    def prepare_wav(self,
                    wav_file: str
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto the correct device and converting its format.

        Parameters
        ----------
        wav_file : str
            Path to the audio file to be loaded.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple containing:
            - The audio tensor, shape [1, time], on the correct device and dtype.
            - The length tensor, shape [1], representing the number of samples in the audio.
        """
        wav = load_audio(wav_file)
        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1],
                            wav.shape[-1],
                            device=self._device)
        return wav, length

    def embed_audio(self,
                    wav_file: str
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.

        Parameters
        ----------
        wav_file : str
            Path to the audio file to process.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple containing:
            - The encoded audio representations tensor.
            - The lengths of the encoded sequences tensor.
        """
        wav, length = self.prepare_wav(wav_file)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self,
                dir_path: str = "."
    ) -> None:
        """
        Export the model's encoder to ONNX format and save it to the specified directory.

        Parameters
        ----------
        dir_path : str, optional
            Directory where the ONNX file will be saved. Defaults to the current directory (".").

        Returns
        -------
        None

        Notes
        -----
        The ONNX file will be named `<model_name>_encoder.onnx`, where `model_name` is taken from `self.cfg.model_name`.
        The encoder's dynamic axes are automatically determined using `self.encoder.dynamic_axes()`.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition.

    This class extends GigaAM to include components for automatic speech recognition (ASR),
    adding a head (e.g., CTCHead) and decoding mechanism (e.g., CTCGreedyDecoding).

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing settings for the preprocessor, encoder, head, and decoding.
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
        """
        Perform a forward pass for ONNX export, passing features through the encoder and head.

        Parameters
        ----------
        features : Tensor
            Input features tensor, expected shape [batch, channels, seq_len].
        feature_lengths : Tensor
            Lengths of the input sequences, expected shape [batch].

        Returns
        -------
        Tensor
            Output tensor after passing through the encoder and head, typically log probabilities.
        """
        return self.head(self.encoder(features, feature_lengths)[0])

    # После записи в onnx это нужно переписать на PyTorch?
    def to_onnx(self,
                dir_path: str = "."
    ) -> None:
        """
        Export the model to ONNX format and save it to the specified directory.

        Parameters
        ----------
        dir_path : str, optional
            Directory where the ONNX file will be saved. Defaults to the current directory (".").

        Returns
        -------
        None

        Notes
        -----
        - This method temporarily overrides the `forward` method to `forward_for_export` for ONNX export.
        - It is specifically designed for models using CTC (e.g., if "ctc" is in `self.cfg.model_name`).
        - The ONNX file will be named `<model_name>.onnx`, where `model_name` is taken from `self.cfg.model_name`.
        - Dynamic axes are specified to support variable batch sizes and sequence lengths.
        """
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