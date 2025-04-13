import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.jit import TracerWarning
from subprocess import CalledProcessError, run

from Constants import Constants
MY_CONSTANTS = Constants()

# Функция загрузки аудио
def load_audio(audio_path: str,
               sample_rate: int = MY_CONSTANTS.SAMPLE_RATE,
               return_format: str = "float"
) -> torch.Tensor:
    """
        Load an audio file using ffmpeg and return it as a torch.Tensor.

        Args:
            audio_path (str): Path to the audio file.
            sample_rate (int): Desired sample rate for the audio.
            return_format (str): Format of the returned tensor ("float" or "int16").

        Returns:
            torch.Tensor: Audio data as a tensor.
        """
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts"))
    ffmpeg_path = os.path.join(venv_path, "ffmpeg.exe")

    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(
            f"ffmpeg.exe not found at {ffmpeg_path}. Please ensure it is installed in .venv/Scripts.")
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-threads", "0",
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-"
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        print(f"Ошибка ffmpeg: {exc.stderr.decode()}")
        raise RuntimeError("Failed to load audio") from exc

    if return_format == "float":
        return torch.frombuffer(audio,
                                dtype=torch.int16).float() / 32768.0
    return torch.frombuffer(audio,
                            dtype=torch.int16)

# Функция экспорта в ONNX
def onnx_converter(
    model_name: str,
    module: torch.nn.Module,
    out_dir: str,
    inputs: Optional[Tuple[torch.Tensor]] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[dict] = None,
    opset_version: int = 17,
) -> None:
    if inputs is None:
        inputs = module.input_example() if hasattr(module, "input_example") else (torch.randn(1, 64, 100), torch.tensor([100], dtype=torch.long))
    if input_names is None:
        input_names = ["features", "feature_lengths"]
    if output_names is None:
        output_names = ["log_probs"]

    Path(out_dir).mkdir(exist_ok=True,
                        parents=True)
    out_path = str(Path(out_dir) / f"{model_name}.onnx")
    warning_types = [UserWarning, TracerWarning]
    with warnings.catch_warnings():
        for warning_type in warning_types:
            warnings.simplefilter("ignore",
                                  category=warning_type)
        torch.onnx.export(
            module.to(torch.float32),
            inputs,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    print(f"Successfully ported onnx {model_name} to {out_path}.")
