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

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    sample_rate : int, optional
        Desired sample rate for the audio. Defaults to MY_CONSTANTS.SAMPLE_RATE.
    return_format : str, optional
        Format of the returned tensor. Must be either "float" or "int16". Defaults to "float".

    Returns
    -------
    torch.Tensor
        Audio data as a tensor. If `return_format` is "float", the tensor contains float values
        normalized to the range [-1.0, 1.0]. If `return_format` is "int16", the tensor contains
        int16 values in the range [-32768, 32767].

    Raises
    ------
    FileNotFoundError
        If `ffmpeg.exe` is not found in the expected path (.venv/Scripts).
    RuntimeError
        If ffmpeg fails to load the audio file.

    Notes
    -----
    This function uses ffmpeg to process the audio file, converting it to a mono PCM 16-bit format
    with the specified sample rate.
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
        audio = run(cmd,
                    capture_output=True,
                    check=True).stdout
    except CalledProcessError as exc:
        print(f"Ошибка ffmpeg: {exc.stderr.decode()}")
        raise RuntimeError("Failed to load audio") from exc

    if return_format == "float":
        return torch.frombuffer(audio,
                                dtype=torch.int16).float() / MY_CONSTANTS.FLOAT_DIVISOR
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
    """
    Convert a PyTorch model to ONNX format and save it to a file.

    Parameters
    ----------
    model_name : str
        Name of the model, used to generate the output file name.
    module : torch.nn.Module
        The PyTorch model to convert to ONNX.
    out_dir : str
        Directory where the ONNX file will be saved.
    inputs : Optional[Tuple[torch.Tensor]], optional
        Example inputs for tracing the model. If None, attempts to use `module.input_example()`
        if available, otherwise defaults to a tensor of shape (1, 64, 100) for features and
        a tensor of shape (1,) for feature lengths.
    input_names : Optional[List[str]], optional
        Names of the input nodes in the ONNX graph. Defaults to ["features", "feature_lengths"].
    output_names : Optional[List[str]], optional
        Names of the output nodes in the ONNX graph. Defaults to ["log probs"].
    dynamic_axes : Optional[dict], optional
        Dictionary specifying which axes of the inputs/outputs are dynamic. Defaults to None.
    opset_version : int, optional
        ONNX opset version to use for export. Defaults to 17.

    Returns
    -------
    None

    Notes
    -----
    - The model is converted to FP32 precision before export.
    - UserWarnings and TracerWarnings are suppressed during export to avoid cluttering the output.
    - The output file will be saved as `<out_dir>/<model_name>.onnx`.
    """
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
