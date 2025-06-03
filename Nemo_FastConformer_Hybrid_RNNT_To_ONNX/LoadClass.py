import os
import urllib
from pathlib import Path
from tqdm import tqdm
import tempfile
import json
import logging
from Constants import Constants, _MODEL_NAMES

# Установите временную директорию на диск E:
os.environ['TEMP'] = 'E:/temp'
os.environ['TMP'] = 'E:/temp'
tempfile.tempdir = 'E:/temp'
import torch
import nemo.collections.asr as nemo_asr
from typing import Tuple, Optional, List


# Аналог MY_CONSTANTS для NeMo
class NeMoConstants:
    DOWNLOAD_CACHE = "E:/Прога (вся)/NeuralSpecter/AUDIO_MODELS_DECRYPTION/cache"
    MODEL_TYPE = "stt_ru_fastconformer_hybrid_large_pc"
    DIRNAME = "onnx_models"
    _URL_DIR = "https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc/resolve/main/stt_ru_fastconformer_hybrid_large_pc.nemo"


NEMO_CONSTANTS = NeMoConstants()
MY_CONSTANTS = Constants()


# Аналог _download_file
def _download_file(file_url: str, file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with urllib.request.urlopen(file_url) as source, open(file_path, "wb") as output:
        with tqdm(
                total=int(source.info().get("Content-Length", 0)),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    return file_path


# Аналог _download_model для NeMo
def _download_model(model_name: str, download_root: str) -> Tuple[str, str]:
    model_url = f"{NEMO_CONSTANTS._URL_DIR}"
    model_path = os.path.join(download_root, f"{model_name}.nemo")
    return model_name, _download_file(model_url, model_path)


# Аналог load_model для NeMo
def load_nemo_model(model_name: str, device: str = "cpu",
                    download_root: Optional[str] = None) -> nemo_asr.models.ASRModel:
    if download_root is None:
        download_root = os.path.expanduser(NEMO_CONSTANTS.DOWNLOAD_CACHE)
    else:
        download_root = os.path.expanduser(MY_CONSTANTS.DOWNLOAD_CACHE_NEMO)

    # Проверяем наличие модели
    model_path = os.path.join(download_root, f"{model_name}.nemo")

    # Always check available models
    print("Проверка доступных моделей:")
    available_models = nemo_asr.models.ASRModel.list_available_models()
    print("Available models:", available_models)

    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"Файл найден: {model_path}, размер: {file_size} байт")
        if file_size == 0:
            print("Файл пустой, удаляем и попробуем скачать заново")
            os.remove(model_path)
    else:
        print(f"Файл не найден: {model_path}, будет выполнен новый запрос")
        _download_model(model_name, download_root)

    # Загружаем модель
    try:
        model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
        logging.info(f"Модель {model_name} успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке модели из файла: {e}")
        os.remove(model_path)
        raise
    return model.to(device)


def export_nemo_to_onnx(
        model_name: str = NEMO_CONSTANTS.MODEL_TYPE,
        onnx_dir: str = NEMO_CONSTANTS.DIRNAME,
        device: str = "cpu",
        download_root: Optional[str] = None,
        decoder_types: Optional[List[str]] = None
) -> Tuple[nemo_asr.models.ASRModel, str, str]:
    # По умолчанию экспортируем только RNNT декодер (как в вашем случае)
    if decoder_types is None:
        decoder_types = ["rnnt"]

    # Создаём директорию для ONNX
    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(exist_ok=True, parents=True)

    # Загружаем модель
    model = load_nemo_model(model_name=model_name, device=device, download_root=download_root)

    # Извлекаем параметры препроцессора
    preprocessor = model.preprocessor
    preprocessor_params = {}
    possible_params = [
        "sample_rate", "window_size", "window_stride", "window", "features", "n_fft",
        "frame_splicing", "dither", "pad_to", "normalize", "log", "log_zero_guard_value"
    ]
    for param in possible_params:
        if hasattr(preprocessor, param):
            preprocessor_params[param] = getattr(preprocessor, param)
    if hasattr(preprocessor, "cfg"):
        preprocessor_params["cfg"] = dict(preprocessor.cfg)

    # Сохраняем параметры препроцессора в JSON
    params_path = onnx_dir / f"preprocessor_params_{model_name}.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(preprocessor_params, f, indent=4, ensure_ascii=False)
    print(f"Параметры препроцессора сохранены в: {params_path}")

    # Извлекаем и сохраняем вокабуляр
    if hasattr(model, "tokenizer"):
        vocab = list(model.tokenizer.vocab)
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        blank_idx = vocab_dict.get("<blank>", len(vocab))  # Fallback if <blank> is not in vocab
        vocab.append("<blank>")  # Ensure blank token is included
        vocab_dict["<blank>"] = blank_idx
    else:
        raise AttributeError("Model does not have a tokenizer attribute to access vocabulary")

    vocab_path = onnx_dir / f"vocab-{model_name}.txt"
    if not vocab_path.exists():
        vocab_path = os.path.join(MY_CONSTANTS.DIRNAME, "vocab-stt_ru_fastconformer_hybrid_large_pc_RNNT.txt")
        os.makedirs(MY_CONSTANTS.DIRNAME, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, idx in vocab_dict.items():
                f.write(f"{token} {idx}\n")
        print(f"Вокабуляр сохранён в: {vocab_path}")

    # Экспортируем модель для каждого декодера
    for decoder_type in decoder_types:
        print(f"Экспортируем модель с декодером: {decoder_type}")
        model.set_export_config({"decoder_type": decoder_type})
        onnx_path = str(onnx_dir / f"{model_name}_{decoder_type}.onnx")
        model.export(onnx_path)
        print(f"Модель '{model_name}' с декодером '{decoder_type}' успешно экспортирована в {onnx_path}")

    logging.info(f"Экспорт модели в ONNX завершён. Вокабуляр доступен в {vocab_path}")
    return model, vocab_path, params_path