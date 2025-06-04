import os
import urllib
from pathlib import Path
from tqdm import tqdm
import tempfile
import json
import logging
from Constants import Constants

# Установите временную директорию на диск E:
os.environ['TEMP'] = 'E:/temp'
os.environ['TMP'] = 'E:/temp'
tempfile.tempdir = 'E:/temp'

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
        model = nemo_asr.models.ASRModel.from_pretrained(f"nvidia/{model_name}")
        logging.info(f"Модель {model_name} успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке модели из файла: {e}")
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

    # Определяем дефолтные значения из AudioToMelSpectrogramPreprocessor
    default_preprocessor_params = {
        "sample_rate": 16000,
        "window_size": 0.02,
        "window_stride": 0.01,
        "n_window_size": None,
        "n_window_stride": None,
        "window": "hann",
        "normalize": "per_feature",
        "n_fft": None,
        "preemph": 0.97,
        "features": 64,
        "lowfreq": 0,
        "highfreq": None,
        "log": True,
        "log_zero_guard_type": "add",
        "log_zero_guard_value": 2**-24,
        "dither": 1e-5,
        "pad_to": 16,
        "frame_splicing": 1,
        "exact_pad": False,
        "pad_value": 0,
        "mag_power": 2.0,
        "rng": None,
        "nb_augmentation_prob": 0.0,
        "nb_max_freq": 4000,
        "use_torchaudio": False,
        "mel_norm": "slaney",
        "stft_exact_pad": False,
        "stft_conv": False
    }

    # Извлекаем параметры препроцессора из конфигурации модели
    preprocessor_params = {}
    if hasattr(model, "cfg") and hasattr(model.cfg, "preprocessor"):
        print("\nПараметры препроцессора из model.cfg.preprocessor:")
        cfg_preprocessor = dict(model.cfg.preprocessor)
        # Список всех возможных параметров на основе дефолтов AudioToMelSpectrogramPreprocessor
        expected_params = list(default_preprocessor_params.keys())
        for param in expected_params:
            if param in cfg_preprocessor:
                value = cfg_preprocessor[param]
                preprocessor_params[param] = value
                print(f"{param}: {value}")
            else:
                # Используем дефолтное значение, если параметр не указан
                value = default_preprocessor_params[param]
                preprocessor_params[param] = value
                print(f"{param}: {value} (дефолтное значение)")
    else:
        print("\nПараметры препроцессора из model.cfg.preprocessor: (не доступны)")
        raise AttributeError("Model configuration or preprocessor config not found")

    # Сохраняем параметры препроцессора в JSON
    params_path = onnx_dir / f"preprocessor_params_{model_name}.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(preprocessor_params, f, indent=4, ensure_ascii=False)
    print(f"\nПараметры препроцессора сохранены в: {params_path}")

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