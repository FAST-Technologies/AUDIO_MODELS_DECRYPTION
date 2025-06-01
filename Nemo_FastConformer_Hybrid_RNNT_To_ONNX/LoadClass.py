import os
import urllib
from pathlib import Path
from tqdm import tqdm

import tempfile
import os

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
    _URL_DIR = "https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc/resolve/main/stt_ru_fastconformer_hybrid_large_pc.nemo" # Пример URL, уточните реальный источник

NEMO_CONSTANTS = NeMoConstants()

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
def load_nemo_model(
    model_name: str,
    device: str = "cpu",
    download_root: Optional[str] = None
) -> nemo_asr.models.ASRModel:
    if download_root is None:
        download_root = os.path.expanduser(NEMO_CONSTANTS.DOWNLOAD_CACHE)

    # Проверяем наличие модели
    model_path = os.path.join(download_root, f"{model_name}.nemo")
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
    model = nemo_asr.models.ASRModel.from_pretrained(f"nvidia/{model_name}")
    return model.to(device)

def export_nemo_to_onnx(
        model_name: str = NEMO_CONSTANTS.MODEL_TYPE,
        onnx_dir: str = NEMO_CONSTANTS.DIRNAME,
        device: str = "cpu",
        download_root: Optional[str] = None,
        decoder_types: Optional[List[str]] = None  # Список декодеров для экспорта
) -> None:
    # По умолчанию экспортируем оба декодера
    if decoder_types is None:
        decoder_types = ["ctc", "rnnt"]

    # Создаём директорию для ONNX
    onnx_dir = Path(onnx_dir)
    onnx_dir.mkdir(exist_ok=True, parents=True)

    # Загружаем модель один раз
    model = load_nemo_model(
        model_name=model_name,
        device=device,
        download_root=download_root
    )

    # Экспортируем модель для каждого декодера
    for decoder_type in decoder_types:
        print(f"Экспортируем модель с декодером: {decoder_type}")

        # Устанавливаем конфигурацию экспорта
        model.set_export_config({"decoder_type": decoder_type})

        # Формируем уникальный путь для ONNX-файла в зависимости от декодера
        onnx_path = str(Path(onnx_dir) / f"{model_name}_{decoder_type}.onnx")
        model.export(onnx_path)

        # Сохраняем словарь (один раз достаточно, так как он одинаков для всех декодеров)
        vocab_path = Path(onnx_dir) / f"vocab-{model_name}.txt"
        if not vocab_path.exists():  # Сохраняем только если файл ещё не существует
            with vocab_path.open("wt", encoding="utf-8") as f:
                for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
                    f.write(f"{token} {i}\n")
            print(f"Словарь сохранён в {vocab_path}")

        print(f"Модель '{model_name}' с декодером '{decoder_type}' успешно экспортирована в {onnx_path}")

# Пример использования
# if __name__ == "__main__":
#     export_nemo_to_onnx(
#         model_name="stt_ru_fastconformer_hybrid_large_pc",
#         onnx_dir="nemo-onnx",
#         device="cpu"
#     )