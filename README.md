# AUDIO_MODELS_DECRYPTION

## Stupakov Practice Task #2

Добро пожаловать в проект **AUDIO_MODELS_DECRYPTION**! Это практическая работа по созданию пакета для автоматического распознавания речи (Automatic Speech Recognition, ASR) с использованием ONNX. Основная цель — разобраться в устройстве современных ASR-систем и реализовать собственный инференс-пакет с минимальными зависимостями.

---

## О проекте

В рамках этой работы реализуются модели нейросетей для ASR:
- **Чисто на NumPy**: Инференс с использованием только NumPy и ONNX Runtime.
- **Чисто на PyTorch**: Инференс с использованием PyTorch для сравнения и отладки.

На данный момент практически завершена реализация модели **GigaAM Conformer v2 CTC**.

---

## Условие задания

### Цель
Разработать Python-пакет для инференса ASR-моделей через ONNX, изучив устройство современных систем распознавания речи.

### Требования
1. Разработка ведётся открыто на GitHub.
2. Зависимости пакета ограничены: только `numpy` и `onnxruntime`.
3. Для каждой модели должны быть реализованы:
   - **Препроцессинг**: Построение мел-спектрограмм.
   - **Жадное декодирование (Greedy Decoding)**: Простое декодирование предсказаний.
   - **Декодирование через поиск по лучу (Beam Search Decoding)**: Более сложный метод декодирования.

### Варианты моделей
Обязательная модель: **GigaAM Conformer v2 CTC**.  
Дополнительно одна из следующих (на выбор):
1. Vosk Zipformer
2. Nemo FastConformer Hybrid RNNT
3. GigaAM Conformer v2 RNNT
4. Nemo FastConformer Hybrid CTC

---

## Реализация GigaAM Conformer v2 CTC

### Условие для модели
1. Взять исходный код и веса GigaAM v2 CTC из репозитория ([ссылка](https://github.com/salute-developers/GigaAM)).
2. Сохранить модель в формате ONNX.
3. Реализовать инференс через `PyTorch`, используя фрагменты из `onnx_utils.py`([ссылка](https://github.com/salute-developers/GigaAM/blob/main/gigaam/onnx_utils.py)) и `preprocess.py`([ссылка](https://github.com/salute-developers/GigaAM/blob/main/gigaam/preprocess.py)) (загрузка аудио через `torchaudio.load`).
4. Изучить мел-спектрограммы и их построение ([ссылка](https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html)).
5. Реализовать инференс через `NumPy` с минимальными зависимостями.

### Текущий прогресс
- Реализован препроцессинг аудио (загрузка, ресемплинг, нормализация) на NumPy и PyTorch.
- Построены мел-спектрограммы через NumPy и сравнены с PyTorch (`torchaudio.transforms.MelSpectrogram`).
- Реализован инференс модели через ONNX с использованием `onnxruntime`.
- Добавлено жадное декодирование для получения транскрипций.

---

## Структура проекта

````
AUDIO_MODELS_DECRYPTION/
├── README.md              # Описание проекта
├── src/                   # Исходный код
│   ├── numpy_inference.py # Инференс на NumPy
│   ├── torch_inference.py # Инференс на PyTorch
│   ├── preprocess.py      # Препроцессинг аудио
│   └── decoding.py        # Декодирование (Greedy и Beam Search)
├── models/                # ONNX-модели
│   └── v2_ctc.onnx        # GigaAM Conformer v2 CTC
├── examples/              # Примеры использования
│   └── example.py         # Пример запуска инференса
├── requirements.txt       # Зависимости
└── LICENSE                # Лицензия (MIT)
````
---

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/AUDIO_MODELS_DECRYPTION.git
   cd AUDIO_MODELS_DECRYPTION
   ```
2. Установите зависимости:
    ```
    pip install -r requirements.txt
    ```
   Минимальные зависимости: **numpy**, **onnxruntime**. Для PyTorch-реализации дополнительно нужны **torch** и **torchaudio**.
3. Скачайте ONNX-модель **v2_ctc.onnx** и поместите её в папку **models/**

---

## Использование

#### Пример запуска инференса на NumPy:

```
from src.numpy_inference import run_inference
from src.preprocess import load_audio_numpy

audio_path = "path/to/audio.wav"
waveform = load_audio_numpy(audio_path)
transcription = run_inference(waveform, model_path="models/v2_ctc.onnx")
print("Транскрипция:", transcription)
```

#### Пример запуска инференса на PyTorch:

```
from src.torch_inference import run_inference
from src.preprocess import load_audio_torch

audio_path = "path/to/audio.wav"
waveform = load_audio_torch(audio_path)
transcription = run_inference(waveform, model_path="models/v2_ctc.onnx")
print("Транскрипция:", transcription)
```
---

## Лицензия

Код и веса модели **GigaAM Conformer v2 CTC** распространяются под лицензией .

Этот проект также использует **MIT License**.

---

## Что дальше?

* Добавить реализацию Beam Search декодирования.
* Завершить поддержку одной из дополнительных моделей (например, Vosk Zipformer).
* Оптимизировать производительность NumPy-реализации.
* Добавить тесты и документацию для функций.

---

### Инструкция
1. Скопируй этот текст в файл `README.md` в корне твоего проекта.
2. Замени `https://github.com/yourusername/AUDIO_MODELS_DECRYPTION.git` на реальный URL твоего репозитория.
3. Укажи правильную ссылку на репозиторий GigaAM v2 CTC вместо `[ссылка](https://github.com/path/to/repo)`.
4. Если хочешь, добавь больше деталей (например, ссылки на документацию или примеры вывода).

Теперь твой проект готов к публикации на GitHub с красивым и информативным `README`!

### MIT License 
### @2025 FAST_DEVELOPMENT

