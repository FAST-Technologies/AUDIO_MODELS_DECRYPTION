# AUDIO_MODELS_DECRYPTION

## Stupakov Practice Task #2

Добро пожаловать в проект **AUDIO_MODELS_DECRYPTION**! Это практическая работа по созданию пакета для автоматического распознавания речи (Automatic Speech Recognition, ASR) с использованием ONNX. Основная цель — разобраться в устройстве современных ASR-систем и реализовать собственный инференс-пакет с минимальными зависимостями.

---

## О проекте

В рамках этой работы реализуются модели нейросетей для ASR:
- **Чисто на NumPy**: Инференс с использованием только NumPy и ONNX Runtime.
- **Чисто на PyTorch**: Инференс с использованием PyTorch для сравнения и отладки.

На данный момент завершена реализация модели **GigaAM Conformer v2 CTC**.

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
- Добавлено декодирование по лучу для получения транскрипций.
- Реализованы полностью модули графики на PyTorch и NumPy
- Были добавлены и изучены различные метрики оценивания параметров аудио (WER, CER, MER, WIL, WIP, BERT и так далее)
- Проведено исследование успешного декодирования на разных видах аудио для жадного декодирования и декодирования по лучу
- Реализован экспорт результатов метрики в JSON файл (но возможно потребует доработки)
---

### Что можно доделать
- Из всего на данный момент корректно не работает лишь только метрика PER

## Структура проекта

````
AUDIO_MODELS_DECRYPTION/
├── README.md              # Описание проекта
├── .gitignore             # Файлы, необходимые для пропуска
├── audio_files/           # Директория с тестируемыми аудио
├── Graphics/              # Директория с сохраненными графиками для каждого аудио
├── metrics_log.json       # JSON файл с результатами метрик
├── GigaAM Conformer v2 CTC (Raw 2).ipynb       # Первая версия реализации в виде блокнота
├── GigaAM Conformer v2 CTC (Raw 2) (Google).ipynb       # Реализация - пример исследования блокнота Google Colab
├── GigaAM Conformer v2 CTC (Raw 2) (Jupiter Notebook).ipynb       # Рабочая версия блокнота Jupiter Notebook для запуска на компьютере
├── gigaam_conformer_v2_ctc_(raw_2).py       # Файл основной реализации
├── Constants.py       # Файл с именованными константами
├── HelpFunctions.py       # Файл с расчётом метрик и реализацией основных методов декодирования (Greedy и Beam Search)
├── CTC_v2_NumPy/                   # Исходный код NumPy реализации
│   ├── CTC_model_NumPy.py # Головной модуль модели CTC на NumPy
│   ├── FeatureExtractor_NumPy.py # Препроцессор на NumPy
│   ├── GraphicsModule_NumPy.py      # Графический модуль NumPy реализации
│   └── HelpFunction_NumPy.py        # Вспомогательные функции NumPy реализации
├── CTC_v2_PyTorch/                   # Исходный код PyTorch
│   ├── CTC_model_PyTorch.py # Головной модуль модели CTC на PyTorch
│   ├── FeatureExtractor_PyTorch.py # Препроцессор на PyTorch
│   ├── GraphicsModule_PyTorch.py      # Графический модуль PyTorch реализации
│   └── HelpFunction_PyTorch.py        # Вспомогательные функции PyTorch реализации
├── GigaAM_to_ONNX/                   # Исходный код инференса Gigaam в ONNX
│   ├── GigaAM.py # Основная логика CTC модели в Gigaam
│   ├── HelpFunction.py # Вспомогательные функции (конвертация в ONNX и загрузка аудио)
│   └── LoadClass.py        # Класс загрузки модели в ONNX формат
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

Или установите всё явно:
```
pip install numpy==1.26.4 torch==2.2.1+cpu torchaudio==2.2.1+cpu torchvision==0.17.1+cpu hydra-core==1.3.2 librosa==0.11.0
matplotlib==3.10.1 pydub==0.25.1 omegaconf==2.3.0 onnxruntime==1.17.3 IPython tqdm==4.67.1 hf-xet==1.0.3 nltk==3.9.1 phonemizer==3.3.0 
pymorphy3==2.0.3 pymorphy3-dicts-ru==2.4.417150.4580142 scipy==1.15.2 torchmetrics==1.7.0 bert-score==0.3.13 jiwer==3.1.0 git+https://github.com/salute-developers/GigaAM.git -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
```
3. Скачайте ONNX-модель **v2_ctc.onnx** и поместите её в папку **models/**
4. При работе желательно поместить файл **ffmpeg.exe** в созданную директорию виртуальной среды **.venv/Scripts** + в идеале туда же положить файл **espeak.exe** для работы метрики PER. Так безопаснее!
5. Настроить виртуальную среду следующими командами (проект реализован на Python 3.12):
***CMD***
```
& "C:\Users\Vladimir\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
.venv/Scripts/activate
```

***Powershell***
```
"C:\Users\Vladimir\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
.venv/Scripts/activate
```

6. Проверить зависимости можно следующей командой:
```
pip list
```

7. После выполнения всего вышеперечисленного - запускаете либо обычный **.py** проект, либо блокнот **.ipynb** для **Jupiter Notebook**:
```
py '.\gigaam_conformer_v2_ctc_(raw_2).py'
```
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

* Завершить поддержку одной из дополнительных моделей (например, Vosk Zipformer).
* Добавить тесты и документацию для функций (практически реализована).
* Доделать вывод данных метрики в JSON формат
* Придумать способ заставить работать PER метрику

---

### MIT License 
### @2025 FAST_DEVELOPMENT

