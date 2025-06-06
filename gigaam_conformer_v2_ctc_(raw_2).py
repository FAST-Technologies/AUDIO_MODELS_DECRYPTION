# Импорт необходимых нам модулей и библиотек
import os
import numpy as np
import torch
import torchaudio
import logging
import omegaconf
import time
import onnxruntime as rt
import hydra
import json
from IPython.display import Audio
import librosa
import soundfile as sf
import torchaudio.functional as F
import torchaudio.transforms as T
from Nemo_FastConformer_Hybrid_RNNT_To_ONNX.LoadClass import export_nemo_to_onnx

# Проверяем версии основных пакетов
print("Torch version:", torch.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("Hydra version:", hydra.__version__)
print("Omegaconf version:", omegaconf.__version__)
print("Numpy version:", np.__version__)
print("Onnxruntime version:", rt.__version__)
print("Librosa version:", librosa.__version__)

import nemo
import matplotlib.pyplot as plt
import nemo.collections.asr as nemo_asr
print(nemo.__version__)
from nemo.collections.asr.models import EncDecRNNTModel
try:
    import nemo_asr
except ImportError:
    print("nemo_asr successfully removed")

# Проверка совместимости NumPy и PyTorch
# Проверяем версию NumPy (должна быть 1.26.4 - ниже второй)
try:
    a = torch.randn(5).cpu()
    b = a.numpy()
    print("NumPy is working fine with PyTorch!")
except Exception as e:
    print("NumPy is not working correctly:", e)

# Импорт модулей для сборки ONNX файла
from GigaAM_to_ONNX.LoadClass import load_model

# Импорт модулей для проверки работы нейронной сети на PyTorch
from CTC_v2_PyTorch.GigaamCtcASRPyTorch import GigaamCtcASRPyTorch
from HelpFunction_PyTorch import (load_audio_PyTorch,
                                  print_statistic_data_PyTorch,
                                  compute_lfcc_PyTorch)
from GraphicsModule_PyTorch import PyTorchGraphicsModule

# Импорт модулей для проверки работы нейронной сети на NumPy
from CTC_v2_NumPy.GigaamCtcASRNumPy import GigaamCtcASRNumPy
from HelpFunction_NumPy import (load_audio_new_V2_0,
                                load_audio_prev)

from Constants import Constants, VOCAB
MY_CONSTANTS = Constants()

# Объявление тестовых констант, они нам понадобятся при построении графиков
n_fft_test = MY_CONSTANTS.N_FFT_TEST # The number of FFT (Fast Fourier Transform) points used to compute the spectrogram
n_mels_test = MY_CONSTANTS.N_MELS_TEST # The number of mel filters (or mel bins) used in the mel spectrogram computation.
n_mfcc_test = MY_CONSTANTS.N_MFCC_TEST # The number of Mel-Frequency Cepstral Coefficients (MFCCs) to compute
n_lfcc_test = MY_CONSTANTS.N_LFCC_TEST # The number of Linear-Frequency Cepstral Coefficients (LFCCs) to compute
hop_length_test = MY_CONSTANTS.HOP_LENGTH_TEST # The number of audio samples between successive frames in the spectrogram computation
win_length_test = MY_CONSTANTS.WIN_LENGTH_TEST # The length of the window (in samples) used for each frame in the spectrogram computation
prev_tok = MY_CONSTANTS.BLANK_IDX
max_vocab_idx = len(VOCAB) - 1

# Тестовые списки для проверки декодирования по лучу
beam_widths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
length_penalties = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#@title 🌌Подготовка окружения
# Путь к аудиофайлу (прописаны в константах - MY_CONSTANTS)
source_path = "E:/Прога (вся)/NeuralSpecter/AUDIO_MODELS_DECRYPTION/audio_files/20250404_174500.wav"
ground_truth = "мне необходимо вам рассказать следующую историю о своей жизни чем четче я говорю тем лучший результат я получу"
# ground_truth = "потому что в самолете все зависит от винта"
if not os.path.exists(source_path):
    raise FileNotFoundError(f"Файл не найден по пути: {source_path}")

## 📝Создаем ONNX файл (для модели GigaAM Conformer v2 CTC)
# Получение модели из виртуальной среды
# Default cache directory
# cache_dir = os.path.expanduser(MY_CONSTANTS.DOWNLOAD_CACHE)
# model_path = os.path.join(cache_dir, f"{MY_CONSTANTS.MODEL_TYPE}.ckpt")
#
# # Если файл не обнаружен - скачаем его
# if os.path.exists(model_path):
#     file_size = os.path.getsize(model_path)
#     print(f"Файл найден: {model_path}, размер: {file_size} байт")
#     if file_size == 0:
#         print("Файл пустой, удаляем и попробуем скачать заново")
#         os.remove(model_path)
# else:
#     print(f"Файл не найден: {model_path}, будет выполнен новый запрос")
#
# model = load_model(model_name=MY_CONSTANTS.MODEL_TYPE,
#                    fp16_encoder=False,
#                    device="cpu")
# model.to_onnx(dir_path=MY_CONSTANTS.DIRNAME)

model, vocab_path, params_path = export_nemo_to_onnx(
    model_name=MY_CONSTANTS.MODEL_TYPE_RNNT,
    onnx_dir=MY_CONSTANTS.DIRNAME,
    device="cpu",
    decoder_types=["rnnt"]
)
logging.info(f"Экспорт модели в ONNX завершён. Вокабуляр доступен в {vocab_path}")

from RnntASPPyTorch_DIR.RnntASPPyTorch import RnntASRPyTorch
from RnntASPNumpy_DIR.RnntASPNumpy import RnntASRNumPy

"""## 🔦Пишем свою версию инференса на PyTorch"""

model_path = f"{MY_CONSTANTS.DIRNAME}/{MY_CONSTANTS.MODEL_NAME}"
PyTorch_Inference = GigaamCtcASRPyTorch(model_path)

audioMONO, original_sample_rateMONO = sf.read(source_path, dtype='float32', always_2d=True)
print("Размерность audioMONO (до преобразования):", audioMONO.shape)
# Преобразуем стерео в моно, усредняя каналы
if audioMONO.shape[1] > 1:
    print("Аудио СТЕРЕО -> МОНО (усреднение каналов)")
    audioMONO = np.mean(audioMONO, axis=1)  # [samples, channels] -> [samples]
print("Размерность audioMONO (после преобразования):", audioMONO.shape)

audioMONOOld = load_audio_PyTorch(audio_path=source_path,
                                  load_type="mono")
print("Размерность audioMONOOld:", audioMONOOld.shape)

# Загрузка аудио только через torchaudio
audioSTEREOTorch1, original_sample_rateTorch = torchaudio.load(source_path)

# Проверка формата аудио
if audioSTEREOTorch1.dim() == 1:
    print("Аудио МОНО, сконвертируем в СТЕРЕО формат, продублировав канал")
    audioSTEREOTorch1 = torch.stack([audioSTEREOTorch1, audioSTEREOTorch1])
elif audioSTEREOTorch1.dim() == 2:
    print("Аудио уже в СТЕРЕО формате")
else:
    raise ValueError("Неподдерживаемое количество каналов в аудио")

# Ресэмплинг, если частота не соответствует ожидаемой (16000 Гц)
if original_sample_rateTorch != 16000:
    print(f"Sample rate of the audio ({original_sample_rateTorch}) does not match expected (16000)")
    audioSTEREOTorch1 = torchaudio.functional.resample(audioSTEREOTorch1, original_sample_rateTorch, 16000)
    original_sample_rateTorch = 16000
    print(f"Ресэмплинг выполнен с {original_sample_rateTorch} Гц до 16000 Гц")

# Выбор первого канала для обработки
if audioSTEREOTorch1.shape[0] > 1:
    audioSTEREOTorch1 = audioSTEREOTorch1[0, :]  # Берем первый канал
    print("Обработано стерео-аудио, выбран первый канал")

# Преобразуем в numpy для дальнейшей работы
audioSTEREO = audioSTEREOTorch1.numpy().astype(np.float32)
print("Полученная размерность audioSTEREO:", audioSTEREO.shape)
print(f"Диапазон значений итогового аудио (audioSTEREO): [{audioSTEREO.min()} ; {audioSTEREO.max()}]")
# audioSTEREO, original_sample_rateSTEREO = sf.read(source_path,
#                                                   dtype='float32',
#                                                   always_2d=True)
#
# # Коррекция частоты дискретизации
# if original_sample_rateSTEREO != 16000:
#     audio_tensor = torch.from_numpy(audioSTEREO.T).float()  # Транспонируем для torchaudio
#     audioSTEREO = torchaudio.functional.resample(audio_tensor, original_sample_rateSTEREO, 16000).numpy().T
#     original_sample_rateSTEREO = 16000
#     print(f"Ресэмплинг выполнен с {original_sample_rateSTEREO} Гц до 16000 Гц")
#
# # Обработка стерео: выбор первого канала для теста
# if audioSTEREO.shape[1] > 1:
#     audioSTEREO = audioSTEREO[:, 0]  # Используем только первый канал
#     print("Обработано стерео-аудио, выбран первый канал")

print("Полученная размерность audioSTEREO:", audioSTEREO.shape)
print(f"Диапазон значений итогового аудио (audioSTEREO): [{audioSTEREO.min()} ; {audioSTEREO.max()}]")

# Загрузка с помощью torchaudio
audioSTEREOTorch, original_sample_rateTorch = torchaudio.load(source_path)
if audioSTEREOTorch.dim() == 1:
    print("Аудио МОНО, сконвертируем в СТЕРЕО формат, продублировав канал")
    audioSTEREOTorch = torch.stack([audioSTEREOTorch, audioSTEREOTorch])
elif audioSTEREOTorch.dim() == 2:
    print("Аудио уже в СТЕРЕО формате")
else:
    raise ValueError("Неподдерживаемое количество каналов в аудио")

# Resampling для audioSTEREO
# if original_sample_rateSTEREO != MY_CONSTANTS.SAMPLE_RATE:
#     print(f"Sample rate of the audio ({original_sample_rateSTEREO}) does not match expected ({MY_CONSTANTS.SAMPLE_RATE})")
#     audio_tensor = torch.from_numpy(audioSTEREO).float()
#     audio_tensor = audio_tensor.transpose(0, 1)  # [samples, channels] -> [channels, samples]
#     audio_tensor = F.resample(audio_tensor, original_sample_rateSTEREO, MY_CONSTANTS.SAMPLE_RATE)
#     audioSTEREO = audio_tensor.transpose(0, 1).numpy()  # [channels, samples] -> [samples, channels]
# print("Полученная размерность audioSTEREO:", audioSTEREO.shape)
# print(f"Диапазон значений итогового аудио (audioSTEREO): [{audioSTEREO.min()} ; {audioSTEREO.max()}]")

# Resampling для audioMONO
if original_sample_rateMONO != MY_CONSTANTS.SAMPLE_RATE:
    print(f"Sample rate of the audio ({original_sample_rateMONO}) does not match expected ({MY_CONSTANTS.SAMPLE_RATE})")
    audio_tensorMONO = torch.from_numpy(audioMONO).float()  # [samples]
    # Добавляем размерность канала: [samples] -> [1, samples]
    audio_tensorMONO = audio_tensorMONO.unsqueeze(0)
    audio_tensorMONO = F.resample(audio_tensorMONO, original_sample_rateMONO, MY_CONSTANTS.SAMPLE_RATE)
    # Убираем размерность канала: [1, samples] -> [samples]
    audioMONO = audio_tensorMONO.squeeze(0).numpy()
audioMONO = audioMONO / MY_CONSTANTS.FLOAT_DIVISOR
print("Полученная размерность audioMONO:", audioMONO.shape)
print(f"Диапазон значений итогового аудио (audioMONO): [{audioMONO.min()} ; {audioMONO.max()}]")

# Resampling для audioSTEREOTorch
if original_sample_rateTorch != MY_CONSTANTS.SAMPLE_RATE:
    audioSTEREOTorch = F.resample(audioSTEREOTorch, original_sample_rateTorch, MY_CONSTANTS.SAMPLE_RATE)
audioSTEREOTorch = audioSTEREOTorch / MY_CONSTANTS.FLOAT_DIVISOR
print("Полученная размерность audioSTEREOTorch:", audioSTEREOTorch.shape)
print(f"Диапазон значений итогового аудио (audioSTEREOTorch): [{audioSTEREOTorch.min()} ; {audioSTEREOTorch.max()}]")

spectrogram = T.Spectrogram(n_fft=n_fft_test)
griffin_lim = T.GriffinLim(n_fft=n_fft_test)
audioMONO_tensor = torch.from_numpy(audioMONO).float()  # [samples]
audioMONO_tensor = audioMONO_tensor.unsqueeze(0)  # [1, samples]
spec = spectrogram(audioMONO_tensor)
reconstructed_waveform = griffin_lim(spec)
# PyTorchGraphicsModule.plot_spectrogram_PyTorch(specgram=spec[0],
#                                                title="Изначальная спектрограмма",
#                                                xlabel="Индекс фрейма",
#                                                ylabel="Частотный диапазон",
#                                                colorbar_label="Цветовой градиент спектрограммы",
#                                                grid_flag=False)
#
# PyTorchGraphicsModule.plot_waveform_PyTorch(waveform=audioMONO,
#                                             sr=MY_CONSTANTS.SAMPLE_RATE,
#                                             title="Оригинальная волновая форма (WaveForm) (audioMONO_PyTorch)",
#                                             xlabel="Время (секунды) [s]",
#                                             ylabel="Амплитуда",
#                                             flag="CW",
#                                             grid_flag=False)
# PyTorchGraphicsModule.plot_waveform_PyTorch(waveform=audioSTEREOTorch,
#                                             sr=MY_CONSTANTS.SAMPLE_RATE,
#                                             title="Оригинальная волновая форма (WaveForm) (audioSTEREO_PyTorch)",
#                                             xlabel="Время (секунды) [s]",
#                                             ylabel="Амплитуда",
#                                             flag="CW",
#                                             grid_flag=False)
# PyTorchGraphicsModule.plot_waveform_PyTorch(waveform=reconstructed_waveform,
#                                             sr=MY_CONSTANTS.SAMPLE_RATE,
#                                             title="Реконструированная волновая форма (WaveForm) (reconstructed_waveform)",
#                                             xlabel="Время (секунды) [s]",
#                                             ylabel="Амплитуда",
#                                             flag="RW",
#                                             grid_flag=False)
print("Выводим информацию по AUDIO: ")
# Audio(audioMONO.numpy(),
#       rate=MY_CONSTANTS.SAMPLE_RATE)
Audio(audioSTEREOTorch.numpy(),
      rate=MY_CONSTANTS.SAMPLE_RATE)
Audio(source_path,
      rate=MY_CONSTANTS.SAMPLE_RATE)

mfcc_transform = T.MFCC(
    sample_rate=MY_CONSTANTS.SAMPLE_RATE,
    n_mfcc=n_mfcc_test,
    melkwargs={
        "n_fft": n_fft_test,
        "n_mels": n_mels_test,
        "hop_length": hop_length_test,
        "mel_scale": "htk",
    },
)
mfcc = mfcc_transform(audioMONO_tensor)
# PyTorchGraphicsModule.plot_spectrogram_PyTorch(specgram=mfcc[0],
#                                                title="MFCC (PyTorch)",
#                                                xlabel="Индекс фрейма",
#                                                ylabel="Частотный диапазон",
#                                                colorbar_label="Цветовой градиент спектрограммы",
#                                                type="MFCC",
#                                                util_type="TorchAudio",
#                                                grid_flag=False)
lfcc_transform = T.LFCC(
    sample_rate=MY_CONSTANTS.SAMPLE_RATE,
    n_lfcc=n_lfcc_test,
    speckwargs={
        "n_fft": n_fft_test,
        "hop_length": hop_length_test,
        "win_length": win_length_test,
    },
)
lfcc = lfcc_transform(audioMONO_tensor)
# PyTorchGraphicsModule.plot_spectrogram_PyTorch(specgram=lfcc[0],
#                                                title="LFCC (PyTorch)",
#                                                xlabel="Индекс фрейма",
#                                                ylabel="Частотный диапазон",
#                                                colorbar_label="Цветовой градиент спектрограммы",
#                                                type="LFCC",
#                                                util_type="TorchAudio",
#                                                grid_flag=False)

# Вынести в отдельную функцию расчёта MFCC (Librosa)
melspec = librosa.feature.melspectrogram(
    y=audioMONO_tensor.numpy().squeeze(),
    sr=MY_CONSTANTS.SAMPLE_RATE,
    n_fft=n_fft_test,
    win_length=win_length_test,
    hop_length=hop_length_test,
    n_mels=n_mels_test,
    htk=True,
    fmin=0.0,
    fmax=MY_CONSTANTS.SAMPLE_RATE / 2.0,
    norm=None
)

mfcc_librosa = librosa.feature.mfcc(
    S=librosa.core.spectrum.power_to_db(melspec),
    n_mfcc=n_mfcc_test,
    dct_type=2,
    norm="ortho",
)
# PyTorchGraphicsModule.plot_spectrogram_PyTorch(specgram=mfcc_librosa,
#                                                title="MFCC (Librosa)",
#                                                xlabel="Индекс фрейма",
#                                                ylabel="Частотный диапазон",
#                                                colorbar_label="Цветовой градиент спектрограммы",
#                                                type="MFCC",
#                                                util_type="Librosa",
#                                                grid_flag=False)

mse_mfcc = torch.square(torch.from_numpy(mfcc_librosa) - mfcc).mean().item()
print(f"MSE между PyTorch и Librosa MFCC: {mse_mfcc}")

# Вынести в отдельную функцию расчёта LFCC (Librosa)
lfcc_librosa = compute_lfcc_PyTorch(
    y=audioMONO_tensor.numpy().squeeze(),
    sr=MY_CONSTANTS.SAMPLE_RATE,
    n_fft=n_fft_test,
    win_length=win_length_test,
    hop_length=hop_length_test,
    n_filters=n_lfcc_test,
    n_lfcc=n_lfcc_test,
    fmin=0.0,
    fmax=MY_CONSTANTS.SAMPLE_RATE / 2.0
)
# PyTorchGraphicsModule.plot_spectrogram_PyTorch(specgram=lfcc_librosa,
#                                                title="LFCC (Librosa)",
#                                                xlabel="Индекс фрейма",
#                                                ylabel="Частотный диапазон",
#                                                colorbar_label="Цветовой градиент спектрограммы",
#                                                type="LFCC",
#                                                util_type="Librosa",
#                                                grid_flag=False)

mse_lfcc = torch.square(torch.from_numpy(lfcc_librosa) - lfcc).mean().item()
print(f"MSE между PyTorch и Librosa LFCC: {mse_lfcc}")

pitch_PyTorch = F.detect_pitch_frequency(audioMONO_tensor, MY_CONSTANTS.SAMPLE_RATE)
print(f"Pitch: {pitch_PyTorch}")
# PyTorchGraphicsModule.plot_pitch_PyTorch(waveform=audioMONO_tensor,
#                                          sr=MY_CONSTANTS.SAMPLE_RATE,
#                                          pitch=pitch_PyTorch,
#                                          title="График Питча",
#                                          language_type="RU",
#                                          grid_flag=False)


audioONNX = load_audio_prev(source_path)
print(f"Размерность аудио (audioONNX): {audioONNX.shape}")
audioONNX_tensor = torch.from_numpy(audioONNX)
audioONNX_tensor = audioONNX_tensor.unsqueeze(0)  # [225963] -> [1, 225963]
audioONNX_tensor = audioONNX_tensor.float()
print(f"Размерность аудио (audioONNX после конвертации: {audioONNX_tensor.shape}")

audioONNX_stereo = torch.stack([audioONNX_tensor[0], audioONNX_tensor[0]])

if audioONNX_tensor.shape[1] != audioMONO_tensor.shape[1]:
    raise ValueError(f"Длина аудио не совпадает: {audioONNX_tensor.shape[1]} != {audioMONO_tensor.shape[1]}")

# Сравнение
print(f"Максимальное расхождение для первого примера (моно): {torch.max(torch.abs(audioONNX_tensor - audioMONO_tensor)).item()} единицы")
print(f"Максимальное расхождение для второго примера (стерео): {torch.max(torch.abs(audioONNX_stereo - audioSTEREOTorch)).item()} единицы")

mse_mono = torch.mean(torch.square(audioONNX_tensor - audioMONO_tensor)).item()
mse_stereo = torch.mean(torch.square(audioONNX_stereo - audioSTEREOTorch)).item()
print(f"MSE между audioONNX_MONO и audioMONO (PyTorch): {mse_mono}")
print(f"MSE между audioONNX_STEREO и audioSTEREO (PyTorch): {mse_stereo}")

diff = audioONNX_tensor - audioMONO_tensor
# PyTorchGraphicsModule.plot_waveform_PyTorch(waveform=diff,
#                                             sr=MY_CONSTANTS.SAMPLE_RATE,
#                                             title="Разница между audioONNX и audioMONO",
#                                             xlabel="Время (секунды) [s]",
#                                             ylabel="Амплитуда",
#                                             flag="CW",
#                                             grid_flag=False)

mel_filters_PyTorch = PyTorch_Inference.mel_fb
print(f"Тип: {type(mel_filters_PyTorch)}")

# PyTorchGraphicsModule.plot_fbank_PyTorch(mel_filters=mel_filters_PyTorch,
#                                          title="Mel Filter Bank - TorchAudio (Финальный Результат)",
#                                          xlabel = "Частота (Hz)",
#                                          ylabel = "Индекс Mel Фильтра",
#                                          colorbar_label = "Веса фильтра (цветовой градиент)",
#                                          cmap='viridis',
#                                          interpolation="bicubic",
#                                          grid_flag=False)

featuresPyMONO, lengthsPyMONO = PyTorch_Inference(audioMONO_tensor.unsqueeze(0),
                                                    torch.tensor([audioMONO_tensor.shape[-1]]))
featuresPyMONO = featuresPyMONO.detach().cpu().numpy().astype(np.float32)
lengthsPyMONO = lengthsPyMONO.detach().cpu().numpy().astype(np.int64)

featuresPySTEREO, lengthsPySTEREO = PyTorch_Inference(audioSTEREOTorch.unsqueeze(0),
                                                        torch.tensor([audioSTEREOTorch.shape[-1]]))
featuresPySTEREO = featuresPySTEREO.detach().cpu().numpy().astype(np.float32)
lengthsPySTEREO = lengthsPySTEREO.detach().cpu().numpy().astype(np.int64)

# # Предоставляем статистические результаты по каждому из каналов
print("Результаты для МОНО канала: ")
print_statistic_data_PyTorch(features=featuresPyMONO)

print("Результаты для СТЕРЕО канала: ")
print_statistic_data_PyTorch(features=featuresPySTEREO)
#
# # Отображение первой фичи в батче (если features.shape = [1, 64, T])
# # График для моно-канала
# PyTorchGraphicsModule.mono_graph_PyTorch(features=featuresPyMONO,
#                                          title='Спектрограмма фич (МОНО)',
#                                          xlabel='Временные кадры',
#                                          ylabel='Фичи',
#                                          colorbar_label='Значение фичи',
#                                          grid_flag=False)

# # Первый график для стерео-канала (с использованием Subplots)
# PyTorchGraphicsModule.stereo_subplots_graph_PyTorch(features=featuresPySTEREO,
#                                                      suptitle='Спектрограмма фич (СТЕРЕО/Subplots)',
#                                                      colorbar_label='Значение фичи',
#                                                      language_type="RU",
#                                                      grid_flag=False)
#
# # Второй график для стерео-канала (с использованием GridSpec)
# PyTorchGraphicsModule.stereo_gridspec_graph_PyTorch(features=featuresPySTEREO,
#                                                     suptitle='Спектрограмма фич (СТЕРЕО/GridSpec)',
#                                                     colorbar_label='Значение фичи',
#                                                     language_type="RU",
#                                                     grid_flag=False)

# transcriptionGD = PyTorch_Inference.recognize(audioSTEREO,
#                                               decode_flag="GD",
#                                               ground_truth=ground_truth)
# global transcriptionBS
# for beam_width in beam_widths:
#     for lp in length_penalties:
#         print(f"\nTesting beam_width={beam_width}, length_penalty={lp}")
#         time.sleep(5)
#         transcriptionBS = PyTorch_Inference.recognize(audioSTEREO,
#                                                       decode_flag="BS",
#                                                       beam_width=beam_width,
#                                                       length_penalty=lp,
#                                                       ground_truth=ground_truth)
#
# print("Транскрипция жадного декодирования (PyTorch):", transcriptionGD)
# print("Транскрипция декодирования по лучу (PyTorch):", transcriptionBS)

"""![link](https://drive.google.com/drive/MyDrive/n_fft.png)"""

"""![n_fft.png](data:image/png;base64,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)

"""## 🔢 Переписываем текущую нейронную сеть с PyTorch на NumPy"""


# Загрузка предыдущей записи
audio_prev = load_audio_prev(source_path)
print(f"Размерность аудио (audio_prev): {audio_prev.shape}")
# audio_prev = audio_prev[np.newaxis, :]  # [225963] -> [1, 225963]
audio_prev = audio_prev.astype(np.float32)

# Подготовка входных данных - загрузка Моно записи
audio_NumPy_MONO = load_audio_new_V2_0(audio_path=source_path,
                                       load_type="mono")
print("Размерность аудио (audio_NumPy_MONO):", audio_NumPy_MONO.shape)
audio_NumPy_MONO = audio_NumPy_MONO.astype(np.float32)

preprocessor_NumPy = GigaamCtcASRNumPy(model_path)

# audio_with_batch = audio_prev[np.newaxis, :]
# length_prev = np.array([audio_prev.shape[-1]],
#                        dtype=np.int64)
# featuresPrev, lengthsPrev = preprocessor_NumPy(audio_with_batch, length_prev)
# featuresPrev = featuresPrev.astype(np.float32)
# lengthsPrev = lengthsPrev.astype(np.int64)

# Подготовка входных данных - загрузка Стерео записи
audio_NumPy_STEREO = load_audio_new_V2_0(audio_path=source_path,
                                         load_type="stereo")
print("Размерность аудио (audio_NumPy_STEREO):", audio_NumPy_STEREO.shape)
print(f"Диапазон значений итогового аудио: [{audio_NumPy_STEREO.min()} ; {audio_NumPy_STEREO.max()}]")

# Сравнение
# print(f"Максимальное расхождение для первого примера (моно): {np.max(np.abs(audio_prev - audio_NumPy_MONO))} единиц")
# print(f"Максимальное расхождение для второго примера (стерео): {np.max(np.abs(audio_prev - audio_NumPy_STEREO))} единиц")

# Получаем результаты для моно и стерео канала
# spectrogram_NumPy = T.Spectrogram(n_fft=n_fft_test)
# griffin_lim_NumPy = T.GriffinLim(n_fft=n_fft_test)
# spec_NumPy = spectrogram_NumPy(torch.from_numpy(audio_NumPy_MONO))  # Конверсия в PyTorch
# reconstructed_waveform_NumPy = griffin_lim_NumPy(spec_NumPy).numpy()  # Обратно в NumPy
#
# # Строим Спектрограмму
# NumpyGraphicsModule.plot_spectrogram(specgram=spec_NumPy[0].numpy(),
#                                      title="Изначальная спектрограмма",
#                                      xlabel="Индекс фрейма",
#                                      ylabel="Частотный диапазон",
#                                      colorbar_label="Цветовой градиент спектрограммы",
#                                      grid_flag=False)
# # Строим WaveForm
# NumpyGraphicsModule.plot_waveform(waveform=audio_NumPy_MONO,
#                                  sr=MY_CONSTANTS.SAMPLE_RATE,
#                                  title="Оригинальная волновая форма (WaveForm) (audioMONO_NumPy)",
#                                  xlabel="Время (секунды) [s]",
#                                  ylabel="Амплитуда",
#                                  flag="CW",
#                                  grid_flag=False)
#
# NumpyGraphicsModule.plot_waveform(waveform=audio_NumPy_STEREO,
#                                   sr=MY_CONSTANTS.SAMPLE_RATE,
#                                   title="Оригинальная волновая форма (WaveForm) (audioSTEREO_NumPy)",
#                                   xlabel="Время (секунды) [s]",
#                                   ylabel="Амплитуда",
#                                   flag="CW",
#                                   grid_flag=False)
#
# NumpyGraphicsModule.plot_waveform(waveform=reconstructed_waveform_NumPy,
#                                   sr=MY_CONSTANTS.SAMPLE_RATE,
#                                  title="Реконструированная волновая форма (WaveForm) (reconstructed_waveform_NumPy)",
#                                  xlabel="Время (секунды) [s]",
#                                  ylabel="Амплитуда",
#                                  flag="RW",
#                                  grid_flag=False)
#
# # MFCC и LFCC с Librosa
# melspec = librosa.feature.melspectrogram(
#     y=audio_NumPy_MONO[0],  # Берем первый канал
#     sr=MY_CONSTANTS.SAMPLE_RATE,
#     n_fft=n_fft_test,
#     win_length=win_length_test,
#     hop_length=hop_length_test,
#     n_mels=n_mels_test,
#     htk=True,
#     fmin=0.0,
#     fmax=MY_CONSTANTS.SAMPLE_RATE / 2.0,
#     norm=None
# )
# mfcc_librosa = librosa.feature.mfcc(
#     S=librosa.core.spectrum.power_to_db(melspec),
#     n_mfcc=n_mfcc_test,
#     dct_type=2,
#     norm="ortho",
# )
# NumpyGraphicsModule.plot_spectrogram(specgram=mfcc_librosa,
#                                      title="MFCC (Librosa)",
#                                      xlabel="Индекс фрейма",
#                                      ylabel="Частотный диапазон",
#                                      colorbar_label="Цветовой градиент спектрограммы",
#                                      type="MFCC",
#                                      util_type="Librosa",
#                                      grid_flag=False)
#
# lfcc_librosa = compute_lfcc_V2_0(
#     y=audio_NumPy_MONO[0],
#     sr=MY_CONSTANTS.SAMPLE_RATE,
#     n_fft=n_fft_test,
#     win_length=win_length_test,
#     hop_length=hop_length_test,
#     n_filters=n_lfcc_test,
#     n_lfcc=n_lfcc_test,
#     fmin=0.0,
#     fmax=MY_CONSTANTS.SAMPLE_RATE / 2.0
# )
# NumpyGraphicsModule.plot_spectrogram(specgram=lfcc_librosa,
#                                      title="LFCC (Librosa)",
#                                      xlabel="Индекс фрейма",
#                                      ylabel="Частотный диапазон",
#                                      colorbar_label="Цветовой градиент спектрограммы",
#                                      type="LFCC",
#                                      util_type="Librosa",
#                                      grid_flag=False)

# pitch_numpy = F.detect_pitch_frequency(torch.from_numpy(audio_NumPy_MONO), MY_CONSTANTS.SAMPLE_RATE).numpy()
# NumpyGraphicsModule.plot_pitch(waveform=audio_NumPy_MONO,
#                                sr=MY_CONSTANTS.SAMPLE_RATE,
#                                pitch=pitch_numpy,
#                                title="График Питча",
#                                language_type="RU",
#                                grid_flag=False)

# mel_filters_NumPy = preprocessor_NumPy.mel_fb
# print(f"Тип: {type(mel_filters_NumPy)}")
# NumpyGraphicsModule.plot_fbank(mel_filters=mel_filters_NumPy,
#                                title="Mel Filter Bank - NumPy (Финальный Результат)",
#                                xlabel = "Частота (Hz)",
#                                ylabel = "Индекс Mel Фильтра",
#                                colorbar_label = "Веса фильтра (цветовой градиент)",
#                                cmap='viridis',
#                                interpolation="bicubic",
#                                grid_flag=False)

# # Получаем фичи и длину фич для Моно записи
# features_NumPy_MONO, lengths_NumPy_MONO = preprocessor_NumPy(audio_NumPy_MONO[np.newaxis, :],
#                                                              np.array([audio_NumPy_MONO.shape[-1]],
#                                                              dtype = np.int64))
# features_NumPy_MONO = features_NumPy_MONO.astype(np.float32)
# lengths_NumPy_MONO = lengths_NumPy_MONO.astype(np.int64)
#
# # Получаем фичи и длину фич для Стерео записи
# features_NumPy_STEREO, lengths_NumPy_STEREO = preprocessor_NumPy(audio_NumPy_STEREO[np.newaxis, :],
#                                                                  np.array([audio_NumPy_STEREO.shape[-1]],
#                                                                  dtype = np.int64))
# features_NumPy_STEREO = features_NumPy_STEREO.astype(np.float32)
# lengths_NumPy_STEREO = lengths_NumPy_STEREO.astype(np.int64)
#
# # Статистика (featuresPrev)
# print_statistic_data_V2_0(features=featuresPrev)
#
# # Статистика (features_NumPy_MONO)
# print_statistic_data_V2_0(features=features_NumPy_MONO)
#
# # Статистика (features_NumPy_STEREO)
# print_statistic_data_V2_0(features=features_NumPy_STEREO)
#
# # Предоставляем статистические результаты по каждому из каналов
# print("Результаты для МОНО канала (NumPy):")
# print_statistic_data_V2_0(features=features_NumPy_MONO)
#
# print("Результаты для СТЕРЕО канала (NumPy):")
# print_statistic_data_V2_0(features=features_NumPy_STEREO)

# Отображение первой фичи в батче (если features.shape = [1, 64, T])
# График для моно-канала
# NumpyGraphicsModule.mono_graph_V2_0(features=features_NumPy_MONO,
#                                     title='Спектрограмма фич (МОНО)',
#                                     xlabel='Временные кадры',
#                                     ylabel='Фичи',
#                                     colorbar_label='Значение фичи',
#                                     grid_flag=False)
#
# # Первый график для стерео-канала (с использованием Subplots)
# NumpyGraphicsModule.stereo_subplots_graph_V2_0(features=features_NumPy_STEREO,
#                                                 suptitle='Спектрограмма фич (СТЕРЕО/Subplots)',
#                                                 colorbar_label='Значение фичи',
#                                                 language_type="RU",
#                                                 grid_flag=False)
#
# # Второй график для стерео-канала (с использованием GridSpec)
# NumpyGraphicsModule.stereo_gridspec_graph_V2_0(features=features_NumPy_STEREO,
#                                                suptitle='Спектрограмма фич (СТЕРЕО/GridSpec)',
#                                                colorbar_label='Значение фичи',
#                                                language_type="RU",
#                                                grid_flag=False)

# Инференс NumPy
# inputs_NumPy_MONO = {"features": features_NumPy_MONO, "feature_lengths": lengths_NumPy_MONO}
# log_probs_NumPy_MONO = session_NumPy.run(["log_probs"], inputs_NumPy_MONO)[0]
#
# # Проверка на nan и inf
# if np.any(np.isnan(log_probs_NumPy_MONO)):
#     raise ValueError("log_probs_NumPy_MONO contains NaN values")
# if np.any(np.isinf(log_probs_NumPy_MONO)):
#     raise ValueError("log_probs_NumPy_MONO contains Inf values")
#
# # Убедимся, что log_probs_NumPy_MONO имеет форму [batch_size, seq_len, num_classes]
# if len(log_probs_NumPy_MONO.shape) == 2:  # [seq_len, num_classes]
#     log_probs_NumPy_MONO = log_probs_NumPy_MONO[np.newaxis, :]  # [1, seq_len, num_classes]
# elif len(log_probs_NumPy_MONO.shape) != 3:
#     raise ValueError(f"Unexpected shape for log_probs_NumPy_MONO: {log_probs_NumPy_MONO.shape}")

# Декодирование (жадное) для NumPy реализации
audio_prev = audio_prev.astype(np.float32)
print(f"Форма audio_prev перед передачей в recognize: {audio_prev.shape}")
transcriptionGD_NumPy, metricsGD_NumPy = preprocessor_NumPy.recognize(
    waveforms=audio_prev,
    decode_flag="GD",
    ground_truth=ground_truth
)

# transcriptionBS_NumPy, metricsBS_NumPy = None, None
# for beam_width in beam_widths:
#     for lp in length_penalties:
#         print(f"\nTesting beam_width={beam_width}, length_penalty={lp}")
#         time.sleep(5)
#         transcriptionBS_NumPy, metricsBS_NumPy = preprocessor_NumPy.recognize(
#             waveforms=audio_prev,
#             decode_flag="BS",
#             beam_width=beam_width,
#             length_penalty=lp,
#             ground_truth=ground_truth
#         )
#         print(f"Транскрипция (Beam Search, beam_width={beam_width}, length_penalty={lp}): {transcriptionBS_NumPy}")
#
# print("Транскрипция жадного декодирования (NumPy):", transcriptionGD_NumPy)
# print("Транскрипция декодирования по лучу (NumPy):", transcriptionBS_NumPy)
# transcription_NumPy, metrics_NumPy = decode_ctc_greedy(
#     log_probs=log_probs_NumPy_MONO,
#     vocab=VOCAB,
#     blank_idx=MY_CONSTANTS.BLANK_IDX,
#     max_vocab_idx=max_vocab_idx,
#     ground_truth=ground_truth,
# )
#
# print("Транскрипция:", transcription_NumPy)
#
# # Декодирование по лучу для NumPy реализации
# for beam_width in beam_widths:
#     for lp in length_penalties:
#         transcription, metric_result = decode_ctc_beam_search(
#             log_probs=log_probs_NumPy_MONO,
#             vocab=VOCAB,
#             blank_idx=MY_CONSTANTS.BLANK_IDX,
#             max_vocab_idx=max_vocab_idx,
#             beam_width=beam_width,
#             length_penalty=lp,
#             ground_truth=ground_truth,
#         )
#         print(f"\nTesting beam_width={beam_width}, length_penalty={lp}")
#         time.sleep(5)
#         print("Транскрипция декодирования по лучу (NumPy):", transcription)

rnnt_model = RnntASRPyTorch(
    encoder_path="onnx_models/encoder-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
    decoder_joint_path="onnx_models/decoder_joint-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
)

waveform, sr = torchaudio.load("audio_files/20250404_174500.wav")
print(f"Waveform shape: {waveform.shape}, sample rate: {sr}")
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)
    print(f"Resampled waveform shape: {waveform.shape}")

plt.figure(figsize=(10, 4))
plt.plot(waveform[0].numpy())
plt.title("Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

ground_truth = "мне необходимо вам рассказать следующую историю о своей жизни чем четче я говорю тем лучший результат я получу"

# beam_widths = [5, 10, 15]
# for beam_width in beam_widths:
#     for lp in [0.3, 0.7, 1.0, 1.5]:
#         print(f"\nТестирование RNN-T: beam_width={beam_width}, length_penalty={lp}")
#         transcription = rnnt_model.recognize(
#             waveforms=waveform.numpy(),
#             decode_flag="BS",
#             beam_width=beam_width,
#             length_penalty=lp,
#             ground_truth=ground_truth,
#             max_steps=2000,
#             min_tokens=20
#         )
#         print(f"Транскрипция: {transcription}")
#
#
# print("Транскрипция декодирования по лучу (RNN-T PyTorch):", transcription)
# Используем ИСПРАВЛЕННОЕ жадное декодирование
print("=== ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО ЖАДНОГО ДЕКОДИРОВАНИЯ ===")
transcription_fixed_gd = rnnt_model.recognize(
    waveforms=waveform.numpy(),
    decode_flag="GD",
    ground_truth=ground_truth,
    max_steps=3000,  # Увеличено для полной обработки
    min_tokens=15,
#     max_steps=2000,
#     min_tokens=18,
    state_init="zero"
)
print("Транскрипция исправленного жадного декодирования (RNN-T PyTorch):", transcription_fixed_gd)

# Сравнение со старым методом
print("\n=== СРАВНЕНИЕ СО СТАРЫМ МЕТОДОМ ===")
transcription_old_gd = rnnt_model.recognize(
    waveform.numpy(),
    decode_flag="GD",
    ground_truth=ground_truth,
    max_steps=2000,
    min_tokens=18,
    state_init="zero"
)
print("Транскрипция старого жадного декодирования:", transcription_old_gd)

print(f"\nGround Truth: '{ground_truth}'")
print(f"Старый метод: '{transcription_old_gd[0]}'")
print(f"Новый метод:  '{transcription_fixed_gd[0]}'")

# Подсчет слов
gt_words = len(ground_truth.split())
old_words = len(transcription_old_gd[0].split())
new_words = len(transcription_fixed_gd[0].split())

print(f"\nКоличество слов:")
print(f"Ground Truth: {gt_words}")
print(f"Старый метод: {old_words}")
print(f"Новый метод:  {new_words}")

# Тестирование beam search с исправленным препроцессингом
print("\n=== ТЕСТИРОВАНИЕ BEAM SEARCH ===")
transcription_bs = None
beam_widths = [8, 10, 12, 15]
for beam_width in beam_widths:
    for lp in [0.3, 0.7, 1.0, 1.5, 2.0]:
        print(f"\nТестирование RNN-T: beam_width={beam_width}, length_penalty={lp}")
        transcription_bs = rnnt_model.recognize(
            waveforms=waveform.numpy(),
            decode_flag="BS",
            beam_width=beam_width,
            length_penalty=lp,
            ground_truth=ground_truth,
            max_steps=3000,
            min_tokens=18
        )
        print(f"Beam Search результат: '{transcription_bs[0]}'")
        print(f"Количество слов: {len(transcription_bs[0].split())}")
print("Транскрипция декодирования по лучу (RNN-T PyTorch):", transcription_bs)

preprocessor_Rnnt_NumPy = RnntASRNumPy(
    encoder_path="onnx_models/encoder-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx",
    decoder_joint_path="onnx_models/decoder_joint-stt_ru_fastconformer_hybrid_large_pc_RNNT.onnx"
)

# Инференс RNN-T NumPy
audio_prev = audio_prev.astype(np.float32)
print(f"Форма audio_prev перед передачей в recognize (RNN-T NumPy): {audio_prev.shape}")
transcriptionGD_Rnnt_NumPy = preprocessor_Rnnt_NumPy.recognize(
    waveforms=audio_prev,
    decode_flag="GD",
    ground_truth=ground_truth
)

transcriptionBS_Rnnt_NumPy, metricsBS_Rnnt_NumPy = None, []
for beam_width in beam_widths:
    for lp in length_penalties:
        print(f"\nTesting RNN-T NumPy beam_width={beam_width}, length_penalty={lp}")
        time.sleep(5)
        transcriptionBS_Rnnt_NumPy= preprocessor_Rnnt_NumPy.recognize(
            waveforms=audio_prev,
            decode_flag="BS",
            beam_width=beam_width,
            length_penalty=lp,
            ground_truth=ground_truth
        )
        print(f"Транскрипция (RNN-T NumPy Beam Search, beam_width={beam_width}, length_penalty={lp}): {transcriptionBS_Rnnt_NumPy}")

print("Транскрипция жадного декодирования (RNN-T NumPy):", transcriptionGD_Rnnt_NumPy)
print("Транскрипция декодирования по лучу (RNN-T NumPy):", transcriptionBS_Rnnt_NumPy)