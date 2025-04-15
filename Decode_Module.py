import os
import subprocess
import re
import json
from datetime import datetime
from typing import (List,
                    Dict,
                    Tuple)
import numpy as np
import onnxruntime as rt
from jiwer import (wer,
                   cer,
                   mer,
                   wil,
                   wip)
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from phonemizer import phonemize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# from rouge_score import rouge_scorer  # Для ROUGE
import pymorphy3
from sentence_transformers import SentenceTransformer, util  # Для Semantic Similarity
from bert_score import score as bert_score # Для BERTScore
from torchmetrics.text import (WordErrorRate,
                               CharErrorRate,
                               BLEUScore,
                               WordInfoLost,
                               MatchErrorRate,
                               WordInfoPreserved)

# Инициализация модели для Semantic Similarity (загружаем модель один раз)
semantic_model = SentenceTransformer('sentence-transformers/LaBSE')  # Модель для русского языка

# Инициализация метрик из torchmetrics
wer_metric = WordErrorRate()
cer_metric = CharErrorRate()
wil_metric = WordInfoLost()
mer_metric = MatchErrorRate()
wip_metric = WordInfoPreserved()
bleu_metric = BLEUScore(n_gram=4,
                        weights=(0.25, 0.25, 0.25, 0.25))

# Инициализация морфологического анализатора для русского языка
morph = pymorphy3.MorphAnalyzer()

phoneme_cache: Dict[str, List[str]] = {}

# # Тест для проверки rouge-score
# def test_rouge_scorer():
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
#     # Простой тест на английском
#     ref = "hello world"
#     hyp = "hello world"
#     scores = scorer.score(ref, hyp)
#     print("Test ROUGE on English text:")
#     for key in scores:
#         print(f'{key}: {scores[key]}')
#     # Тест на русском (нормализованном)
#     ref_ru = "потому что в самолете все зависит от винта"
#     hyp_ru = "потому что в самолете все зависит от винта"
#     scores_ru = scorer.score(ref_ru, hyp_ru)
#     print("Test ROUGE on Russian text:")
#     for key in scores_ru:
#         print(f'{key}: {scores_ru[key]}')
#
# # Вызов теста перед использованием return_metrics
# test_rouge_scorer()
def compute_rouge_manual(reference: str,
                         hypothesis: str
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L manually.

    Parameters
    ----------
    reference : str
        Reference text (ground truth).
    hypothesis : str
        Hypothesis text (transcription).

    Returns
    -------
    Dict[str, float]
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1-scores.
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # ROUGE-1 (униграммы)
    ref_unigrams = set(ref_tokens)
    hyp_unigrams = set(hyp_tokens)
    overlapping_unigrams = len(ref_unigrams & hyp_unigrams)
    precision_1 = overlapping_unigrams / len(hyp_tokens) if hyp_tokens else 0
    recall_1 = overlapping_unigrams / len(ref_tokens) if ref_tokens else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    # ROUGE-2 (биграммы)
    ref_bigrams = set(tuple(ref_tokens[i:i+2]) for i in range(len(ref_tokens)-1))
    hyp_bigrams = set(tuple(hyp_tokens[i:i+2]) for i in range(len(hyp_tokens)-1))
    overlapping_bigrams = len(ref_bigrams & hyp_bigrams)
    precision_2 = overlapping_bigrams / len(hyp_bigrams) if hyp_bigrams else 0
    recall_2 = overlapping_bigrams / len(ref_bigrams) if ref_bigrams else 0
    f1_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0

    # ROUGE-L (наибольшая общая подпоследовательность)
    def lcs(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]

    lcs_length = lcs(ref_tokens, hyp_tokens)
    precision_l = lcs_length / len(hyp_tokens) if hyp_tokens else 0
    recall_l = lcs_length / len(ref_tokens) if ref_tokens else 0
    f1_l = 2 * (precision_l * recall_l) / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0

    return {
        "ROUGE-1": f1_1,
        "ROUGE-2": f1_2,
        "ROUGE-L": f1_l
    }

def normalize_text(text: str) -> str:
    """
    Normalize text by removing punctuation and extra spaces.

    Parameters
    ----------
    text : str
        Input text to normalize.

    Returns
    -------
    str
        Normalized text (lowercase, no punctuation, single spaces).
    """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Удаляем пунктуацию
    text = re.sub(r'\s+', ' ', text).strip()     # Удаляем двойные пробелы
    return text

def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using pymorphy2 for Russian language.

    Parameters
    ----------
    text : str
        Input text to lemmatize.

    Returns
    -------
    str
        Lemmatized text (words in normal form, joined by spaces).
    """
    return " ".join(morph.parse(word)[0].normal_form for word in text.split())

def get_phonemes(text: str) -> List[str]:
    """
    Convert text to phonemes using phonemizer with caching.

    Parameters
    ----------
    text : str
        Input text to convert to phonemes.

    Returns
    -------
    List[str]
        List of phonemes.

    Notes
    -----
    Uses a global cache (phoneme_cache) to avoid recomputing phonemes for the same text.
    """
    if text in phoneme_cache:
        return phoneme_cache[text]
    phonemes = phonemize(text,
                         language='ru',
                         backend='espeak').split()
    phoneme_cache[text] = phonemes
    return phonemes

def ensure_espeak_in_path() -> None:
    """
    Ensure that espeak is available in the system PATH for phonemizer.

    Raises
    ------
    subprocess.CalledProcessError
        If espeak version check fails.
    FileNotFoundError
        If espeak executable is not found.
    """
    current_path = os.environ.get("PATH", "")
    print(f"Current PATH: {current_path}")

    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts"))
    espeak_venv_path = os.path.join(venv_path, "command_line", "espeak.exe")

    if os.path.exists(espeak_venv_path):
        print(f"Found espeak.exe in virtual environment: {espeak_venv_path}")
        command_line_path = os.path.dirname(espeak_venv_path)
        if command_line_path not in current_path:
            os.environ["PATH"] = f"{command_line_path}{os.pathsep}{current_path}"
            print(f"Added {command_line_path} to PATH: {os.environ['PATH']}")
    else:
        print(f"espeak.exe not found in {espeak_venv_path}. Falling back to system path.")
        espeak_system_path = r"C:\Program Files (x86)\eSpeak\command_line"
        if espeak_system_path not in current_path:
            print(f"Adding {espeak_system_path} to PATH.")
            os.environ["PATH"] = f"{espeak_system_path}{os.pathsep}{current_path}"
            print(f"Updated PATH: {os.environ['PATH']}")
        else:
            print(f"{espeak_system_path} already in PATH.")

    try:
        result = subprocess.run(["espeak", "--version"],
                                capture_output=True,
                                text=True,
                                check=True)
        print(f"espeak version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error verifying espeak: {e}")
        raise

def round_metrics(metrics: Dict[str, float],
                  precision: int = 14
) -> Dict[str, float]:
    """
    Round numerical values in the metrics dictionary to the specified precision.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics.
    precision : int, optional
        Number of decimal places to round to. Defaults to 14.

    Returns
    -------
    Dict[str, float]
        Dictionary with rounded values.
    """
    rounded_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rounded_metrics[key] = round(value, precision)
        else:
            rounded_metrics[key] = value
    return rounded_metrics

def log_metrics_to_json(metrics: Dict[str, float],
                        ground_truth: str,
                        transcription: str,
                        total_log_prob: float,
                        beam_width: int,
                        length_penalty: float,
                        flag: str,
                        filename: str = "metrics_log.json"
) -> None:
    """
    Log metrics to a JSON file as a list of entries, including context and timestamp.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of computed metrics.
    ground_truth : str
        Ground truth text.
    transcription : str
        Transcription text.
    total_log_prob: float
        Total log probability.
    beam_width: int
        Beam width.
    length_penalty: float
        Length penalty.
    flag: str
        Shows which method of decode is currently using
        Options: "greedy" - greedy decoding,
                 "beam" - decode by using a beam search
    filename : str, optional
        Path to the JSON log file. Defaults to "metrics_log.json".
    """
    if flag == "greedy":
        entry = {
            "timestamp": datetime.now().isoformat(),
            "flag": flag,
            "ground_truth": ground_truth,
            "transcription": transcription,
            "total_log_prob": total_log_prob,
            "metrics": round_metrics(metrics),
        }
    elif flag == "beam":
        entry = {
            "timestamp": datetime.now().isoformat(),
            "flag": flag,
            "ground_truth": ground_truth,
            "transcription": transcription,
            "total_log_prob": total_log_prob,
            "beam_width": beam_width,
            "length_penalty": length_penalty,
            "metrics": round_metrics(metrics),
        }
    if os.path.exists(filename):
        try:
            with open(filename,
                      "r",
                      encoding="utf-8") as file:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {filename}: {e}. Starting with an empty list!")
            data = []
    else:
        data = []

    data.append(entry)
    try:
        with open(filename,
                  "w",
                  encoding="utf-8") as file:
            json.dump(data,
                      file,
                      indent=4,
                      ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

def return_metrics(transcription: str,
                   ground_truth: str = None,
                   metrics: Dict[str, float] = None,
                   compute_semantic: bool = True,
                   total_log_prob: float = 0.0,
                   beam_width: int = 3,
                   length_penalty: float = 1.0,
                   flag: str = "greedy"
) -> Dict[str, float]:
    """
    Compute various evaluation metrics for comparing transcription with ground truth.

    Parameters
    ----------
    transcription : str
        Predicted transcription text.
    ground_truth : str, optional
        Reference (ground truth) text.
    metrics : Dict[str, float], optional
        Dictionary to store computed metrics. If None, a new dictionary is created.
    compute_semantic: bool, optional
        Flag that shows need we to count Semantic Similarity or not.
    total_log_prob: float
        Total log probability.
    beam_width : int, optional
        Number of beams to keep at each step. Defaults to 3.
    length_penalty : float, optional
        Length penalty to apply during beam selection. Defaults to 1.0.
    flag: str
        Shows which method of decode is currently using
        Options: "greedy" - greedy decoding,
                 "beam" - decode by using a beam search

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics:
        - WER, CER, MER, WIL, WIP, RIL (from jiwer)
        - WER_torchmetrics, CER_torchmetrics, WIL_torchmetrics, MER_torchmetrics, WIP_torchmetrics (from torchmetrics)
        - BLEU, BLEU_torchmetrics
        - ROUGE-1, ROUGE-2, ROUGE-L
        - Semantic Similarity (cosine similarity between embeddings)
        - PER (Phoneme Error Rate)
        - METEOR
        - BERTScore

    Raises
    ------
    ValueError
        If ground truth or transcription is empty.

    Notes
    -----
    - Some metrics (e.g., PER, Semantic Similarity) may return None if computation fails.
    - Texts are normalized and lemmatized before computing ROUGE.
    - Metrics are logged to 'metrics_log.json'.
    """
    if not ground_truth or not transcription:
        raise ValueError("Ground truth and transcription must be provided (not empty).")

    if metrics is None:
        metrics = {}

        # Метрики из jiwer
    metrics["WER"] = wer(ground_truth, transcription)
    metrics["CER"] = cer(ground_truth, transcription)
    metrics["MER"] = mer(ground_truth, transcription)
    metrics["WIL"] = wil(ground_truth, transcription)
    metrics["WIP"] = wip(ground_truth, transcription)
    metrics["RIL"] = 1 - metrics["WIP"]

    # Метрики из torchmetrics
    metrics["WER_torchmetrics"] = wer_metric([transcription], [ground_truth]).item()
    metrics["CER_torchmetrics"] = cer_metric([transcription], [ground_truth]).item()
    metrics["WIL_torchmetrics"] = wil_metric([transcription], [ground_truth]).item()
    metrics["MER_torchmetrics"] = mer_metric([transcription], [ground_truth]).item()
    metrics["WIP_torchmetrics"] = wip_metric([transcription], [ground_truth]).item()
    metrics["RIL_torchmetrics"] = 1 - metrics["WIP_torchmetrics"]

    # BLEU
    ref_tokens = ground_truth.split()
    hyp_tokens = transcription.split()
    if len(hyp_tokens) < 4:
        metrics["BLEU"] = 0.0
        metrics["BLEU_torchmetrics"] = 0.0
        print("Warning: Transcription too short for meaningful BLEU score")
    else:
        metrics["BLEU"] = sentence_bleu([ref_tokens],
                                        hyp_tokens,
                                        weights=(0.25, 0.25, 0.25, 0.25))
        metrics["BLEU_torchmetrics"] = bleu_metric([transcription], [[ground_truth]]).item()

    # ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
    # Вариант ниже работает только с английским текстом, не поддерживая кириллицу
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
    #                                   use_stemmer=False)
    # Нормализация текста
    ground_truth_norm = normalize_text(ground_truth)
    transcription_norm = normalize_text(transcription)
    print(f"Ground Truth (normalized): '{ground_truth_norm}'")
    print(f"Transcription (normalized): '{transcription_norm}'")
    assert ground_truth_norm.strip() != "", "Ground Truth is empty after normalization!"
    assert transcription_norm.strip() != "", "Transcription is empty after normalization!"

    # Для отладки: выведем токены, которые передаются в ROUGE
    ground_truth_tokens = ground_truth_norm.split()
    transcription_tokens = transcription_norm.split()
    print(f"Ground Truth tokens: {ground_truth_tokens}")
    print(f"Transcription tokens: {transcription_tokens}")
    print([ord(c) for c in ground_truth_norm])
    print([ord(c) for c in transcription_norm])

    # Вычисляем ROUGE
    rouge_scores = compute_rouge_manual(ground_truth_norm,
                                        transcription_norm)
    print("ROUGE scores:")
    for key, value in rouge_scores.items():
        print(f"{key}: {value}")
    metrics["ROUGE-1"] = rouge_scores["ROUGE-1"]
    metrics["ROUGE-2"] = rouge_scores["ROUGE-2"]
    metrics["ROUGE-L"] = rouge_scores["ROUGE-L"]

    # Semantic Similarity
    if compute_semantic:
        try:
            ref_embedding = semantic_model.encode(ground_truth,
                                                  convert_to_tensor=True)
            hyp_embedding = semantic_model.encode(transcription,
                                                  convert_to_tensor=True)
            semantic_similarity = util.cos_sim(ref_embedding, hyp_embedding).item()
            metrics["Semantic Similarity"] = semantic_similarity
            if semantic_similarity < 0.5:
                print("Warning: Low semantic similarity, texts may have different meanings")
        except Exception as e:
            print(f"Error computing Semantic Similarity: {e}")
            metrics["Semantic Similarity"] = None
    else:
        metrics["Semantic Similarity"] = None

    # Phoneme Error Rate (PER)
    try:
        ref_phonemes = get_phonemes(ground_truth)
        hyp_phonemes = get_phonemes(transcription)
        metrics["PER"] = wer(" ".join(ref_phonemes), " ".join(hyp_phonemes))
    except Exception as e:
        print(f"Error computing PER with phonemizer: {e}")
        metrics["PER"] = None

    # METEOR
    try:
        metrics["METEOR"] = meteor_score([ground_truth.split()], transcription.split())
    except Exception as e:
        print(f"Error computing METEOR: {e}")
        metrics["METEOR"] = None

    # BERTScore
    try:
        P, R, F1 = bert_score([transcription],
                              [ground_truth],
                              lang="ru",
                              verbose=False)
        metrics["BERTScore"] = F1.item()
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        metrics["BERTScore"] = None

    # Логирование метрик в файл
    log_metrics_to_json(metrics=metrics,
                        ground_truth=ground_truth,
                        transcription=transcription,
                        total_log_prob=total_log_prob,
                        beam_width=beam_width,
                        length_penalty=length_penalty,
                        flag=flag
                        )

    print("Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.15f}")
        else:
            print(f"{metric}: Not computed due to error")
    return metrics
# Вызов функции перед использованием phonemizer
# ensure_espeak_in_path()

# Использование WER (Word Error Rate)
# WER = (S + D + I) / N (CSR модель)
# WER = S / (N1 = H + S) = 1 - H/N1 (IWR модель)
# Смысл: измерение процента ошибок на уровне слов: сравнение эталонного текста с предсказанным
# S - количество замен (substitutions)
# D - количество удвлений (deletions)
# I - количество вставок (insertations)
# H - количество попадений (hits)
# N  - общее число слов в эталонном тексте = H + S + D - количество входных слов
# N1 = H + S - число пар-совпадений (matched I/O word pairs)

# Использование CER (Character Error Rate)
# CER = (Sc + Dc + Ic) / N
# Смысл: Сравнение эталонного и предсказанного текста посимвольно

# Использование MER (Match Error Rate)
# MER = N_not_matched / N_total = (S + D + I) / (N = H + S + D + I) = 1 - H / N_total
# Смысл: Показ доли слов, не совпавших с эталоном
# N_not_matched - число несовпавших слов
# N_total - общее число слов в эталоне

# Использование WIL (Word Information Lost)
# WIL = 1 - H^2 / (N1 * N2) = 1 - WIP = I(x,y) / H(Y) (на выравнивании слов и штрафах за ошибки)
# Смысл: Метрика для оценки потери информации на уровне слов с учётом совпадений подстрок
# H >> S + D + I
# N1 - число входных слов, N2 - число выходных слов

# Использование WIP (Word Information Preserved)
# WIP = H^2 / (N1 * N2)
# Смысл: Метрика для оценки сохранённой информации на уровне слов
# H - количество попаданий (hits)
# N1 - число слов в эталоне (ground truth), N2 - число слов в предсказании (transcription)
# Значение: от 0 до 1, где 1 означает, что вся информация сохранена (H = N1 = N2)

# Использование RIL (Reference Information Lost)
# RIL = 1 - WIP = 1 - H^2 / (N1 * N2)
# Смысл: Метрика для оценки потерянной информации из эталона (ground truth) при предсказании
# Значение: от 0 до 1, где 0 означает, что вся информация из эталона сохранена, а 1 — что вся информация потеряна
# RIL дополняет WIP: если WIP высок (много информации сохранено), то RIL низкий, и наоборот

# Использование BLEU (Bilingual Evaluation Understudy)
# BLEU = BP * exp(sum(n=1, N) w(n)*log(p(n)))
# Смысл: Метрика машинного перевода для измерения точности n-грамм (последовательностей слов) в предсказании по сравнению с эталоном
# BP - штраф за длину
# p(n) - точность n-грамм
# w(n) - веса

# Использование ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
# ROUGE-N = (пересечение n-грамм между эталоном и предсказанием) / (число n-грамм в эталоне)
# ROUGE-L = (длина наибольшей общей подпоследовательности) / (длина эталона)
# Смысл: Метрика суммаризации текстов, измеряет пересечение n-грамм и общих подпоследовательностей

# Использование PER (Phenomene Error Rate)
# PER = (S(p) + D(p) + I(p)) / N(p) (пока не используется)
# Смысл: Метрика для оценки ошибок на уровне фонем (звуков): преобразование текстов в фонетическую транскрипцию
# S(p) - количество замен фонем
# D(p) - количество удаления фонем
# I(p) - количество вставок фонем
# N(p) - число фонем в эталоне

# Использование семантической близости (Semantic Similarity)
# Смысл: Использует методы embeddings (BERT, FastText) для оценки семантической близости между эталоном и предсказанием
# Значение: Косинусное сходство между эмбеддингами (от 0 до 1)

# Использование METEOR (Metric for Evaluation of Translation with Explicit ORdering)
# METEOR = P * R / ((1 - α) * R + α * P) * (1 - γ * (frag_score)^β)
# Смысл: Метрика для оценки качества перевода или текста с учётом синонимии, морфологии и порядка слов
# P - точность (precision): доля слов в предсказании, совпадающих с эталоном (с учётом синонимов)
# R - полнота (recall): доля слов из эталона, найденных в предсказании
# α - весовой коэффициент для балансировки между P и R (обычно 0.9)
# frag_score - штраф за фрагментацию: m / u, где m - число "кусков" совпадающих слов, u - число совпавших слов
# γ - вес штрафа за фрагментацию (обычно 0.5)
# β - степень штрафа за фрагментацию (обычно 3.0)
# Значение: от 0 до 1, где 1 означает идеальное совпадение

# Использование BERTScore
# BERTScore = max(cos_sim(emb_pred_i, emb_ref_j)) для каждого токена i в предсказании и j в эталоне
# Смысл: Метрика для оценки семантической близости текста с использованием эмбеддингов BERT
# emb_pred_i - эмбеддинг i-го токена предсказания (получается из BERT)
# emb_ref_j - эмбеддинг j-го токена эталона (получается из BERT)
# cos_sim - косинусное сходство между эмбеддингами токенов
# P - средняя точность: усреднённое максимальное сходство для токенов предсказания
# R - средняя полнота: усреднённое максимальное сходство для токенов эталона
# F1 - гармоническое среднее между P и R
# Значение: от 0 до 1, где 1 означает полное семантическое совпадение

def tensor_info(flag: str,
                tensors: List[rt.NodeArg]
) -> None:
    """
    Print information about the input or output tensors of an ONNX model.

    Parameters
    ----------
    flag : str
        Either "i" for inputs or "o" for outputs.
    tensors : List[rt.NodeArg]
        List of input or output nodes from the ONNX model.

    Raises
    ------
    ValueError
        If flag is neither 'i' nor 'o'.
    """
    if flag == "i":
        print("Inputs expected by the model:")
    elif flag == "o":
        print("Outputs of the model:")
    else:
        raise ValueError(f"Invalid flag: {flag}. Must be 'i' for inputs or 'o' for outputs.")

    if not tensors:
        print("No tensors found.")
        return

    for tensor in tensors:
        print(f"Name: {tensor.name}, Shape: {tensor.shape}")

def decode_ctc_greedy(log_probs: np.ndarray,
                      vocab: List[str],
                      blank_idx: int,
                      max_vocab_idx: int,
                      ground_truth: str = None
) -> Tuple[str, Dict[str, float]]:
    """
    Perform greedy decoding on CTC log probabilities to produce a transcription.

    Parameters
    ----------
    log_probs : np.ndarray
        Log probabilities from the model.
        Shape: [batch_size, seq_len, num_classes], where batch_size must be 1.
    vocab : List[str]
        Vocabulary list mapping token indices to characters/tokens.
    blank_idx : int
        Index of the blank token in the vocabulary.
    max_vocab_idx : int
        Maximum valid token index (len(vocab) - 1).
    ground_truth : str, optional
        Ground truth transcription for computing metrics.

    Returns
    -------
    Tuple[str, Dict[str, float]]
        - transcription : str
            Decoded transcription.
        - metrics : Dict[str, float]
            Evaluation metrics (if ground_truth is provided).

    Raises
    ------
    ValueError
        If batch_size is not 1.
    """
    if log_probs.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")

    # Extract log probabilities for the single batch
    # Shape: [seq_len, num_classes]
    log_prob = log_probs[0]

    # Get the most probable token at each time step
    # Shape: [seq_len], dtype: int
    token_ids = log_probs.argmax(-1).squeeze().tolist()
    print("Log probs shape:", log_probs.shape)
    print("Predicted token indices:", token_ids)

    # Decode by removing blanks and consecutive duplicates
    decoded_ids: List[int] = []
    total_log_prob = 0.0
    prev_tok = None
    for t, tok in enumerate(token_ids):
        # Accumulate log probability of the selected token
        total_log_prob += float(log_prob[t, tok])
        if tok > max_vocab_idx:
            print(f"Warning: Token {tok} exceeds VOCAB size ({max_vocab_idx}), skipping")
            continue
        # Add token if it is not a blank and not a duplicate of the previous token
        if (tok != prev_tok or prev_tok == blank_idx) and tok != blank_idx:
            decoded_ids.append(tok)
        prev_tok = tok
    # Convert token IDs to text using the vocabulary
    transcription = "".join(vocab[tok] for tok in decoded_ids)
    print(f"Decoded transcription (Greedy): {transcription}")
    print(f"Log probability (Greedy): {total_log_prob:.15f}")

    # Compute metrics if ground truth is provided
    metrics = {}
    if ground_truth:
        metrics = return_metrics(transcription=transcription,
                                 ground_truth=ground_truth,
                                 metrics=metrics,
                                 total_log_prob=total_log_prob,
                                 flag="greedy")
    return transcription, metrics

def decode_ctc_beam_search(
    log_probs: np.ndarray,
    vocab: List[str],
    blank_idx: int,
    max_vocab_idx: int,
    beam_width: int = 3,
    length_penalty: float = 1.0,
    ground_truth: str = None
) -> Tuple[str, Dict[str, float]]:
    """
    Perform beam search decoding on CTC log probabilities to produce a transcription.

    Parameters
    ----------
    log_probs : np.ndarray
        Log probabilities from the model.
        Shape: [batch_size, seq_len, num_classes], where batch_size must be 1.
    vocab : List[str]
        Vocabulary list mapping token indices to characters/tokens.
    blank_idx : int
        Index of the blank token in the vocabulary.
    max_vocab_idx : int
        Maximum valid token index (len(vocab) - 1).
    beam_width : int, optional
        Number of beams to keep at each step. Defaults to 3.
    length_penalty : float, optional
        Length penalty to apply during beam selection. Defaults to 1.0.
    ground_truth : str, optional
        Ground truth transcription for computing metrics.

    Returns
    -------
    Tuple[str, Dict[str, float]]
        - transcription : str
            Decoded transcription.
        - metrics : Dict[str, float]
            Evaluation metrics (if ground_truth is provided).

    Raises
    ------
    TypeError
        If log_probs is not a numpy.ndarray.
    ValueError
        If batch_size is not 1, or if log_probs shape is incorrect, or if max_vocab_idx/blank_idx are invalid.
    """
    # Проверка типа и формы log_probs
    if not isinstance(log_probs, np.ndarray):
        raise TypeError(f"Expected log_probs to be a numpy.ndarray, got {type(log_probs)}")

    if log_probs.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")

    if len(log_probs.shape) != 3:
        raise ValueError(f"Expected log_probs shape [batch_size, seq_len, num_classes], got {log_probs.shape}")

    # Extract log probabilities for the single batch
    # Shape: [seq_len, num_classes]
    log_prob = log_probs[0]
    seq_len, num_classes = log_prob.shape

    if max_vocab_idx >= num_classes:
        raise ValueError(f"max_vocab_idx ({max_vocab_idx}) must be less than num_classes ({num_classes})")
    if blank_idx >= num_classes:
        raise ValueError(f"blank_idx ({blank_idx}) must be less than num_classes ({num_classes})")

    # Инициализация лучей: (sequence, prob_blank, prob_non_blank, seq_len)
    # prob_blank — вероятность последовательности, заканчивающейся на blank
    # prob_non_blank — вероятность последовательности, заканчивающейся на non-blank
    # Shape: Dict[Tuple[int, ...], Tuple[float, float, int]]
    beams: Dict[Tuple[int, ...], Tuple[float, float, int]] = {tuple(): (0.0, float('-inf'), 0)}

    # Для каждого временного шага
    for t in range(seq_len):
        # Shape: Dict[Tuple[int, ...], Tuple[float, float, int]]
        new_beams: Dict[Tuple[int, ...], Tuple[float, float, int]] = {}

        # Для каждой текущей гипотезы
        for seq, (prob_blank, prob_non_blank, seq_len_so_far) in beams.items():
            # Общая вероятность последовательности на предыдущем шаге
            prob_total = np.logaddexp(prob_blank, prob_non_blank)

            # Для каждого возможного токена
            for token in range(num_classes):
                if token > max_vocab_idx and token != blank_idx:
                    continue

                token_log_prob = float(log_prob[t, token])

                # Случай 1: Текущий токен — blank
                if token == blank_idx:
                    # Вероятность последовательности, заканчивающейся на blank
                    new_prob_blank = prob_total + token_log_prob
                    current = new_beams.get(tuple(seq), (float('-inf'), float('-inf'), seq_len_so_far))
                    new_beams[tuple(seq)] = (
                        np.logaddexp(current[0], new_prob_blank),
                        current[1],
                        seq_len_so_far
                    )
                else:
                    # Случай 2: Текущий токен — не blank
                    new_seq = list(seq)

                    # Если последовательность не пуста и последний токен совпадает с текущим
                    if new_seq and new_seq[-1] == token:
                        # Вероятность последовательности, заканчивающейся на non-blank (без добавления токена)
                        new_prob_non_blank = prob_non_blank + token_log_prob
                        current = new_beams.get(tuple(seq), (float('-inf'), float('-inf'), seq_len_so_far))
                        new_beams[tuple(seq)] = (
                            current[0],
                            np.logaddexp(current[1], new_prob_non_blank),
                            seq_len_so_far
                        )
                    else:
                        # Добавляем токен в последовательность
                        new_seq.append(token)
                        new_seq_len = seq_len_so_far + 1

                        # Вероятность последовательности, заканчивающейся на non-blank
                        new_prob_non_blank = prob_total + token_log_prob
                        current = new_beams.get(tuple(new_seq), (float('-inf'), float('-inf'), new_seq_len))
                        new_beams[tuple(new_seq)] = (
                            current[0],
                            np.logaddexp(current[1], new_prob_non_blank),
                            new_seq_len
                        )

        # Выбираем beam_width лучших гипотез
        # Sort beams by total probability (with length penalty) and keep top beam_width
        beams = {}
        for seq, (prob_blank, prob_non_blank, seq_len_so_far) in sorted(
            new_beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]) / (x[1][2] ** length_penalty if x[1][2] > 0 else 1.0),
            reverse=True
        )[:beam_width]:
            beams[tuple(seq)] = (prob_blank, prob_non_blank, seq_len_so_far)

    # Выбираем лучшую гипотезу
    best_seq, (best_prob_blank, best_prob_non_blank, _) = max(
        beams.items(),
        key=lambda x: np.logaddexp(x[1][0], x[1][1])
    )
    best_log_prob = np.logaddexp(best_prob_blank, best_prob_non_blank)
    transcription = "".join(vocab[tok] for tok in best_seq)
    print(f"Transcription (Beam Search, beam_width={beam_width}, length_penalty={length_penalty}): {transcription}")
    print(f"Log probability (Beam Search): {best_log_prob + 1:.14f}")

    # Compute metrics if ground truth is provided
    metrics = {}
    if ground_truth:
        metrics = return_metrics(transcription=transcription,
                                 ground_truth=ground_truth,
                                 metrics=metrics,
                                 total_log_prob=best_log_prob + 1,
                                 beam_width=beam_width,
                                 length_penalty=length_penalty,
                                 flag="beam")

    return transcription, metrics