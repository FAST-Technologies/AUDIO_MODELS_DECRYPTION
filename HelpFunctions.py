import os
import subprocess
import re
from typing import List, Dict, Tuple
import numpy as np
import onnxruntime as rt
from jiwer import wer, cer, mer, wil, wip
from nltk.translate.bleu_score import sentence_bleu
from phonemizer import phonemize
import nltk
nltk.download('punkt')

from rouge_score import rouge_scorer  # Для ROUGE
from sentence_transformers import SentenceTransformer, util  # Для Semantic Similarity
from torchmetrics.text import WordErrorRate, CharErrorRate, BLEUScore, WordInfoLost, MatchErrorRate, WordInfoPreserved

# Инициализация модели для Semantic Similarity (загружаем модель один раз)
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Модель для русского языка

# Инициализация метрик из torchmetrics
wer_metric = WordErrorRate()
cer_metric = CharErrorRate()
wil_metric = WordInfoLost()
mer_metric = MatchErrorRate()
wip_metric = WordInfoPreserved()
bleu_metric = BLEUScore(n_gram=4,
                        weights=(0.25, 0.25, 0.25, 0.25))

def normalize_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Удаляем пунктуацию
    text = re.sub(r'\s+', ' ', text).strip()     # Удаляем двойные пробелы
    return text

def ensure_espeak_in_path():
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
        result = subprocess.run(["espeak", "--version"], capture_output=True, text=True, check=True)
        print(f"espeak version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error verifying espeak: {e}")
        raise

def return_metrics(transcription: str,
                   ground_truth: str = None,
                   metrics=None
) -> Dict[str, float]:
    if metrics is None:
        metrics = {}
    metrics["WER"] = wer(ground_truth, transcription)
    metrics["CER"] = cer(ground_truth, transcription)
    metrics["MER"] = mer(ground_truth, transcription)
    metrics["WIL"] = wil(ground_truth, transcription)
    metrics["WIP"] = wip(ground_truth, transcription)
    metrics["RIL"] = 1 - metrics["WIP"]  # RIL = 1 - WIP

    metrics["WER_torchmetrics"] = wer_metric([transcription], [ground_truth]).item()
    metrics["CER_torchmetrics"] = cer_metric([transcription], [ground_truth]).item()
    metrics["WIL_torchmetrics"] = wil_metric([transcription], [ground_truth]).item()
    metrics["MER_torchmetrics"] = mer_metric([transcription], [ground_truth]).item()
    metrics["WIP_torchmetrics"] = wip_metric([transcription], [ground_truth]).item()

    ref_tokens = ground_truth.split()
    hyp_tokens = transcription.split()
    metrics["BLEU"] = sentence_bleu([ref_tokens],
                                    hyp_tokens,
                                    weights=(0.25, 0.25, 0.25, 0.25))

    # BLEU из torchmetrics (для сравнения)
    metrics["BLEU_torchmetrics"] = bleu_metric([transcription],
                                               [[ground_truth]]).item()

    # ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                      use_stemmer=False)
    ground_truth_norm = normalize_text(ground_truth)
    transcription_norm = normalize_text(transcription)
    print(f"Ground Truth (normalized): '{ground_truth_norm}'")
    print(f"Transcription (normalized): '{transcription_norm}'")
    assert ground_truth_norm.strip() != "", "Ground Truth is empty after normalization!"
    assert transcription_norm.strip() != "", "Transcription is empty after normalization!"
    rouge_scores = scorer.score(ground_truth_norm, transcription_norm)
    for key in rouge_scores:
        print(f'{key}: {rouge_scores[key]}')
    metrics["ROUGE-1"] = rouge_scores['rouge1'].fmeasure
    metrics["ROUGE-2"] = rouge_scores['rouge2'].fmeasure
    metrics["ROUGE-L"] = rouge_scores['rougeL'].fmeasure

    # Semantic Similarity
    try:
        # Получаем эмбеддинги для эталона и предсказания
        ref_embedding = semantic_model.encode(ground_truth,
                                              convert_to_tensor=True)
        hyp_embedding = semantic_model.encode(transcription,
                                              convert_to_tensor=True)
        # Вычисляем косинусное сходство
        semantic_similarity = util.cos_sim(ref_embedding, hyp_embedding).item()
        metrics["Semantic Similarity"] = semantic_similarity
    except Exception as e:
        print(f"Error computing Semantic Similarity: {e}")
        metrics["Semantic Similarity"] = None

    try:
        # Сохраняем текущую рабочую директорию
        # original_cwd = os.getcwd()
        # # Переходим в директорию с espeak
        # espeak_dir = r"E:\Прога (вся)\NeuralSpecter\AUDIO_MODELS_DECRYPTION\.venv\Scripts\command_line"
        # os.chdir(espeak_dir)
        # print(f"Changed working directory to: {espeak_dir}")
        # Выполняем phonemize
        ref_phonemes = phonemize(ground_truth,
                                 language='ru',
                                 backend='espeak').split()
        hyp_phonemes = phonemize(transcription,
                                 language='ru',
                                 backend='espeak').split()
        metrics["PER"] = wer(" ".join(ref_phonemes), " ".join(hyp_phonemes))
    except Exception as e:
        print(f"Error computing PER with phonemizer: {e}")
        metrics["PER"] = None
    # finally:
    #     # Возвращаем исходную рабочую директорию
    #     os.chdir(original_cwd)
    #     print(f"Restored working directory to: {original_cwd}")

    print("Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.7f}")
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

def tensor_info(flag: str,
                tensors: List[rt.NodeArg]
) -> None:
    """
    Print information about the input or output tensors of an ONNX model.

    Args:
        flag (str): Either "i" for inputs or "o" for outputs.
        tensors (List[rt.NodeArg]): List of input or output nodes from the ONNX model.
    """
    if flag == "i":
        print("Inputs expected by the model:")
    elif flag == "o":
        print("Outputs of the model:")
    else:
        raise ValueError(f"Invalid flag: {flag}. Must be 'i' for inputs or 'o' for outputs.")

    if not tensors:
        print("  No tensors found.")
        return

    for tensor in tensors:
        print(f"Name: {tensor.name}, Shape: {tensor.shape}")

def decode_ctc_greedy(log_probs: np.ndarray,
                      vocab: List[str],
                      blank_idx: int,
                      max_vocab_idx: int,
                      ground_truth: str = None
) -> Tuple[str, Dict[str, float]]:
    if log_probs.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")

    log_prob = log_probs[0]
    token_ids = log_probs.argmax(-1).squeeze().tolist()
    print("Log probs shape:", log_probs.shape)
    print("Predicted token indices:", token_ids)

    decoded_ids: List[int] = []
    total_log_prob = 0.0
    prev_tok = None
    for t, tok in enumerate(token_ids):
        total_log_prob += float(log_prob[t, tok])
        if tok > max_vocab_idx:
            print(f"Warning: Token {tok} exceeds VOCAB size ({max_vocab_idx}), skipping")
            continue
        if (tok != prev_tok or prev_tok == blank_idx) and tok != blank_idx:
            decoded_ids.append(tok)
        prev_tok = tok

    transcription = "".join(vocab[tok] for tok in decoded_ids)
    print(f"Decoded transcription (Greedy): {transcription}")
    print(f"Log probability (Greedy): {total_log_prob:.7f}")

    metrics = {}
    if ground_truth:
        metrics = return_metrics(transcription=transcription,
                                 ground_truth=ground_truth,
                                 metrics=metrics)
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

    Args:
        log_probs (np.ndarray): Log probabilities from the model, shape [batch_size, seq_len, num_classes].
        vocab (List[str]): Vocabulary list mapping token indices to characters/tokens.
        blank_idx (int): Index of the blank token in the vocabulary.
        max_vocab_idx (int): Maximum valid token index (len(vocab) - 1).
        beam_width (int): Number of beams to keep at each step.
        length_penalty (float): Length penalty to apply during beam selection.
        ground_truth (str): Ground truth transcription for computing metrics.

    Returns:
        Tuple[str, Dict[str, float]]: Decoded transcription and metrics.
    """
    # Проверка типа и формы log_probs
    if not isinstance(log_probs, np.ndarray):
        raise TypeError(f"Expected log_probs to be a numpy.ndarray, got {type(log_probs)}")

    if log_probs.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")

    if len(log_probs.shape) != 3:
        raise ValueError(f"Expected log_probs shape [batch_size, seq_len, num_classes], got {log_probs.shape}")

    log_prob = log_probs[0]  # [seq_len, num_classes]
    seq_len, num_classes = log_prob.shape

    if max_vocab_idx >= num_classes:
        raise ValueError(f"max_vocab_idx ({max_vocab_idx}) must be less than num_classes ({num_classes})")
    if blank_idx >= num_classes:
        raise ValueError(f"blank_idx ({blank_idx}) must be less than num_classes ({num_classes})")

    # Инициализация лучей: (sequence, prob_blank, prob_non_blank, seq_len)
    # prob_blank — вероятность последовательности, заканчивающейся на blank
    # prob_non_blank — вероятность последовательности, заканчивающейся на non-blank
    beams: Dict[Tuple[int, ...], Tuple[float, float, int]] = {tuple(): (0.0, float('-inf'), 0)}

    # Для каждого временного шага
    for t in range(seq_len):
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
    print(f"Log probability (Beam Search): {best_log_prob + 1:.7f}")

    metrics = {}
    if ground_truth:
        metrics = return_metrics(transcription=transcription,
                                 ground_truth=ground_truth,
                                 metrics=metrics)

    return transcription, metrics