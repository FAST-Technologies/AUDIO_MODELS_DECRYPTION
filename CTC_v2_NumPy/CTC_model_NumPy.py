from typing import List, Optional
import numpy as np

class CTCHead_V2_0:
    """
    CTC Head module for Connectionist Temporal Classification.
    """
    def __init__(self,
                 feat_in: int,
                 num_classes: int
    ) -> None:
        k = 1.0 / feat_in
        self.feat_in = feat_in
        self.num_classes = num_classes
        # self.weights = np.random.randn(feat_in, num_classes) * np.sqrt(2.0 / feat_in)
        # self.bias = np.zeros(num_classes)
        self.weights = np.random.uniform(-np.sqrt(k), np.sqrt(k), (num_classes, feat_in))
        self.bias = np.random.uniform(-np.sqrt(k), np.sqrt(k), (1, num_classes))


    def forward(self,
                encoder_output: np.ndarray
    ) -> np.ndarray:
        # encoder_output: [B, T, feat_in] -> нужно транспонировать в [B, feat_in, T] для свёртки
        x = encoder_output.transpose(0, 2, 1)  # [B, feat_in, T]

        # Свёртка 1x1 эквивалентна матричному умножению по каналам
        # [B, feat_in, T] @ [feat_in, num_classes] -> [B, num_classes, T]
        conv_output = np.matmul(x.transpose(0, 2, 1), self.weights) + self.bias  # [B, T, num_classes]

        # Транспонируем для log_softmax: [B, T, num_classes] -> [B, T, C]
        conv_output = conv_output.transpose(0, 1, 2)

        # Вычисление log_softmax
        def log_softmax(x: np.ndarray):
            x_max = np.max(x,
                           axis=-1,
                           keepdims=True)  # Для численной стабильности
            exp_x = np.exp(x - x_max)
            sum_exp_x = np.sum(exp_x,
                               axis=-1,
                               keepdims=True)
            return x - x_max - np.log(sum_exp_x)

        return log_softmax(conv_output)

# Класс жадного декодирования для CTC
class CTCGreedyDecoding_V2_0:
    # Инициализация
    def __init__(self,
                 vocabulary: List[str],
                 model_path: Optional[str] = None
    ) -> None:
        self.vocabulary = vocabulary
        self.blank_id = len(self.vocabulary)
        if model_path is not None:
            raise ValueError("SentencePiece model is not supported in this implementation")

    # Декодирование
    def decode(self,
               head: CTCHead_V2_0,
               encoded: np.ndarray,
               lengths: np.ndarray
    ) -> List[str]:
        """
        Decode the output of a CTC model into a list of hypotheses.
        """
        log_probs = head(encoder_output=encoded)
        assert (
            len(log_probs.shape) == 3
        ), f"Expected log_probs shape {log_probs.shape} == [B, T, C]"
        b, _, c = log_probs.shape
        assert (
            c == len(self.vocabulary) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.vocabulary) + 1}"

        labels = np.argmax(log_probs,
                           axis=-1)  # [B, T]

        skip_mask = labels != self.blank_id

        skip_mask[:, 1:] = np.logical_and(
            skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1]
        )

        # Применяем длины последовательностей
        for i, length in enumerate(lengths):
            skip_mask[i, length:] = 0

        pred_texts: List[str] = []
        for i in range(b):
            token_ids = labels[i][skip_mask[i]].tolist()
            text = "".join(self.vocabulary[tok] for tok in token_ids)
            pred_texts.append(text)
        print(f"Current text NumPy: {pred_texts}")
        return pred_texts