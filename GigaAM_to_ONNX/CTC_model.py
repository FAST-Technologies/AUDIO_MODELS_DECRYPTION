from typing import List, Optional
import torch
from torch import Tensor
import torch.nn as nn
from .Tokenizer import Tokenizer

class CTCHead(nn.Module):
    """
    CTC Head module for Connectionist Temporal Classification.
    """

    def __init__(self,
                 feat_in: int,
                 num_classes: int
    ) -> None:
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(feat_in, num_classes, kernel_size=1)
        )

    def forward(self,
                encoder_output: Tensor
    ) -> Tensor:
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
        )

# Класс жадного декодирования для CTC
class CTCGreedyDecoding:
    # Инициализация
    def __init__(self,
                 vocabulary: List[str],
                 model_path: Optional[str] = None
    ) -> None:
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)

    @classmethod
    def from_vocabulary(cls,
                        vocabulary: List[str],
                        model_path: Optional[str] = None
    ) -> 'CTCGreedyDecoding':
        """
        Alternative constructor to create a CTCGreedyDecoding instance from a vocabulary.

        Args:
            vocabulary (List[str]): List of tokens in the vocabulary.
            model_path (Optional[str]): Path to a pre-trained model (optional).

        Returns:
            CTCGreedyDecoding: An instance of the CTCGreedyDecoding class.\

        Usage example:
            decoder = CTCGreedyDecoding.from_vocabulary(vocabulary=["a", "b", "c"])
        """
        return cls(vocabulary, model_path)

    # Декодирование
    @torch.inference_mode()
    def decode(self,
               head: CTCHead,
               encoded: Tensor,
               lengths: Tensor
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
            c == len(self.tokenizer) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.tokenizer) + 1}"
        labels = log_probs.argmax(dim=-1, keepdim=False)

        skip_mask = labels != self.blank_id
        skip_mask[:, 1:] = torch.logical_and(
            skip_mask[:, 1:],
            torch.ne(labels[:, 1:], labels[:, :-1])
        )
        for length in lengths:
            skip_mask[length:] = 0

        pred_texts: List[str] = []
        for i in range(b):
            pred_texts.append(
                "".join(self.tokenizer.decode(labels[i][skip_mask[i]].cpu().tolist()))
            )
        return pred_texts