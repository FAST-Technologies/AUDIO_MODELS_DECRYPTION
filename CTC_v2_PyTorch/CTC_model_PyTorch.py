from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from .Tokenizer_PyTorch import Tokenizer_PyTorch

class CTCHead_PyTorch(nn.Module):
    """
    CTC Head module for Connectionist Temporal Classification.
    """
    @torch.inference_mode()
    def __init__(self,
                 feat_in: int,
                 num_classes: int
    ) -> None:
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(feat_in, num_classes, kernel_size=1)
        )

    @torch.inference_mode()
    def forward(self,
                encoder_output: Tensor
    ) -> Tensor:
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
        )

# Класс жадного декодирования для CTC
class CTCGreedyDecoding_PyTorch:
    # Инициализация
    @torch.inference_mode()
    def __init__(self,
                 vocabulary: List[str],
                 model_path: Optional[str] = None
    ) -> None:
        self.tokenizer = Tokenizer_PyTorch(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)

    # Декодирование
    @torch.inference_mode()
    def decode(self,
               head: CTCHead_PyTorch,
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