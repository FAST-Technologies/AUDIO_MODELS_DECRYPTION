from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor

# Модуль не используется, но присутствует как пример на PyTorch
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
            torch.nn.Conv1d(feat_in,
                            num_classes,
                            kernel_size=1)
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
        self.vocabulary = vocabulary
        self.blank_id = len(self.vocabulary)
        if model_path is not None:
            raise ValueError("SentencePiece model is not supported in this implementation")

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
            c == len(self.vocabulary) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.vocabulary) + 1}"
        labels = log_probs.argmax(dim=-1,
                                  keepdim=False)

        skip_mask = labels != self.blank_id
        skip_mask[:, 1:] = torch.logical_and(
            skip_mask[:, 1:],
            torch.ne(labels[:, 1:], labels[:, :-1])
        )
        # for length in lengths:
        #     skip_mask[length:] = 0
        for i, length in enumerate(lengths):
            skip_mask[i, length:] = 0

        pred_texts: List[str] = []
        for i in range(b):
            token_ids = labels[i][skip_mask[i]].cpu().tolist()
            text = "".join(self.vocabulary[tok] for tok in token_ids)
            pred_texts.append(text)
        print(f"Current text (Greedy Search): {pred_texts}")
        return pred_texts

class BeamSearchDecoder_PyTorch:
    # Инициализация
    @torch.inference_mode()
    def __init__(self,
                 vocabulary: List[str],
                 model_path: Optional[str] = None
                 ) -> None:
        self.vocabulary = vocabulary
        self.blank_id = len(self.vocabulary)
        if model_path is not None:
            raise ValueError("SentencePiece model is not supported in this implementation")

    # Декодирование
    @torch.inference_mode()
    def decode(self,
               head: CTCHead_PyTorch,
               encoded: Tensor,
               lengths: Tensor,
               beam_width: int = 3
               ) -> List[str]:
        """
        Decode the output of a CTC model into a list of hypotheses.
        """
        log_probs = head(encoder_output=encoded)
        assert (
                len(log_probs.shape) == 3
        ), f"Expected log_probs shape {log_probs.shape} == [B, T, C]"
        b, t, c = log_probs.shape
        assert (
                c == len(self.vocabulary) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.vocabulary) + 1}"

        pred_texts: List[str] = []
        for batch_idx in range(b):
            log_probs_batch = log_probs[batch_idx]
            seq_len = lengths[batch_idx].item()
            beams: List[Tuple[List[int], float]] = [([], 0.0)]
            for t in range(seq_len):
                new_beams: Dict[Tuple[int, ...], float] = {}
                for seq, seq_log_prob in beams:
                    for token in range(c):
                        token_log_prob = log_probs_batch[t, token].item()
                        new_log_prob = seq_log_prob + token_log_prob
                        new_seq = seq.copy()

                        if token != self.blank_id:
                            if new_seq and new_seq[-1] == token:
                                new_beams[tuple(new_seq)] = new_beams.get(tuple(new_seq),
                                                                          float('-inf')) + token_log_prob
                            else:
                                new_seq.append(token)
                                new_beams[tuple(new_seq)] = new_log_prob
                        else:
                            new_beams[tuple(new_seq)] = new_beams.get(tuple(new_seq),
                                                                      float('-inf')) + token_log_prob
                beams = []
                for seq, log_probs_batch in sorted(new_beams.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True)[:beam_width]:
                    beams.append([list(seq), log_probs_batch])
            best_seq, best_log_prob = beams[0]
            text = "".join(self.vocabulary[tok] for tok in best_seq)
            pred_texts.append(text)

        print(f"Current text (Beam Search Method, beam_width={beam_width}): {pred_texts}")
        return pred_texts