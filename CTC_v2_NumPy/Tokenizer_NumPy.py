from typing import List, Optional
from sentencepiece import SentencePieceProcessor

class Tokenizer_V2_0:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """
    def __init__(self,
                 vocab: List[str],
                 model_path: Optional[str] = None
    ) -> None:
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            # используется аргумент из библиотеки SentencePieceProcessor
            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self,
               tokens: List[int]
    ) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)

    def __len__(self) -> None:
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)