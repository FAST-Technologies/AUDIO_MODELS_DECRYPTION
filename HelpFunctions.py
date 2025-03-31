from typing import List
import numpy as np
import onnxruntime as rt

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
                      max_vocab_idx: int
) -> str:
    """
    Perform greedy decoding on CTC log probabilities to produce a transcription.

    Args:
        log_probs (np.ndarray): Log probabilities from the model, shape [batch_size, seq_len, num_classes].
        vocab (List[str]): Vocabulary list mapping token indices to characters/tokens.
        blank_idx (int): Index of the blank token in the vocabulary.
        max_vocab_idx (int): Maximum valid token index (len(vocab) - 1).

    Returns:
        str: The decoded transcription.
    """
    # Ensure batch_size=1 for simplicity
    if log_probs.shape[0] != 1:
        raise ValueError(f"Expected batch_size=1, got {log_probs.shape[0]}")

    # Get the predicted token indices
    token_ids = log_probs.argmax(-1).squeeze().tolist()  # List[int], length=seq_len
    print("Log probs shape:", log_probs.shape)
    print("Predicted token indices:", token_ids)

    # Decode: remove blanks and repeats
    decoded_ids: List[int] = []
    prev_tok = None
    for tok in token_ids:
        if tok > max_vocab_idx:
            print(f"Warning: Token {tok} exceeds VOCAB size ({max_vocab_idx}), skipping")
            continue
        if (tok != prev_tok or prev_tok == blank_idx) and tok != blank_idx:
            decoded_ids.append(tok)
        prev_tok = tok

    # Map token indices to text
    transcription = "".join(vocab[tok] for tok in decoded_ids)
    return transcription