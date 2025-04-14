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
        """
        Initialize the CTC Head module.

        Parameters
        ----------
        feat_in : int
            Number of input features (dimensionality of the encoder output).
        num_classes : int
            Number of output classes (including blank token for CTC).

        Notes
        -----
        Initializes weights and biases using uniform distribution with bounds based on fan-in.
        """
        k = 1.0 / feat_in  # Scaling factor for weight initialization (1/fan_in)
        self.feat_in = feat_in
        self.num_classes = num_classes
        # Initialize weights using uniform distribution with bounds [-sqrt(k), sqrt(k)]
        # Shape: [num_classes, feat_in] - weights for 1x1 convolution (equivalent to matrix multiplication)
        self.weights = np.random.uniform(-np.sqrt(k), np.sqrt(k), (num_classes, feat_in))
        # Initialize bias using uniform distribution with bounds [-sqrt(k), sqrt(k)]
        # Shape: [1, num_classes] - bias for each class
        self.bias = np.random.uniform(-np.sqrt(k), np.sqrt(k), (1, num_classes))

    def forward(self,
                encoder_output: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass of the CTC Head module.

        Parameters
        ----------
        encoder_output : np.ndarray
            Input features from the encoder.
            Shape: [B, T, feat_in], where B is batch size, T is time steps, feat_in is feature dimension.

        Returns
        -------
        np.ndarray
            Log probabilities after log_softmax.
            Shape: [B, T, C], where C is num_classes (including blank token).

        Notes
        -----
        Performs a 1x1 convolution (equivalent to matrix multiplication) followed by log_softmax.
        """
        # Transpose encoder_output to [B, feat_in, T] for matrix multiplication
        # Input shape: [B, T, feat_in]
        # Output shape: [B, feat_in, T]
        x = encoder_output.transpose(0, 2, 1)

        # Perform 1x1 convolution as matrix multiplication: x @ weights + bias
        # x shape: [B, T, feat_in]
        # weights shape: [num_classes, feat_in]
        # Matmul: [B, T, feat_in] @ [feat_in, num_classes] -> [B, T, num_classes]
        # Bias shape: [1, num_classes] (broadcasted to [B, T, num_classes])
        # Output shape: [B, T, num_classes]
        conv_output = np.matmul(x.transpose(0, 2, 1), self.weights) + self.bias

        # Transpose back to [B, T, num_classes] for log_softmax
        conv_output = conv_output.transpose(0, 1, 2)

        # Define log_softmax function for numerical stability
        def log_softmax(x: np.ndarray) -> np.ndarray:
            """
            Compute log_softmax along the last axis.

            Parameters
            ----------
            x : np.ndarray
                Input array.
                Shape: [B, T, C], where C is the number of classes.

            Returns
            -------
            np.ndarray
                Log probabilities.
                Shape: [B, T, C]
            """
            # Compute the maximum value along the last axis (C) for numerical stability
            # Shape: [B, T, 1]
            x_max = np.max(x, axis=-1, keepdims=True)
            # Compute exp(x - x_max) to avoid overflow
            # Shape: [B, T, C]
            exp_x = np.exp(x - x_max)
            # Sum along the last axis to get the denominator
            # Shape: [B, T, 1]
            sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
            # Compute log_softmax: x - x_max - log(sum(exp(x - x_max)))
            # Shape: [B, T, C]
            return x - x_max - np.log(sum_exp_x)

        # Apply log_softmax to the convolution output
        # Input shape: [B, T, num_classes]
        # Output shape: [B, T, num_classes]
        return log_softmax(conv_output)


# Класс жадного декодирования для CTC
class CTCGreedyDecoding_V2_0:
    def __init__(self,
                 vocabulary: List[str],
                 model_path: Optional[str] = None
    ) -> None:
        """
        Initialize the CTC Greedy Decoding module.

        Parameters
        ----------
        vocabulary : List[str]
            List of tokens (e.g., characters or subwords) in the vocabulary.
        model_path : Optional[str], optional
            Path to a SentencePiece model (not supported in this implementation). Defaults to None.

        Raises
        ------
        ValueError
            If model_path is provided (SentencePiece model is not supported).
        """
        self.vocabulary = vocabulary
        # Blank token ID is set to the length of the vocabulary (index after the last token)
        self.blank_id = len(self.vocabulary)
        if model_path is not None:
            raise ValueError("SentencePiece model is not supported in this implementation")

    def decode(self,
               head: CTCHead_V2_0,
               encoded: np.ndarray,
               lengths: np.ndarray
    ) -> List[str]:
        """
        Decode the output of a CTC model into a list of hypotheses using greedy decoding.

        Parameters
        ----------
        head : CTCHead_V2_0
            CTC Head module to compute log probabilities.
        encoded : np.ndarray
            Encoded features from the encoder.
            Shape: [B, T, feat_in], where B is batch size, T is time steps, feat_in is feature dimension.
        lengths : np.ndarray
            Lengths of each sequence in the batch.
            Shape: [B], dtype: int

        Returns
        -------
        List[str]
            List of decoded texts (hypotheses) for each sequence in the batch.

        Notes
        -----
        Uses greedy decoding: at each time step, selects the token with the highest probability,
        removes blanks, and collapses repeated tokens.
        """
        # Compute log probabilities using the CTC Head
        # Input shape: [B, T, feat_in]
        # Output shape: [B, T, C], where C = len(vocabulary) + 1 (including blank token)
        log_probs = head.forward(encoder_output=encoded)

        # Assert that log_probs has the correct shape
        assert (
                len(log_probs.shape) == 3
        ), f"Expected log probs shape {log_probs.shape} == [B, T, C]"
        b, _, c = log_probs.shape
        assert (
                c == len(self.vocabulary) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.vocabulary) + 1}"

        # Get the most probable token at each time step
        # Input shape: [B, T, C]
        # Output shape: [B, T], dtype: int (indices of the most probable tokens)
        labels = np.argmax(log_probs, axis=-1)

        # Create a mask to skip blank tokens
        # Shape: [B, T], dtype: bool
        skip_mask = labels != self.blank_id

        # Create a mask to skip repeated tokens (keep only the first occurrence of consecutive repeats)
        # Compare each time step with the previous one: keep token if it's different
        # Shape: [B, T], dtype: bool
        skip_mask[:, 1:] = np.logical_and(
            skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1]
        )

        # Apply sequence lengths to mask out positions beyond each sequence's length
        # Shape: [B, T], dtype: bool
        for i, length in enumerate(lengths):
            skip_mask[i, length:] = 0

        # Decode each sequence in the batch
        pred_texts: List[str] = []
        for i in range(b):
            # Extract token IDs for the current sequence, skipping blanks and repeats
            # Shape: [num_tokens], dtype: int
            token_ids = labels[i][skip_mask[i]].tolist()
            # Convert token IDs to text by mapping them to the vocabulary
            text = "".join(self.vocabulary[tok] for tok in token_ids)
            pred_texts.append(text)

        print(f"Current text NumPy: {pred_texts}")
        return pred_texts