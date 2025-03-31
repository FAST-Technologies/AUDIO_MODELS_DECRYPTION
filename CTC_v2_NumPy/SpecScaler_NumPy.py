import numpy as np

from Constants import Constants
MY_CONSTANTS = Constants()

# Рефакторед
class SpecScaler_V2_0:
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self,
                x: np.ndarray
    ) -> np.ndarray:
        return np.log(np.clip(x, 1e-9, 1e9))

    def __call__(self,
                 x: np.ndarray
    ) -> np.ndarray:
        """Делает класс вызываемым, перенаправляя вызов на forward."""
        return self.forward(x)
