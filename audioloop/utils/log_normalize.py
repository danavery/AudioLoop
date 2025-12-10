import torchaudio.transforms as T
from torch import nn


class LogNormalize(nn.Module):
    """
    Logarithmic normalization transform for spectrograms.

    Converts spectrogram values to decibel scale and normalizes to [-1, 1] range.
    This transform helps with dynamic range compression and standardization.

    Args:
        stype: Type of input spectrogram - "power" or "magnitude" (default: "power")
        top_db: Minimum negative cut-off in decibels (default: 80)
    """

    def __init__(self, stype="power", top_db=80):
        super().__init__()
        self.db_transform = T.AmplitudeToDB(stype=stype, top_db=top_db)
        self.top_db = top_db

    def forward(self, x):
        """
        Apply log normalization to spectrogram.

        Args:
            x: Input spectrogram tensor

        Returns:
            Normalized spectrogram in range approximately [-1, 1]
        """
        x = x.clamp(min=1e-10)  # Avoid log(0)
        x = self.db_transform(x)  # Convert to decibels (0, top_db]
        x = x - x.max()  # Shift to max 0 (-top_db, 0]
        x = 2 * (x / self.top_db) + 1  # Normalize to [-1, 1]
        return x
