import torchaudio.transforms as T
from torch import nn


class LogNormalize(nn.Module):
    def __init__(self, stype="power", top_db=80):
        super().__init__()
        self.db_transform = T.AmplitudeToDB(stype=stype, top_db=top_db)
        self.top_db = top_db

    def forward(self, x):
        x = x.clamp(min=1e-10)  # Avoid log(0)
        x = self.db_transform(x)  # Convert to decibels (0, top_db]
        x = x - x.max()  # shift to max 0 (-top_db, 0]
        x = 2 * (x / self.top_db) + 1 #  normalize to [-1, 1]
        return x
