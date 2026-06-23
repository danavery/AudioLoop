import torch
import torch.nn as nn

from .audio_loop_model import AudioLoopModel


class LinearProbe(AudioLoopModel):
    """Linear classifier over frozen 1D embeddings — the standard linear-probe protocol.

    A single nn.Linear on top of a frozen feature extractor's pooled embedding (e.g. a
    (768,) wav2vec2/AVES vector). Because the backbone is frozen and cached upstream, this
    head alone is trained, measuring how linearly separable the representation is.

    The Linear is sized EAGERLY at construction (not nn.LazyLinear): the training pipeline
    counts parameters and constructs the optimizer before any forward pass, which a lazy
    module — whose params don't exist until first forward — cannot satisfy. The input
    dimension therefore must be known at __init__ time.
    """

    description = "Linear probe over frozen 1D embeddings"

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # in_features sizes the Linear. Two provenance paths:
        #  - Train time: create_model passes dataset_shape from the feature extractor.
        #  - Reload time (load_model): no extractor exists; in_features is read back from the
        #    checkpoint, which persists it via get_model_info(). Whichever is supplied wins.
        in_features = kwargs.get("in_features")
        if in_features is None:
            dataset_shape = kwargs.get("dataset_shape")
            if dataset_shape is None:
                raise ValueError(
                    "LinearProbe requires `in_features` or `dataset_shape` to size its Linear layer."
                )
            in_features = dataset_shape[0]
        self.in_features = in_features

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, in_features): embeddings are pooled to one vector per file upstream, and
        # variable_length_collate_fn stacks the equal-length vectors with no padding.
        return self.fc(x)

    def get_model_info(self) -> dict:
        # in_features is part of the construction contract: get_model_info() is what the
        # checkpoint persists and load_model() reconstructs from, so the layer-sizing param
        # must travel through here for inference to rebuild an identically-shaped Linear.
        return {
            "model_type": "linearprobe",
            "num_classes": self.num_classes,
            "in_features": self.in_features,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        # 1D features only (a pooled embedding like (768,)); rejects 2D spectrograms.
        return len(shape) == 1
