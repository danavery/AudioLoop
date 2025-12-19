import warnings

# Suppress overly aggressive torchaudio mel filterbank warnings
warnings.filterwarnings(
    "ignore",
    message=r".*At least one mel filterbank has all zero values.*",
    module=r"torchaudio\.functional\.functional",
)
