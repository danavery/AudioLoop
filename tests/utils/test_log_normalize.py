"""Tests for LogNormalize: the dB-conversion + normalization step in the spectrogram
pipeline.

The extractor tests stub _create_transform (stub-the-edge pattern), so this module is the
transform component that needs its own direct test — otherwise the actual normalization
math is the one link in the feature pipeline no test touches.
"""

import torch

from audioloop.utils.log_normalize import LogNormalize


def test_output_range_and_max_anchor():
    """Output lands in [-1, 1] with the loudest bin mapped exactly to 1.0 (the max-shift
    anchors the top of the range regardless of absolute input scale)."""
    transform = LogNormalize()

    out = transform(torch.rand(1, 128, 50) * 1000)

    assert out.min() >= -1.0
    assert out.max() == torch.tensor(1.0)
    assert torch.isfinite(out).all()


def test_known_values_power_scale():
    """Power scale: 10*log10. A bin 1e8 below the max is -80 dB, hits the top_db clamp,
    and maps exactly to -1; the max maps to +1; -40 dB lands exactly at 0."""
    transform = LogNormalize(stype="power", top_db=80)

    # (freq, time)-shaped: AmplitudeToDB requires at least 2D, as real spectrograms are.
    out = transform(torch.tensor([[1.0, 1e-4, 1e-8]]))

    assert torch.allclose(out, torch.tensor([[1.0, 0.0, -1.0]]), atol=1e-5)


def test_known_values_magnitude_scale():
    """Magnitude scale: 20*log10, so -80 dB is reached at 1e-4 of the max instead."""
    transform = LogNormalize(stype="magnitude", top_db=80)

    out = transform(torch.tensor([[1.0, 1e-2, 1e-4]]))

    assert torch.allclose(out, torch.tensor([[1.0, 0.0, -1.0]]), atol=1e-5)


def test_zeros_are_clamped_not_log_of_zero():
    """All-zero input (silence) must produce finite output via the 1e-10 clamp."""
    out = LogNormalize()(torch.zeros(1, 128, 10))

    assert torch.isfinite(out).all()
    # Every bin equals the max, so everything normalizes to the top of the range.
    assert torch.allclose(out, torch.ones_like(out))


def test_normalization_is_scale_invariant():
    """Multiplying the input by a constant must not change the output: the max-shift
    cancels absolute level, which is what makes cached features comparable across clips
    recorded at different gains."""
    transform = LogNormalize()
    spec = torch.rand(128, 30) + 0.01

    assert torch.allclose(transform(spec), transform(spec * 123.4), atol=1e-5)


def test_monotonicity_preserved():
    """Louder bins stay larger after normalization (log is monotone)."""
    transform = LogNormalize()
    out = transform(torch.tensor([[0.001, 0.01, 0.1, 1.0]]))[0]

    assert torch.all(out[1:] > out[:-1])
