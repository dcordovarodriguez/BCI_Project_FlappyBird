from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class PreprocessingConfig:
    """
    Preprocessing settings that match the training pipeline in blink_classifier.py.

    selected_channel_indices : which EEG channels to keep (default: [0, 1] matching training)
    bandpass_low              : lower bandpass cutoff in Hz  (default: 1 Hz)
    bandpass_high             : upper bandpass cutoff in Hz  (default: 15 Hz)
    bandpass_order            : Butterworth filter order     (default: 4)
    sampling_rate             : EEG sampling rate in Hz      (default: 250 Hz)
    center_each_channel       : remove DC offset per channel (default: True)
    clip_uv                   : optional hard clip in µV     (default: None)
    """

    selected_channel_indices: Optional[Iterable[int]] = field(
        default_factory=lambda: [0, 1]
    )
    bandpass_low: float = 1.0
    bandpass_high: float = 15.0
    bandpass_order: int = 4
    sampling_rate: float = 250.0
    center_each_channel: bool = True
    clip_uv: Optional[float] = None


class EEGPreprocessor:
    """
    Preprocessing that mirrors the training pipeline exactly:
        1. Channel selection
        2. 1-15 Hz Butterworth bandpass  (matches blink_classifier.py)
        3. Per-channel DC removal
        4. Optional hard clip

    Use the same PreprocessingConfig for both training and live inference
    to avoid train/test mismatch.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._b, self._a = self._design_filter()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _design_filter(self):
        nyq = self.config.sampling_rate / 2.0
        low = self.config.bandpass_low / nyq
        high = self.config.bandpass_high / nyq
        # clamp to valid range just in case config is edited
        low = max(1e-4, min(low, 0.9999))
        high = max(low + 1e-4, min(high, 0.9999))
        b, a = butter(self.config.bandpass_order, [low, high], btype="band")
        return b, a

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, eeg_window: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        eeg_window : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        np.ndarray, shape (n_selected_channels, n_samples)
        """
        x = np.asarray(eeg_window, dtype=float)

        # 1. Channel selection
        if self.config.selected_channel_indices is not None:
            idx = list(self.config.selected_channel_indices)
            x = x[idx, :]

        # 2. Bandpass filter (only if we have enough samples for filtfilt)
        min_samples_for_filter = 3 * (self.config.bandpass_order + 1)
        if x.shape[1] >= min_samples_for_filter:
            x = filtfilt(self._b, self._a, x, axis=1)

        # 3. DC removal
        if self.config.center_each_channel:
            x = x - np.mean(x, axis=1, keepdims=True)

        # 4. Optional clip
        if self.config.clip_uv is not None:
            x = np.clip(x, -float(self.config.clip_uv), float(self.config.clip_uv))

        return x
