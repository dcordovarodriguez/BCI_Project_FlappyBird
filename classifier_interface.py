from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClassifierOutput:
    jump_detected: bool
    score: Optional[float] = None   # probability of blink (0-1)


class BaseLiveClassifier:
    """
    Interface for a live EEG classifier.

    Contract:
        input  -> preprocessed EEG window, shape (channels, samples)
        output -> ClassifierOutput
    """

    def predict_window(self, eeg_window: np.ndarray) -> ClassifierOutput:
        raise NotImplementedError


class PlaceholderClassifier(BaseLiveClassifier):
    """No-op classifier — always returns no-jump. Useful for testing the game loop."""

    def predict_window(self, eeg_window: np.ndarray) -> ClassifierOutput:
        return ClassifierOutput(jump_detected=False, score=0.0)


class BlinkModelClassifier(BaseLiveClassifier):
    """
    Wraps the trained blink_model.joblib for live (or offline-replay) inference.

    The model expects exactly 10 features extracted from a 2-channel EEG epoch:
        [max, min, ptp, std, abs_sum]  x  2 channels  =  10 features

    This matches the extract_features() function in blink_classifier.py.

    Parameters
    ----------
    model_path  : path to the saved joblib file
    threshold   : blink probability above which a jump is triggered (default 0.5)
    """

    def __init__(self, model_path: str = "blink_model.joblib", threshold: float = 0.5):
        import joblib
        self.threshold = threshold
        bundle = joblib.load(model_path)
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]

    # ------------------------------------------------------------------
    # Feature extraction — must exactly match blink_classifier.py
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(epoch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        epoch : np.ndarray, shape (2, n_samples)  — already preprocessed

        Returns
        -------
        np.ndarray, shape (10,)
        """
        feats = []
        for ch in epoch:
            feats.extend([
                float(np.max(ch)),
                float(np.min(ch)),
                float(np.ptp(ch)),
                float(np.std(ch)),
                float(np.sum(np.abs(ch))),
            ])
        return np.array(feats, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_window(self, eeg_window: np.ndarray) -> ClassifierOutput:
        """
        Parameters
        ----------
        eeg_window : np.ndarray, shape (2, n_samples)
                     Already preprocessed (bandpass + DC removal done upstream).

        Returns
        -------
        ClassifierOutput with jump_detected=True when blink probability > threshold
        """
        feats = self._extract_features(eeg_window).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        prob = float(self.clf.predict_proba(feats_scaled)[0, 1])
        return ClassifierOutput(jump_detected=prob > self.threshold, score=prob)
