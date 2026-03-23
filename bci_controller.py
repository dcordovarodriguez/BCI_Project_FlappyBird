from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from classifier_interface import BaseLiveClassifier, BlinkModelClassifier, PlaceholderClassifier
from cyton_stream import CytonStreamConfig, make_stream
from preprocessing import EEGPreprocessor, PreprocessingConfig


@dataclass
class BCIControllerConfig:
    """
    Top-level config for the BCI controller.

    Live mode (default):
        Leave offline_eeg_path as None. Set serial_port and use_synthetic_board
        as appropriate.

    Offline replay mode:
        Set offline_eeg_path to a .npy EEG file of shape (n_channels, n_samples).
        Optionally set offline_timestamps_path for accurate timestamps.
        The stream replays the file in real time by default; set
        realtime_replay=False to process as fast as possible.

    Parameters
    ----------
    serial_port             : COM port for live Cyton (e.g. "COM3" or "/dev/ttyUSB0")
    use_synthetic_board     : use BrainFlow's synthetic board (no hardware needed)
    sampling_rate           : samples/sec — used in offline mode and preprocessing
    window_seconds          : EEG window length fed to the classifier
                              Must match the epoch length used during training (0.8 s)
    cooldown_seconds        : minimum gap between two consecutive jumps
    blink_threshold         : classifier probability threshold for triggering a jump
    model_path              : path to the trained blink_model.joblib
    enabled                 : set False to disable BCI entirely (keyboard-only mode)
    offline_eeg_path        : .npy file for offline replay (None = live)
    offline_timestamps_path : optional matching timestamps .npy for offline replay
    realtime_replay         : whether offline replay respects original timing
    """
    serial_port: str = "COM3"
    use_synthetic_board: bool = False
    sampling_rate: float = 250.0
    window_seconds: float = 0.8          # must match training epoch length
    cooldown_seconds: float = 0.35
    blink_threshold: float = 0.5
    model_path: str = "blink_model.joblib"
    enabled: bool = True

    # Offline replay switch
    offline_eeg_path: Optional[str] = None
    offline_timestamps_path: Optional[str] = None
    realtime_replay: bool = True


class BCIController:
    """
    Bridge between the EEG stream and the game.

    Public API used by flappy.py:
        start()
        stop()
        should_jump() -> bool

    Internally:
        1. Pull the newest EEG window from the stream (live or offline)
        2. Preprocess with bandpass + DC removal (matching training pipeline)
        3. Extract features and classify with the trained LDA blink model
        4. Apply cooldown
        5. Return True only when the bird should flap

    Selecting live vs offline:
        Pass offline_eeg_path in BCIControllerConfig (or via CLI — see bottom
        of this file) to replay a recorded .npy file instead of connecting to
        the Cyton board.
    """

    def __init__(
        self,
        config: Optional[BCIControllerConfig] = None,
        classifier: Optional[BaseLiveClassifier] = None,
    ):
        self.config = config or BCIControllerConfig()
        self.enabled = self.config.enabled
        self.last_jump_time: float = 0.0

        # Build stream config
        stream_cfg = CytonStreamConfig(
            serial_port=self.config.serial_port,
            sampling_rate=self.config.sampling_rate,
            window_seconds=self.config.window_seconds,
            use_synthetic_board=self.config.use_synthetic_board,
            offline_eeg_path=self.config.offline_eeg_path,
            offline_timestamps_path=self.config.offline_timestamps_path,
            realtime_replay=self.config.realtime_replay,
            eeg_channels=[0, 1],   # frontal channels used in training
        )
        self.stream = make_stream(stream_cfg)

        # Preprocessing — matches training: 1-15 Hz bandpass + DC removal
        prep_cfg = PreprocessingConfig(
            selected_channel_indices=[0, 1],
            bandpass_low=1.0,
            bandpass_high=15.0,
            bandpass_order=4,
            sampling_rate=self.config.sampling_rate,
            center_each_channel=True,
        )
        self.preprocessor = EEGPreprocessor(prep_cfg)

        # Classifier — use the trained model unless a custom one is injected
        if classifier is not None:
            self.classifier = classifier
        else:
            try:
                self.classifier = BlinkModelClassifier(
                    model_path=self.config.model_path,
                    threshold=self.config.blink_threshold,
                )
            except Exception as e:
                print(f"[BCIController] WARNING: Could not load model ({e}). "
                      f"Falling back to PlaceholderClassifier.")
                self.classifier = PlaceholderClassifier()

        mode = "OFFLINE REPLAY" if self.config.offline_eeg_path else "LIVE"
        print(f"[BCIController] Mode: {mode} | "
              f"Classifier: {type(self.classifier).__name__} | "
              f"Window: {self.config.window_seconds}s | "
              f"Threshold: {self.config.blink_threshold}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            return
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()

    def should_jump(self) -> bool:
        """
        Returns True if a blink was detected and cooldown has elapsed.
        Call this once per game frame.
        """
        if not self.enabled:
            return False

        window = self.stream.get_latest_window()
        if window is None:
            return False

        processed = self.preprocessor.transform(window)
        result = self.classifier.predict_window(processed)

        if result.jump_detected and self._cooldown_ready():
            self.last_jump_time = time.time()
            return True

        return False

    @property
    def is_offline_exhausted(self) -> bool:
        """True when offline replay has finished. Always False in live mode."""
        from cyton_stream import _OfflineStream
        if isinstance(self.stream, _OfflineStream):
            return self.stream.is_exhausted
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cooldown_ready(self) -> bool:
        return (time.time() - self.last_jump_time) >= self.config.cooldown_seconds


# ---------------------------------------------------------------------------
# Convenience: build a BCIController directly from command-line args or kwargs
# ---------------------------------------------------------------------------

def make_bci_controller(
    offline_eeg_path: Optional[str] = None,
    offline_timestamps_path: Optional[str] = None,
    serial_port: str = "COM3",
    use_synthetic_board: bool = False,
    realtime_replay: bool = True,
    blink_threshold: float = 0.5,
    model_path: str = "blink_model.joblib",
    enabled: bool = True,
) -> BCIController:
    """
    Convenience factory.  Example usage:

        # Live mode
        ctrl = make_bci_controller()

        # Offline replay
        ctrl = make_bci_controller(offline_eeg_path="my_eeg.npy")

        # Offline, fast (non-realtime)
        ctrl = make_bci_controller(offline_eeg_path="my_eeg.npy", realtime_replay=False)
    """
    cfg = BCIControllerConfig(
        serial_port=serial_port,
        use_synthetic_board=use_synthetic_board,
        blink_threshold=blink_threshold,
        model_path=model_path,
        enabled=enabled,
        offline_eeg_path=offline_eeg_path,
        offline_timestamps_path=offline_timestamps_path,
        realtime_replay=realtime_replay,
    )
    return BCIController(config=cfg)
