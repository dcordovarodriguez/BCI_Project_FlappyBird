from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
except ImportError:
    BoardIds = None
    BoardShim = None
    BrainFlowInputParams = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CytonStreamConfig:
    """
    Configuration for EEG data source.

    Live mode  (default):
        Set use_synthetic_board=False and provide serial_port.
        Set use_synthetic_board=True to use BrainFlow's built-in synthetic board
        (no hardware needed, but still requires brainflow installed).

    Offline replay mode:
        Set offline_eeg_path to a .npy file of shape (n_channels, n_samples).
        Optional offline_timestamps_path for real timestamps; otherwise uniform
        timestamps are generated from sampling_rate.
        The stream will replay the file in real time (or as fast as possible if
        realtime_replay=False).

    Parameters
    ----------
    serial_port             : COM port for Cyton (live only)
    board_id                : explicit BrainFlow board ID (None = auto)
    sampling_rate           : samples per second (used in offline mode; live mode
                              reads this from the board)
    window_seconds          : length of each data window returned
    startup_buffer_seconds  : live only — wait this long after start() before
                              data is considered valid
    use_synthetic_board     : live mode using BrainFlow synthetic board
    offline_eeg_path        : path to .npy EEG file for offline replay
                              shape must be (n_channels, n_samples)
    offline_timestamps_path : optional path to matching timestamps .npy (1-D)
    realtime_replay         : if True, offline mode sleeps to mimic real time
    """

    serial_port: str = "COM3"
    board_id: Optional[int] = None
    sampling_rate: float = 250.0
    window_seconds: float = 0.8
    startup_buffer_seconds: float = 2.0
    use_synthetic_board: bool = False

    # Offline replay
    offline_eeg_path: Optional[str] = None
    offline_timestamps_path: Optional[str] = None
    realtime_replay: bool = True

    # Derived / read-only
    eeg_channels: List[int] = field(default_factory=lambda: [0, 1])

    @property
    def is_offline(self) -> bool:
        return self.offline_eeg_path is not None

    def resolved_board_id(self) -> int:
        if self.use_synthetic_board:
            if BoardIds is None:
                raise ImportError("brainflow is not installed")
            return BoardIds.SYNTHETIC_BOARD.value
        if self.board_id is not None:
            return self.board_id
        if BoardIds is None:
            raise ImportError("brainflow is not installed")
        return BoardIds.CYTON_BOARD.value


# ---------------------------------------------------------------------------
# Offline replay stream
# ---------------------------------------------------------------------------

class _OfflineStream:
    """
    Replays a pre-recorded EEG .npy file as if it were a live stream.

    The public interface mirrors CytonStream so BCIController can use either
    without any changes.
    """

    def __init__(self, config: CytonStreamConfig):
        self.config = config
        self._eeg: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None
        self._cursor: int = 0
        self._start_wall: float = 0.0
        self._started: bool = False
        self.sampling_rate: float = config.sampling_rate
        self.eeg_channels: List[int] = config.eeg_channels
        self.window_size_samples: int = 0

    def start(self) -> None:
        if self._started:
            return

        eeg = np.load(self.config.offline_eeg_path)   # (n_ch, n_samples)
        if eeg.ndim != 2:
            raise ValueError(
                f"Offline EEG file must be shape (n_channels, n_samples), got {eeg.shape}"
            )

        self._eeg = eeg

        if self.config.offline_timestamps_path is not None:
            self._timestamps = np.load(self.config.offline_timestamps_path)
        else:
            n_samples = eeg.shape[1]
            self._timestamps = np.arange(n_samples) / self.sampling_rate

        self.window_size_samples = max(1, int(self.config.window_seconds * self.sampling_rate))
        self._cursor = self.window_size_samples   # start so first window is valid
        self._start_wall = time.time()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_latest_window(self) -> Optional[np.ndarray]:
        if not self._started or self._eeg is None:
            return None

        n_total = self._eeg.shape[1]

        if self.config.realtime_replay:
            # Advance cursor based on elapsed wall time
            elapsed = time.time() - self._start_wall
            new_cursor = min(
                int(elapsed * self.sampling_rate) + self.window_size_samples,
                n_total,
            )
            self._cursor = new_cursor
        else:
            # Advance one window per call
            self._cursor = min(self._cursor + self.window_size_samples, n_total)

        if self._cursor < self.window_size_samples:
            return None

        start = self._cursor - self.window_size_samples
        window = self._eeg[self.eeg_channels, start : self._cursor]

        if window.shape[1] < self.window_size_samples:
            return None

        return window

    @property
    def is_exhausted(self) -> bool:
        """True once the entire file has been replayed."""
        if self._eeg is None:
            return False
        return self._cursor >= self._eeg.shape[1]


# ---------------------------------------------------------------------------
# Live Cyton stream
# ---------------------------------------------------------------------------

class CytonStream:
    """
    Thin wrapper around BrainFlow for live Cyton data.

    Output shape from get_latest_window(): (n_channels, n_samples)
    """

    def __init__(self, config: CytonStreamConfig):
        self.config = config
        self.board = None
        self.sampling_rate: Optional[float] = None
        self.eeg_channels: List[int] = []
        self.window_size_samples: Optional[int] = None
        self.started = False

    def start(self) -> None:
        if BoardShim is None or BrainFlowInputParams is None:
            raise ImportError(
                "brainflow is required for live Cyton streaming. "
                "Install with: pip install brainflow"
            )

        if self.started:
            return

        board_id = self.config.resolved_board_id()
        params = BrainFlowInputParams()
        if not self.config.use_synthetic_board:
            params.serial_port = self.config.serial_port

        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

        self.sampling_rate = float(BoardShim.get_sampling_rate(board_id))
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.window_size_samples = max(1, int(self.config.window_seconds * self.sampling_rate))
        self.started = True

        startup_wait = max(self.config.window_seconds, self.config.startup_buffer_seconds)
        time.sleep(startup_wait)

    def stop(self) -> None:
        if not self.board:
            self.started = False
            return
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass
        self.board = None
        self.started = False

    def get_latest_window(self) -> Optional[np.ndarray]:
        if not self.started or self.board is None or self.window_size_samples is None:
            return None

        data = self.board.get_current_board_data(self.window_size_samples)
        if data.size == 0:
            return None

        eeg = data[self.eeg_channels, :]
        if eeg.shape[1] < self.window_size_samples:
            return None

        return eeg


# ---------------------------------------------------------------------------
# Factory — returns either a live or offline stream
# ---------------------------------------------------------------------------

def make_stream(config: CytonStreamConfig) -> "CytonStream | _OfflineStream":
    """
    Returns an _OfflineStream if config.offline_eeg_path is set,
    otherwise returns a live CytonStream.

    Both share the same interface: start(), stop(), get_latest_window().
    """
    if config.is_offline:
        return _OfflineStream(config)
    return CytonStream(config)
