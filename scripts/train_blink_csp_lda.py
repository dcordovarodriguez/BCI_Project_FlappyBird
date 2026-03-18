import os
import glob
import joblib
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# =========================
# SETTINGS
# =========================
folder_path = "data/blink_task/sub-37/ses-01/"
sampling_rate = 250

# blink prompt lasts 1.0 s in run_vep.py
blink_window_sec = 1.0

# use a non-blink segment during "Relax" after the blink period
# skip a short gap after blink cue to avoid spillover artifact
rest_offset_sec = 0.25
rest_window_sec = 1.0

# band for blink detection
band_low = 1.0
band_high = 10.0

# number of CSP pairs (total features = 2 * n_csp_pairs)
n_csp_pairs = 2

# model save path
model_path = "model.joblib"


# =========================
# HELPERS
# =========================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_epoch(epoch, fs, lowcut=1.0, highcut=10.0, order=4):
    epoch = np.nan_to_num(epoch, nan=0.0, posinf=0.0, neginf=0.0)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    out = filtfilt(b, a, epoch, axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def average_covariance(X, eps=1e-6):
    # X shape: [n_trials, n_channels, n_samples]
    covs = []
    for trial in X:
        trial = np.nan_to_num(trial, nan=0.0, posinf=0.0, neginf=0.0)
        c = np.cov(trial)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        tr = np.trace(c)
        if not np.isfinite(tr) or abs (tr) < eps:
            continue

        c = c / (tr + eps)
        c = 0.5 * (c + c.T)
        c = c + eps * np.eye(c.shape[0])

        covs.append(c)

    if len(covs) == 0:
        raise RuntimeError("No valid covariance matrices were computed.")

    C = np.mean(covs, axis=0)
    C = 0.5 * (C + C.T)
    C = C + eps * np.eye(C.shape[0])
    return C


class CSP:
    def __init__(self, n_pairs=2):
        self.n_pairs = n_pairs
        self.W = None

    def fit(self, X, y, eps=1e-6):
        # binary classes assumed: 0 and 1
        X0 = X[y == 0]
        X1 = X[y == 1]

        if len(X0) == 0 or len(X1) == 0:
            raise RuntimeError("Both classes need at least one trial - Diego.")

        C0 = average_covariance(X0, eps=eps)
        C1 = average_covariance(X1, eps=eps)

        C = C0 + C1
        C = 0.5 * (C + C.T)
        C = C + eps * np.eye(C.shape[0])

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        eigvals = np.clip(eigvals, eps, None)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        P = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        S0 = P @ C0 @ P.T
        S0 = 0.5 * (S0 + S0.T)
        S0 = S0 + eps * np.eye(S0.shape[0])

        d, B = np.linalg.eigh(S0)
        d = np.real(d)
        B = np.real(B)

        idx = np.argsort(d)[::-1]
        B = B[:, idx]

        W = B.T @ P

        # take top and bottom filters
        filters = []
        for i in range(self.n_pairs):
            filters.append(W[i, :])
        for i in range(self.n_pairs):
            filters.append(W[-(i + 1), :])

        self.W = np.stack(filters, axis=0)
        return self

    def transform(self, X):
        # X shape: [n_trials, n_channels, n_samples]
        Z = np.einsum("fc,tcs->tfs", self.W, X)  # [trials, filters, samples]
        var = np.var(Z, axis=2)
        var = var / np.sum(var, axis=1, keepdims=True)
        feats = np.log(var + 1e-12)
        return feats


def load_events(path):
    # saved as np.array(events, dtype=object)
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        return list(arr)
    return arr


def find_event_times(events, event_name):
    times = []
    for ev in events:
        if isinstance(ev, dict) and ev.get("event") == event_name:
            times.append(float(ev["time"]))
    return times


def build_epochs_from_run(eeg, events, fs):
    """
    Uses event times from run_vep.py.
    Assumes event times are relative to task timeline and maps them
    onto the continuous EEG by proportion of total task duration.

    This is an approximation because run_vep.py stores event time from
    PsychoPy's global clock and EEG timestamps separately. However,
    because the task timing is fixed and data are continuous, this is a
    practical way to create training windows from your saved runs.
    """
    # total task duration from events
    event_times = [float(ev["time"]) for ev in events if isinstance(ev, dict) and "time" in ev]
    if len(event_times) == 0:
        return [], []

    t0 = min(event_times)
    t1 = max(event_times)
    total_event_duration = t1 - t0
    if total_event_duration <= 0:
        return [], []

    n_samples = eeg.shape[1]
    total_eeg_duration = n_samples / fs

    blink_times = find_event_times(events, "blink_now")
    rest_times = find_event_times(events, "rest")

    X = []
    y = []

    # make sure trials align by order
    n_trials = min(len(blink_times), len(rest_times))

    blink_len = int(blink_window_sec * fs)
    rest_offset = int(rest_offset_sec * fs)
    rest_len = int(rest_window_sec * fs)

    def time_to_sample(task_time):
        # map event time within task to eeg sample index
        rel = (task_time - t0) / total_event_duration
        rel = np.clip(rel, 0.0, 1.0)
        return int(rel * (n_samples - 1))

    for i in range(n_trials):
        blink_start = time_to_sample(blink_times[i])
        rest_start = time_to_sample(rest_times[i])

        # blink epoch
        b0 = blink_start
        b1 = b0 + blink_len

        # non-blink epoch during relax
        r0 = rest_start + rest_offset
        r1 = r0 + rest_len

        if b1 <= n_samples and r1 <= n_samples:
            blink_epoch = eeg[:, b0:b1]
            rest_epoch = eeg[:, r0:r1]

            if blink_epoch.shape[1] == blink_len and rest_epoch.shape[1] == rest_len:
                X.append(blink_epoch)
                y.append(1)
                X.append(rest_epoch)
                y.append(0)

    return X, y


def load_dataset(folder_path, fs):
    eeg_files = sorted(glob.glob(os.path.join(folder_path, "eeg_run-*.npy")))
    event_files = sorted(glob.glob(os.path.join(folder_path, "events_run-*.npy")))

    if len(eeg_files) == 0:
        raise FileNotFoundError(f"No eeg_run-*.npy files found in {folder_path}")
    if len(event_files) == 0:
        raise FileNotFoundError(f"No events_run-*.npy files found in {folder_path}")

    X_all = []
    y_all = []
    run_ids = []

    for eeg_file, event_file in zip(eeg_files, event_files):
        eeg = np.load(eeg_file)  # [channels, samples]
        events = load_events(event_file)

        X_run, y_run = build_epochs_from_run(eeg, events, fs)

        for x, y in zip(X_run, y_run):
            X_all.append(x)
            y_all.append(y)
            run_ids.append(os.path.basename(eeg_file))

    if len(X_all) == 0:
        raise RuntimeError("No epochs were extracted. Check event timing / run files.")

    X_all = np.stack(X_all, axis=0)  # [trials, channels, samples]
    y_all = np.array(y_all)
    run_ids = np.array(run_ids)

    return X_all, y_all, run_ids


def leave_one_run_out_eval(X, y, run_ids, fs):
    unique_runs = np.unique(run_ids)
    all_true = []
    all_pred = []

    for test_run in unique_runs:
        train_mask = run_ids != test_run
        test_mask = run_ids == test_run

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # bandpass
        X_train_f = np.stack([bandpass_epoch(ep, fs, band_low, band_high) for ep in X_train], axis=0)
        X_test_f = np.stack([bandpass_epoch(ep, fs, band_low, band_high) for ep in X_test], axis=0)

        csp = CSP(n_pairs=n_csp_pairs).fit(X_train_f, y_train)
        F_train = csp.transform(X_train_f)
        F_test = csp.transform(X_test_f)

        clf = LinearDiscriminantAnalysis()
        clf.fit(F_train, y_train)
        y_pred = clf.predict(F_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"Test run: {test_run}")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"  Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print("\n=== Overall leave-one-run-out results ===")
    print(f"Accuracy: {accuracy_score(all_true, all_pred):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(all_true, all_pred):.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(all_true, all_pred))

    return all_true, all_pred


def fit_final_model(X, y, fs):
    X_f = np.stack([bandpass_epoch(ep, fs, band_low, band_high) for ep in X], axis=0)

    csp = CSP(n_pairs=n_csp_pairs).fit(X_f, y)
    F = csp.transform(X_f)

    clf = LinearDiscriminantAnalysis()
    clf.fit(F, y)

    model = {
        "fs": fs,
        "band": (band_low, band_high),
        "blink_window_sec": blink_window_sec,
        "rest_window_sec": rest_window_sec,
        "rest_offset_sec": rest_offset_sec,
        "n_csp_pairs": n_csp_pairs,
        "csp": csp,
        "clf": clf,
    }
    return model


def main():
    print(f"Loading dataset from: {folder_path}")
    X, y, run_ids = load_dataset(folder_path, sampling_rate)

    print("X shape:", X.shape)
    print("NaN in X:", np.isnan(X).any())
    print("Inf in X:", np.isinf(X).any())
    print("Min trial variance:", np.min(np.var(X, axis=2)))
    print("Max trial variance:", np.max(np.var(X, axis=2)))


    print(f"Extracted epochs: {len(X)}")
    print(f"Blink trials: {np.sum(y == 1)}")
    print(f"Non-blink trials: {np.sum(y == 0)}")
    print(f"Channels: {X.shape[1]}, Samples/epoch: {X.shape[2]}")

    leave_one_run_out_eval(X, y, run_ids, sampling_rate)

    model = fit_final_model(X, y, sampling_rate)
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()

