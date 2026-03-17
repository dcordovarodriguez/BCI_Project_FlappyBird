import os
import sys
import time
import glob
import joblib
import numpy as np
import serial

from serial import Serial
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ======================
# BASIC SETTINGS
# ======================
FS = 250
EEG_CH = [0, 1, 2, 3]   # FP1, FP2, F3, F4
AUX_CH = 1
THRESH = 50

MODEL_FILE = "blink_model.joblib"


# ======================
# FILTER
# ======================
def bandpass(x, low=1, high=15, fs=FS):
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=1)


# ======================
# FIND BLINK TIMES (photosensor)
# ======================
def get_blink_windows(aux):
    b = (aux > THRESH).astype(int)
    starts = np.where(np.diff(b) == 1)[0] + 1
    ends = np.where(np.diff(b) == -1)[0] + 1

    out = []
    j = 0
    for s in starts:
        while j < len(ends) and ends[j] <= s:
            j += 1
        if j < len(ends):
            out.append((s, ends[j]))
            j += 1
    return out


# ======================
# MAKE DATASET
# ======================
def make_epochs(eeg, aux):
    eeg = eeg[EEG_CH, :]
    eeg = bandpass(eeg)

    aux = aux[AUX_CH]
    pairs = get_blink_windows(aux)

    blink = []
    noblink = []

    for s, _ in pairs:
        a = s - int(0.2 * FS)
        b = s + int(0.8 * FS)

        c = s + int(1.3 * FS)
        d = s + int(2.3 * FS)

        if a < 0 or b > eeg.shape[1]: continue
        if c < 0 or d > eeg.shape[1]: continue

        e1 = eeg[:, a:b]
        e2 = eeg[:, c:d]

        e1 = e1 - e1.mean(axis=1, keepdims=True)
        e2 = e2 - e2.mean(axis=1, keepdims=True)

        blink.append(e1)
        noblink.append(e2)

    X = np.array(blink + noblink)
    y = np.array([1]*len(blink) + [0]*len(noblink))

    return X, y


# ======================
# ICA + LDA
# ======================
def train_model(X, y):
    n_ep, n_ch, n_t = X.shape

    # fit ICA on all training data
    data = np.transpose(X, (1,0,2)).reshape(n_ch, -1).T
    ica = FastICA(n_components=n_ch, random_state=0, max_iter=2000)
    ica.fit(data)

    # transform each epoch
    S = np.array([ica.transform(ep.T).T for ep in X])

    # pick component that best separates blink vs no-blink
    scores = []
    for i in range(n_ch):
        m1 = S[y==1, i, :].mean()
        m0 = S[y==0, i, :].mean()
        scores.append(abs(m1 - m0))

    best = int(np.argmax(scores))

    # simple features
    def feats(S):
        x = S[:, best, :]
        return np.column_stack([
            x.max(axis=1),
            x.min(axis=1),
            x.ptp(axis=1),
            x.std(axis=1),
        ])

    Xf = feats(S)

    scaler = StandardScaler()
    Xf = scaler.fit_transform(Xf)

    clf = LinearDiscriminantAnalysis()
    clf.fit(Xf, y)

    return {
        "ica": ica,
        "best": best,
        "scaler": scaler,
        "clf": clf
    }


def predict(model, ep):
    ica = model["ica"]
    best = model["best"]
    scaler = model["scaler"]
    clf = model["clf"]

    S = ica.transform(ep.T).T
    x = S[best]

    feat = np.array([
        x.max(),
        x.min(),
        x.ptp(),
        x.std()
    ]).reshape(1, -1)

    feat = scaler.transform(feat)
    p = clf.predict_proba(feat)[0,1]

    return int(p > 0.7), p


# ======================
# LOAD DATA
# ======================
def load_all(folder):
    X_all = []
    y_all = []

    for f in os.listdir(folder):
        if f.startswith("eeg_run"):
            run = f.split("-")[1].split(".")[0]

            eeg = np.load(os.path.join(folder, f))
            aux = np.load(os.path.join(folder, f"aux_run-{run}.npy"))

            X, y = make_epochs(eeg, aux)
            X_all.append(X)
            y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)


# ======================
# CYTON PORT
# ======================
def find_port():
    for i in range(1, 256):
        port = f"COM{i}"
        try:
            s = Serial(port, 115200, timeout=1)
            s.write(b'v')
            time.sleep(1)
            data = s.read(2000).decode(errors='ignore')
            s.close()
            if "OpenBCI" in data:
                return port
        except:
            pass
    raise Exception("Cyton not found")


# ======================
# LIVE MODE
# ======================
def run_live(model):
    from brainflow.board_shim import BoardShim, BrainFlowInputParams

    params = BrainFlowInputParams()
    params.serial_port = find_port()

    board = BoardShim(0, params)
    board.prepare_session()
    board.start_stream()

    print("running... ctrl+c to stop")

    try:
        while True:
            data = board.get_board_data()
            if data.shape[1] < FS:
                continue

            eeg = data[board.get_eeg_channels(0), :]
            eeg = eeg[EEG_CH, -FS:]

            pred, prob = predict(model, eeg)

            if pred:
                print("BLINK", round(prob,3))
            else:
                print("no", round(prob,3))

            time.sleep(0.2)

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "train":
        folder = sys.argv[2]

        X, y = load_all(folder)
        model = train_model(X, y)

        joblib.dump(model, MODEL_FILE)
        print("saved:", MODEL_FILE)

    elif mode == "live":
        model = joblib.load(MODEL_FILE)
        run_live(model)
