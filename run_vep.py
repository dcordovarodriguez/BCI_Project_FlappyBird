from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
import os, time, glob, sys, serial, pickle
from serial import Serial
from threading import Thread, Event
from queue import Queue

# SETTINGS

cyton_in = True
width = 1536
height = 864
sampling_rate = 250
subject = 1
session = 1
run = 1

n_trials = 20
baseline_duration = 2.0
countdown_step = 1.0
blink_duration = 1.0
rest_duration = 1.5

save_dir = f"data/blink_task/sub-{subject:02d}/ses-{session:02d}/"
save_file_eeg = save_dir + f"eeg_run-{run}.npy"
save_file_aux = save_dir + f"aux_run-{run}.npy"
save_file_timestamp = save_dir + f"timestamp_run-{run}.npy"
save_file_events = save_dir + f"events_run-{run}.npy"

# PSYCHOPY WINDOW

kb = keyboard.Keyboard()

window = visual.Window(
    size=[width, height],
    checkTiming=True,
    allowGUI=False,
    fullscr=True,
    useRetina=False,
    color="black"
)

instruction_text = visual.TextStim(
    window,
    text="",
    color="white",
    units="norm",
    height=0.12,
    wrapWidth=1.5
)

trial_text = visual.TextStim(
    window,
    text="",
    color="white",
    units="norm",
    pos=(0, -0.85),
    height=0.05
)

fixation = visual.TextStim(
    window,
    text="+",
    color="white",
    units="norm",
    height=0.15
)

def create_photosensor_dot(size=0.08):
    width, height = window.size
    aspect_ratio = width / height
    return visual.Rect(
        win=window,
        units="norm",
        width=size,
        height=size * aspect_ratio,
        fillColor='black',
        lineWidth=0,
        pos=[1 - size/2, -1 + size/2]
    )

photosensor_dot = create_photosensor_dot()

# OPENBCI / BRAINFLOW

if cyton_in:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams

    CYTON_BOARD_ID = 0
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2'

    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = [f'COM{i+1}' for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Unsupported OS')

        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, serial.SerialException):
                pass

        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
        return openbci_port

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()

    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    print(board.config_board('/0'))
    print(board.config_board('//'))
    print(board.config_board(ANALOGUE_MODE))
    board.start_stream(45000)

    stop_event = Event()
    queue_in = Queue()

    def get_data(queue_in):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)

    cyton_thread = Thread(target=get_data, args=(queue_in,))
    cyton_thread.daemon = True
    cyton_thread.start()


# DATA STORAGE

eeg = np.zeros((8, 0))
aux = np.zeros((3, 0))
timestamp = np.zeros((0,))
events = []   # list of dicts with trial/event timing

global_clock = core.Clock()

def collect_queue_data():
    global eeg, aux, timestamp
    if cyton_in:
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0)

def check_escape():
    keys = kb.getKeys()
    if 'escape' in keys:
        shutdown_and_save()
        core.quit()

def draw_screen(main_stim=None, trial_label="", photo_white=False):
    window.color = "black"
    if main_stim is not None:
        main_stim.draw()
    if trial_label:
        trial_text.text = trial_label
        trial_text.draw()
    photosensor_dot.fillColor = 'white' if photo_white else 'black'
    photosensor_dot.draw()
    window.flip()

def timed_screen(stim, duration, event_name=None, trial_idx=None, photo_white=False):
    start_time = global_clock.getTime()
    if event_name is not None:
        events.append({
            "trial": trial_idx,
            "event": event_name,
            "time": start_time
        })

    timer = core.Clock()
    while timer.getTime() < duration:
        check_escape()
        collect_queue_data()
        draw_screen(main_stim=stim, trial_label=f"Trial {trial_idx+1}/{n_trials}", photo_white=photo_white)

def shutdown_and_save():
    collect_queue_data()
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_file_eeg, eeg)
    np.save(save_file_aux, aux)
    np.save(save_file_timestamp, timestamp)
    np.save(save_file_events, np.array(events, dtype=object))

    if cyton_in:
        stop_event.set()
        board.stop_stream()
        board.release_session()

# TASK START SCREEN

instruction_text.text = "You will see a countdown.\nBlink once when the screen says BLINK NOW.\n\nPress any key to begin."
while True:
    draw_screen(main_stim=instruction_text, photo_white=False)
    keys = kb.getKeys()
    if len(keys) > 0:
        break

# TRIAL LOOP

for i_trial in range(n_trials):
    fixation.text = "+"
    timed_screen(fixation, baseline_duration, event_name="baseline", trial_idx=i_trial, photo_white=False)

    instruction_text.text = "Blink in 3"
    timed_screen(instruction_text, countdown_step, event_name="countdown_3", trial_idx=i_trial, photo_white=False)

    instruction_text.text = "2"
    timed_screen(instruction_text, countdown_step, event_name="countdown_2", trial_idx=i_trial, photo_white=False)

    instruction_text.text = "1"
    timed_screen(instruction_text, countdown_step, event_name="countdown_1", trial_idx=i_trial, photo_white=False)

    instruction_text.text = "BLINK NOW"
    timed_screen(instruction_text, blink_duration, event_name="blink_now", trial_idx=i_trial, photo_white=True)

    instruction_text.text = "Relax"
    timed_screen(instruction_text, rest_duration, event_name="rest", trial_idx=i_trial, photo_white=False)

instruction_text.text = "Done.\nThank you."
timed_screen(instruction_text, 2.0, event_name="task_end", trial_idx=-1, photo_white=False)

shutdown_and_save()
window.close()
core.quit()
