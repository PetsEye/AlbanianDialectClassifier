import os

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
CHUNK_MS = 5000

CSV_PATH = "albanian_ml_dataset.csv"
RAW_DATA_DIR = "albanian_recordings"
PROCESSED_DIR = "processed_dataset"
MODEL_PATH = "albanian_cnn_model.pth"

LABELS = ["Gheg", "Tosk"]


def categorize_dialect(text):
    text = str(text).lower()
    if "gheg" in text:
        return "Gheg"
    if "tosk" in text or "arvanit" in text:
        return "Tosk"
    return None


def speaker_id_from_file(file_name):
    return os.path.basename(str(file_name)).replace(".mp3", "")
