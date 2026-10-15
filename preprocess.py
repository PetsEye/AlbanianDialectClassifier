import os
import sys

import torch
import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit

from common import CSV_PATH, RAW_DATA_DIR, PROCESSED_DIR, CHUNK_MS, speaker_id_from_file

TEST_SIZE = 0.2
SPLIT_SEED = 42
TEST_MODE = False


def load_pipeline():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("CRITICAL ERROR: HF_TOKEN not found. Check your .env file!")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def diarize(pipeline, file_path):
    res = pipeline(file_path)
    if hasattr(res, "speaker_diarization"):
        return res.speaker_diarization
    if hasattr(res, "itertracks"):
        return res
    return res[0]


def process_audio(pipeline, file_id, file_path, output_dir):
    print(f"--- Diarizing Speaker {file_id} ---")
    diarization = diarize(pipeline, file_path)

    speaker_durations = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + turn.duration

    if not speaker_durations:
        print(f"   [!] No speakers detected in {file_id}")
        return

    main_speaker = max(speaker_durations, key=speaker_durations.get)
    audio = AudioSegment.from_mp3(file_path)

    chunk_count = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker == main_speaker and turn.duration >= CHUNK_MS / 1000:
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]

            for i in range(0, len(segment) - CHUNK_MS + 1, CHUNK_MS):
                chunk = segment[i:i + CHUNK_MS]
                chunk_name = f"{file_id}_chunk_{chunk_count}.wav"
                chunk.export(os.path.join(output_dir, chunk_name), format="wav")
                chunk_count += 1

    print(f"   [+] Generated {chunk_count} chunks.")


def load_speakers():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"file_name": "file_path"})
    df["id"] = df["file_path"].apply(speaker_id_from_file)

    if "status" in df.columns:
        df = df[df["status"] == "Success"]
    else:
        print("⚠️ 'status' column not found, skipping filter.")

    if df.empty:
        raise ValueError("CRITICAL ERROR: DataFrame is empty after filtering! Check your data.")
    return df


def main():
    df = load_speakers()

    for folder in ["train", "test"]:
        os.makedirs(os.path.join(PROCESSED_DIR, folder), exist_ok=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["id"]))
    train_speakers = set(df.iloc[train_idx]["id"].astype(str))

    print(f"Total Speakers: {len(df)}")
    print(f"Training on: {len(train_idx)} speakers")
    print(f"Testing on: {len(test_idx)} speakers")

    pipeline = load_pipeline()

    rows = list(df.iterrows())
    if TEST_MODE:
        rows = rows[:1]

    for _, row in rows:
        speaker_id = str(row["id"])
        input_path = os.path.join(RAW_DATA_DIR, os.path.basename(row["file_path"]))
        category = "train" if speaker_id in train_speakers else "test"
        output_subdir = os.path.join(PROCESSED_DIR, category)

        if any(fname.startswith(f"{speaker_id}_chunk") for fname in os.listdir(output_subdir)):
            print(f"Skipping {speaker_id} (Already processed)")
            continue

        if not os.path.exists(input_path):
            print(f"   [!] Missing audio for {speaker_id}: {input_path}")
            continue

        try:
            process_audio(pipeline, speaker_id, input_path, output_subdir)
        except Exception as e:
            print(f"   [!] Critical error on {speaker_id}: {e}")


if __name__ == "__main__":
    main()
