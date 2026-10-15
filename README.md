# Albanian Dialect Classifier

This project tries to tell two Albanian dialects apart from audio: **Gheg** and **Tosk**. Arvanitic and Arbëresh are counted as Tosk.

The data comes from the [Albanian dialects archive](http://dialects.albanianlanguage.net/).

## The pipeline

1. **Scrape** (`scraper.py`): downloads the speaker recordings and their details (dialect, location, speaker). Saves them to `albanian_ml_dataset.csv`.
2. **Preprocess** (`preprocess.py`): finds the main speaker in each recording and cuts their speech into 5-second chunks. It splits the speakers into train and test sets.
3. **Train** (`train_cnn.py`): trains a ResNet-18 model on spectrograms of the chunks. Saves the model to `albanian_cnn_model.pth`.
4. **Predict** (`predict.py`): classifies new `.wav` files or a folder of them.
5. **Visualize** (`visualize_dialects.py`): draws a chart of how many recordings there are per dialect.

Shared settings and helpers live in `common.py` and `model_utils.py`.

## Setup

You need Python 3 and `ffmpeg` (used to read MP3 files).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio torchvision pandas scikit-learn matplotlib tqdm pydub python-dotenv pyannote.audio playwright
playwright install chromium
```

The preprocessing step uses a pyannote model, which needs a Hugging Face token. Put it in a file called `.env`:

```
HF_TOKEN=your_token_here
```

## Running it

```bash
python scraper.py
python preprocess.py
python train_cnn.py
python predict.py processed_dataset/test/some_chunk.wav
python visualize_dialects.py
```

To predict a whole folder, pass the folder instead of a file.

## Try it on your own voice

1. Record 10–30 seconds of yourself speaking your dialect in a quiet room. Voice Memos on a phone works.
2. Convert it to mono 16 kHz WAV (needs `ffmpeg`, e.g. `brew install ffmpeg`):
   ```bash
   ffmpeg -i my_voice.m4a -ac 1 -ar 16000 my_voice.wav
   ```
3. Cut it into 5-second pieces, since the model was trained on chunks that long:
   ```bash
   mkdir my_chunks
   ffmpeg -i my_voice.wav -f segment -segment_time 5 -c copy my_chunks/chunk_%03d.wav
   ```
4. Predict on the folder:
   ```bash
   python predict.py my_chunks
   ```

If most chunks give the same answer, that's the overall guess. Treat it as a rough check, not a verdict: the model was trained on old interview recordings, so a phone recording can score differently for reasons that have nothing to do with dialect.

## Results

With the current data (200 speakers, 135 Gheg and 65 Tosk):

- About **84%** accuracy on the validation set.
- Gheg is easier to spot than Tosk.

The validation set is also used to pick the best model, so this number is a bit optimistic. A separate third split would give a fairer score.

## Files

| File | What it is |
|---|---|
| `scraper.py` | Downloads recordings and details |
| `preprocess.py` | Cuts speech into chunks and splits the data |
| `train_cnn.py` | Trains the model |
| `predict.py` | Classifies audio |
| `visualize_dialects.py` | Draws the dataset chart |
| `common.py` | Shared settings and dialect labels |
| `model_utils.py` | Model and audio feature code |
| `albanian_ml_dataset.csv` | Speaker details |
