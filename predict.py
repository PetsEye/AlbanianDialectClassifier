import os
import sys

import torch

from common import MODEL_PATH
from model_utils import build_model, make_mel_transform, load_features


def load_model(path=MODEL_PATH):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    labels = checkpoint["labels"]

    model = build_model(len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, labels, device


def predict_file(path, model, labels, device, mel):
    spec = load_features(path, mel).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(spec), dim=1)[0]
    idx = probs.argmax().item()
    print(f"🎤 Audio: {os.path.basename(path)}")
    print(f"🧠 Prediction: {labels[idx]}")
    print(f"📊 Confidence: {probs[idx].item() * 100:.2f}%")
    print("   " + ", ".join(f"{name} {p * 100:.1f}%" for name, p in zip(labels, probs.tolist())) + "\n")


def collect_wavs(paths):
    wavs = []
    for path in paths:
        if os.path.isdir(path):
            wavs.extend(os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".wav"))
        else:
            wavs.append(path)
    return wavs


def main(argv):
    if not argv:
        print("Usage: python predict.py <file.wav | folder> [...]")
        sys.exit(1)

    model, labels, device = load_model()
    mel = make_mel_transform()
    for path in collect_wavs(argv):
        predict_file(path, model, labels, device, mel)


if __name__ == "__main__":
    main(sys.argv[1:])
