import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models

from common import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH


def build_model(num_classes, pretrained=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
    model.conv1 = new_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_mel_transform():
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        ),
        torchaudio.transforms.AmplitudeToDB(),
    )


def load_features(path, mel):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    spec = mel(waveform)
    return (spec - spec.mean()) / (spec.std() + 1e-6)
