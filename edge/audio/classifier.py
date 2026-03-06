import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Literal

_model = None

AudioClass = Literal["chainsaw", "gunshot", "fire", "birds", "silence", "unknown"]

@dataclass
class AudioResult:
    label: AudioClass
    confidence: float
    raw_scores: dict

TARGET_CLASSES = {
    "Chainsaw":     "chainsaw",
    "Gunshot":      "gunshot",
    "Fire":         "fire",
    "Bird":         "birds",
    "Silence":      "silence",
}

def _load_model():
    global _model
    if _model is None:
        import tensorflow_hub as hub
        _model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _model


def classify(audio_path: str) -> AudioResult:
    model = _load_model()

    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # mono

    scores, embeddings, spectrogram = model(waveform)
    mean_scores = scores.numpy().mean(axis=0)

    import csv, io, requests
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

    try:
        import importlib.resources
        class_names = _get_class_names()
    except Exception:
        class_names = [f"class_{i}" for i in range(521)]

    top_idx = mean_scores.argmax()
    top_name = class_names[top_idx] if top_idx < len(class_names) else "unknown"
    top_conf = float(mean_scores[top_idx])

    label: AudioClass = "unknown"
    for yamnet_name, our_label in TARGET_CLASSES.items():
        if yamnet_name.lower() in top_name.lower():
            label = our_label
            break

    raw = {class_names[i]: float(mean_scores[i])
           for i in mean_scores.argsort()[-5:][::-1]
           if i < len(class_names)}

    return AudioResult(label=label, confidence=top_conf, raw_scores=raw)


def _get_class_names() -> list[str]:
    import os, csv
    cache_path = "/tmp/yamnet_classes.txt"

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return [line.strip() for line in f]

    import urllib.request
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    urllib.request.urlretrieve(url, "/tmp/yamnet_class_map.csv")

    names = []
    with open("/tmp/yamnet_class_map.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["display_name"])

    with open(cache_path, "w") as f:
        f.write("\n".join(names))

    return names
