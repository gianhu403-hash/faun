import logging
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Literal
import os

logger = logging.getLogger(__name__)

_yamnet = None
_head = None
_yamnet_class_names = None

CLASSES = ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]
AudioClass = Literal[
    "chainsaw", "gunshot", "engine", "axe", "fire", "background", "unknown"
]

MODEL_PATH = os.environ.get(
    "YAMNET_HEAD_PATH",
    os.path.join(os.path.dirname(__file__), "yamnet_forest_classifier_v8.keras"),
)

# Mapping from YAMNet base class names to our target classes
YAMNET_CLASS_MAP = {
    "Chainsaw": "chainsaw",
    "Power tool": "chainsaw",
    "Sawing": "chainsaw",
    "Drill": "chainsaw",
    "Tools": "chainsaw",
    "Gunshot": "gunshot",
    "Gunfire": "gunshot",
    "Firearms": "gunshot",
    "Machine gun": "gunshot",
    "Cap gun": "gunshot",
    "Engine": "engine",
    "Vehicle": "engine",
    "Motor vehicle (road)": "engine",
    "Motorcycle": "engine",
    "Light engine (high frequency)": "engine",
    "Medium engine (mid frequency)": "engine",
    "Heavy engine (low frequency)": "engine",
    "Accelerating, revving, vroom": "engine",
    "Truck": "engine",
    "Bus": "engine",
    "Chop": "axe",
    "Wood": "axe",
    "Whack, thwack": "axe",
    "Thump, thud": "axe",
    "Fire": "fire",
    "Crackle": "fire",
    "Fire alarm": "fire",
}

YAMNET_THRESHOLD = 0.15


@dataclass
class AudioResult:
    label: AudioClass
    confidence: float
    raw_scores: dict


def _unknown() -> AudioResult:
    return AudioResult(label="unknown", confidence=0.0, raw_scores={})


def _load_yamnet_class_names():
    global _yamnet_class_names
    if _yamnet_class_names is not None:
        return _yamnet_class_names
    try:
        import csv, io, urllib.request

        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        resp = urllib.request.urlopen(url)
        reader = csv.reader(io.StringIO(resp.read().decode()))
        next(reader)
        _yamnet_class_names = [row[2] for row in reader]
    except Exception:
        _yamnet_class_names = []
    return _yamnet_class_names


def _load_models():
    global _yamnet, _head
    if _yamnet is None:
        import tensorflow_hub as hub

        _yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    if _head is None:
        import tensorflow as tf

        try:
            _head = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            logger.warning("Failed to load head model %s: %s", MODEL_PATH, e)
            _head = None
    return _yamnet, _head


def _classify_base_yamnet(scores_np: np.ndarray) -> AudioResult:
    """Classify using base YAMNet 521 classes mapped to our target classes."""
    class_names = _load_yamnet_class_names()
    if not class_names:
        return _unknown()

    mean_scores = scores_np.mean(axis=0)

    # Sum scores per target class (aggregate evidence from related YAMNet classes)
    agg: dict[str, float] = {c: 0.0 for c in CLASSES}
    for idx, name in enumerate(class_names):
        target = YAMNET_CLASS_MAP.get(name)
        if target and idx < len(mean_scores):
            agg[target] += float(mean_scores[idx])

    # Cap at 1.0
    for k in agg:
        agg[k] = min(agg[k], 1.0)

    best_class = max(agg, key=agg.get)
    best_score = agg[best_class]

    if best_score < YAMNET_THRESHOLD or best_class == "background":
        return AudioResult(
            label="background",
            confidence=1.0 - sum(v for k, v in agg.items() if k != "background"),
            raw_scores=agg,
        )

    return AudioResult(label=best_class, confidence=best_score, raw_scores=agg)


def classify(audio_path: str) -> AudioResult:
    yamnet, head = _load_models()

    waveform, sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if len(waveform) == 0:
        return _unknown()
    if len(waveform) < 15600:
        waveform = np.pad(waveform, (0, 15600 - len(waveform)))

    peak = np.max(np.abs(waveform))
    if peak > 1e-6:
        waveform = waveform / peak

    scores, embeddings, spectrogram = yamnet(waveform)
    scores_np = scores.numpy()
    emb_np = embeddings.numpy()

    if emb_np.shape[0] == 0:
        return _unknown()

    # Try fine-tuned head first
    if head is not None:
        mean_emb = emb_np.mean(axis=0)
        max_emb = emb_np.max(axis=0)
        features_parts = [np.concatenate([mean_emb, max_emb])]  # 2048

        # PCEN (128-dim)
        try:
            import librosa

            S = librosa.feature.melspectrogram(
                y=waveform,
                sr=16000,
                n_mels=64,
                fmin=125,
                fmax=7500,
                hop_length=160,
                n_fft=400,
            )
            pcen_S = librosa.pcen(
                S * (2**31),
                sr=16000,
                hop_length=160,
                gain=0.98,
                bias=2,
                power=0.5,
                time_constant=0.4,
            )
            features_parts.append(
                np.concatenate([pcen_S.mean(axis=1), pcen_S.max(axis=1)])
            )
        except Exception as e:
            logger.warning("PCEN failed: %s", e)
            features_parts.append(np.zeros(128))

        # Temporal variance (5-dim)
        try:
            if scores_np.shape[0] >= 2:
                variance = np.var(scores_np, axis=0)
                top5 = np.argsort(variance)[-5:]
                features_parts.append(variance[top5])
            else:
                features_parts.append(np.zeros(5))
        except Exception:
            features_parts.append(np.zeros(5))

        features = np.concatenate(features_parts)  # 2181

        # Backward compat: truncate if old model loaded
        expected_dim = head.input_shape[-1]
        features = features[:expected_dim]
        if features.shape[0] < expected_dim:
            features = np.concatenate(
                [features, np.zeros(expected_dim - features.shape[0])]
            )

        pred = head(features[np.newaxis, :], training=False).numpy()[0]
        pred_idx = int(np.argmax(pred))
        confidence = float(pred[pred_idx])
        label = CLASSES[pred_idx] if pred_idx < len(CLASSES) else "unknown"
        raw = {CLASSES[i]: float(pred[i]) for i in range(len(CLASSES))}

        # If head model is confident about a non-background class, use it
        if label != "background" and confidence >= 0.50:
            return AudioResult(label=label, confidence=confidence, raw_scores=raw)

    # Fallback: base YAMNet class mapping (sum aggregation)
    return _classify_base_yamnet(scores_np)
