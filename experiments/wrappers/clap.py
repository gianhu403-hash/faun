"""CLAP zero-shot: скоринг текст-промптов против аудио-окон.

transformers ClapModel (laion/clap-htsat-unfused), torch; GPU если доступен
(образ faun-ml-torch, --device nvidia.com/gpu=all). Аудио — 48 кГц mono.
"""

from __future__ import annotations

import os

import numpy as np

SR = 48_000
DEFAULT_MODEL = os.environ.get("CLAP_MODEL", "laion/clap-htsat-unfused")

_cache: dict = {}


def load_model(name: str = DEFAULT_MODEL):
    """-> (model, processor, device); кэшируется по имени модели."""
    if name not in _cache:
        import torch
        from transformers import ClapModel, ClapProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ClapModel.from_pretrained(name).to(device).eval()
        processor = ClapProcessor.from_pretrained(name)
        _cache[name] = (model, processor, device)
    return _cache[name]


def score(
    audio_windows: np.ndarray,
    prompts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
) -> np.ndarray:
    """Zero-shot скоринг: [N_окон, 48k*len] @ 48kHz x prompts -> logits [N, P].

    Softmax по оси промптов — на стороне вызывающего, если нужен.
    """
    import torch

    model, processor, device = load_model(model_name)
    audio_windows = np.asarray(audio_windows, dtype=np.float32)
    if audio_windows.ndim == 1:
        audio_windows = audio_windows[None, :]

    rows = []
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_emb = model.get_text_features(
            **{k: v.to(device) for k, v in text_inputs.items()}
        )
        # transformers 5.x: get_text_features returns BaseModelOutputWithPooling
        # whose .pooler_output is already L2-normalized; fall back for older API.
        text_emb = getattr(text_emb, "pooler_output", text_emb)

        for i in range(0, len(audio_windows), batch_size):
            batch = list(audio_windows[i : i + batch_size])
            audio_inputs = processor(
                audio=batch, sampling_rate=SR, return_tensors="pt", padding=True
            )
            audio_emb = model.get_audio_features(
                **{k: v.to(device) for k, v in audio_inputs.items()}
            )
            audio_emb = getattr(audio_emb, "pooler_output", audio_emb)
            logit_scale = model.logit_scale_a.exp()
            rows.append((logit_scale * audio_emb @ text_emb.T).cpu().numpy())

    return np.concatenate(rows, axis=0)
