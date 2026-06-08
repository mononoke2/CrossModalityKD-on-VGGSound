"""Scarica i pesi pretrained dei modelli leggeri (Extra 3) in pretrained_weights/.

Eseguire in locale (non richiede GPU, solo internet):

    python scripts/download_light_weights.py

Poi sincronizzare sul cluster:

    ./scripts/sync_to_cluster.sh --models
"""

from pathlib import Path

import torch
import torchvision  # noqa: F401 — assicura che il modulo sia disponibile
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)

_TARGETS = [
    ("efficientnet_b0", efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, "efficientnet_b0.pth"),
    ("mobilenet_v3_small", mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT, "mobilenet_v3_small.pth"),
]

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "pretrained_weights"


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"

    for pattern, model_fn, weights, dst_name in _TARGETS:
        dst = _OUTPUT_DIR / dst_name
        if dst.exists():
            print(f"[skip] {dst_name} già presente.")
            continue

        print(f"[download] {pattern} ...")
        model_fn(weights=weights)  # scarica in cache (~/.cache/torch/hub/checkpoints/)

        match = next(cache_dir.glob(f"{pattern}*.pth"), None)
        if match is None:
            print(f"[ERROR] file non trovato in cache: {cache_dir}/{pattern}*.pth")
            continue

        import shutil
        shutil.copy(match, dst)
        print(f"[ok] {match.name} → {dst}")

    print("Download completato.")


if __name__ == "__main__":
    main()
