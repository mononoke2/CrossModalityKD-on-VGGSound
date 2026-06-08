"""Modelli audio leggeri per il benchmark

Adatta backbone CNN pre-addestrati (EfficientNet-B0, MobileNetV3-Small) da
``torchvision`` per classificare mel-spectrogram audio monocanale.

Interfaccia identica ad ``AudioSpectrogramTransformer``:
  - ``forward(x) → logits``             ``(B, num_classes)``
  - ``forward_features(x) → embedding`` ``(B, embed_dim)``
  - ``embed_dim: int``

Il backbone conserva il suo ``AdaptiveAvgPool2d(1)`` finale, quindi accetta
mel-spectrogram ``(B, 1, 128, 1024)`` senza resize esplicito.

Adattamento primo conv
-----------------------
``Conv2d(in_channels=3, …)`` → ``Conv2d(in_channels=1, …)``.
Se i pesi pretrained sono disponibili, i 3 kernel RGB vengono mediati sulla
dimensione del canale (strategia standard per input monocanale, cfr. ``ast_model.py``).

Benchmark Extra 3 vs AST baseline
-----------------------------------
| Modello               | embed_dim | Size attesa |
|-----------------------|-----------|-------------|
| AST (ViT-B/16)        |       768 |    ~327 MB  |
| EfficientNetB0Audio   |      1280 |     ~20 MB  |
| MobileNetV3SmallAudio |       576 |      ~9 MB  |

Cluster (offline) — download pesi pretrained
---------------------------------------------
Eseguire in locale una sola volta::

    python scripts/download_light_weights.py

Oppure manualmente::

    python -c "
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    import torch, shutil
    from pathlib import Path
    cache = Path(torch.hub.get_dir()) / 'checkpoints'
    Path('pretrained_weights').mkdir(exist_ok=True)
    for pattern, dst in [('efficientnet_b0', 'efficientnet_b0.pth'),
                         ('mobilenet_v3_small', 'mobilenet_v3_small.pth')]:
        match = next(cache.glob(f'{pattern}*.pth'), None)
        if match: shutil.copy(match, f'pretrained_weights/{dst}')
    "

Poi sincronizzare sul cluster::

    CLUSTER_USER=... ./scripts/sync_to_cluster.sh --models

E impostare nel YAML::

    student:
      type: efficientnet_b0_audio          # oppure mobilenet_v3_small_audio
      weights_path: pretrained_weights/efficientnet_b0.pth
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)

_EFFICIENTNET_B0_EMBED_DIM: int = 1280
_MOBILENET_V3_SMALL_EMBED_DIM: int = 576


class EfficientNetB0Audio(nn.Module):
    """EfficientNet-B0 adattato per mel-spectrogram monocanale.

    Args:
        num_classes:     Numero di classi di output.
        pretrained:      Se ``True``, carica i pesi ImageNet-pretrained.
        weights_path:    Percorso locale ai pesi (per il cluster offline).
        drop_rate:       Dropout prima del classification head.
        freeze_backbone: Se ``True``, congela il backbone (solo head trainable).
    """

    def __init__(
        self,
        num_classes: int = 25,
        pretrained: bool = True,
        weights_path: str | None = None,
        drop_rate: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = _EFFICIENTNET_B0_EMBED_DIM

        if pretrained and weights_path and os.path.isfile(weights_path):
            print(f"[EfficientNetB0Audio] Pesi pretrained da file: {weights_path}")
            backbone = efficientnet_b0(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
        elif pretrained:
            print("[EfficientNetB0Audio] Download pesi EfficientNet-B0 da pytorch.org...")
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            backbone = efficientnet_b0(weights=None)

        # features[0] è ConvNormActivation; [0][0] è Conv2d(3, 32, 3, stride=2, padding=1)
        first_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        backbone.classifier = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Mel-spectrogram ``(B, 1, H, W)`` → embedding pooled ``(B, 1280)``."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"EfficientNetB0Audio("
            f"backbone=EfficientNet-B0 (torchvision), "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"params_total={total:,}, params_trainable={trainable:,})"
        )


class MobileNetV3SmallAudio(nn.Module):
    """MobileNetV3-Small adattato per mel-spectrogram monocanale.

    Args:
        num_classes:     Numero di classi di output.
        pretrained:      Se ``True``, carica i pesi ImageNet-pretrained.
        weights_path:    Percorso locale ai pesi (per il cluster offline).
        drop_rate:       Dropout prima del classification head.
        freeze_backbone: Se ``True``, congela il backbone (solo head trainable).
    """

    def __init__(
        self,
        num_classes: int = 25,
        pretrained: bool = True,
        weights_path: str | None = None,
        drop_rate: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = _MOBILENET_V3_SMALL_EMBED_DIM

        if pretrained and weights_path and os.path.isfile(weights_path):
            print(f"[MobileNetV3SmallAudio] Pesi pretrained da file: {weights_path}")
            backbone = mobilenet_v3_small(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
        elif pretrained:
            print("[MobileNetV3SmallAudio] Download pesi MobileNetV3-Small da pytorch.org...")
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # features[0] è ConvNormActivation; [0][0] è Conv2d(3, 16, 3, stride=2, padding=1)
        first_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        # Rimuove il classifier originale (Linear(576, 1024) + Hardswish + Dropout + Linear(1024, 1000))
        # per esporre solo l'embedding da 576 dim dopo avgpool.
        backbone.classifier = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Mel-spectrogram ``(B, 1, H, W)`` → embedding pooled ``(B, 576)``."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"MobileNetV3SmallAudio("
            f"backbone=MobileNetV3-Small (torchvision), "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"params_total={total:,}, params_trainable={trainable:,})"
        )


def build_efficientnet_b0_audio(cfg: dict) -> EfficientNetB0Audio:
    """Factory: costruisce un EfficientNetB0Audio da un dict di configurazione."""
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    return EfficientNetB0Audio(
        num_classes=int(ds_cfg.get("num_classes", 25)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        weights_path=model_cfg.get("weights_path", None),
        drop_rate=float(model_cfg.get("drop_rate", 0.2)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )


def build_mobilenet_v3_small_audio(cfg: dict) -> MobileNetV3SmallAudio:
    """Factory: costruisce un MobileNetV3SmallAudio da un dict di configurazione."""
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    return MobileNetV3SmallAudio(
        num_classes=int(ds_cfg.get("num_classes", 25)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        weights_path=model_cfg.get("weights_path", None),
        drop_rate=float(model_cfg.get("drop_rate", 0.2)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )


def build_light_audio_student(model_type: str, cfg: dict) -> nn.Module:
    """Factory unificata per modelli audio leggeri.

    Args:
        model_type: ``"efficientnet_b0_audio"`` o ``"mobilenet_v3_small_audio"``.
        cfg:        Dict di configurazione con chiavi ``model.*`` e ``dataset.*``.

    Raises:
        ValueError: Se ``model_type`` non è riconosciuto.
    """
    if model_type == "efficientnet_b0_audio":
        return build_efficientnet_b0_audio(cfg)
    if model_type == "mobilenet_v3_small_audio":
        return build_mobilenet_v3_small_audio(cfg)
    raise ValueError(
        f"model_type non riconosciuto: {model_type!r}. "
        "Validi: 'efficientnet_b0_audio', 'mobilenet_v3_small_audio'."
    )


if __name__ == "__main__":
    print("Smoke test modelli audio leggeri (pretrained=False, no internet richiesto)...")
    dummy = torch.randn(2, 1, 128, 1024)

    for cls, expected_dim in [
        (EfficientNetB0Audio, _EFFICIENTNET_B0_EMBED_DIM),
        (MobileNetV3SmallAudio, _MOBILENET_V3_SMALL_EMBED_DIM),
    ]:
        model = cls(num_classes=25, pretrained=False)
        print(model)
        logits = model(dummy)
        features = model.forward_features(dummy)
        assert logits.shape == (2, 25), f"[{cls.__name__}] Logits shape errata: {logits.shape}"
        assert features.shape == (2, expected_dim), f"[{cls.__name__}] Features shape errata: {features.shape}"
        print(f"  logits:   {tuple(logits.shape)}")
        print(f"  features: {tuple(features.shape)}")

    print("Smoke test PASSATO.")
