"""Vision Teacher (ResNet-50) per la classificazione dei frame video

``VisionTeacher`` adatta un ResNet-50 pre-addestrato su ImageNet (da
``torchvision.models``, già disponibile nel container Apptainer del cluster) al
nostro subset a 25 classi di VGGSound, fine-tunando gli ultimi blocchi.

È il modello usato come:
- **Teacher visivo** (Fase 2): fine-tunato con cross-entropy sui frame centrali.
- **Teacher frozen** (Fase 3): in ``eval()``, fornisce logits ed embedding allo
  Student AST tramite la DistillationLoss.

Il modello espone sia ``forward(x) → logits`` che ``forward_features(x) → embedding``
(il vettore 2048-d dopo l'average pooling, prima del classification head) per
consentire la feature-level distillation nella Fase 3.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Dimensione embedding di ResNet-50 (output di avgpool, prima del fc).
_RESNET50_EMBED_DIM = 2048

# Ordine degli stage di ResNet-50, dal più superficiale al più profondo. Usato
# per interpretare ``freeze_until`` (si congela tutto fino allo stage indicato).
_RESNET_STAGES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


class VisionTeacher(nn.Module):
    """ResNet-50 fine-tunato per la classificazione dei frame video.

    Args:
        num_classes: Numero di classi di output (es. 25 per il nostro subset).
        pretrained: Se ``True``, carica i pesi ImageNet-pretrained di ResNet-50.
        weights_path: Path a un file ``.pth`` di pesi locali (per il cluster
            offline). Se fornito ed esistente ha precedenza sul download.
        freeze_until: Nome dello stage fino al quale congelare i parametri
            (incluso). Valori validi: ``None`` (nessun congelamento), ``"conv1"``,
            ``"layer1"``, ``"layer2"``, ``"layer3"``, ``"layer4"``. Default
            ``"layer2"``: fine-tune di layer3, layer4 e fc.
        drop_rate: Dropout applicato prima del classification head.
    """

    def __init__(
        self,
        num_classes: int = 25,
        pretrained: bool = True,
        weights_path: str | None = None,
        freeze_until: str | None = "layer2",
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = _RESNET50_EMBED_DIM

        # ------------------------------------------------------------------
        # 1. Carica backbone ResNet-50 da torchvision.
        #    Se weights_path è fornito e il file esiste, carica i pesi dal file
        #    locale (per il cluster senza accesso a pytorch.org). Altrimenti
        #    scarica automaticamente (solo uso locale).
        # ------------------------------------------------------------------
        if pretrained and weights_path and os.path.isfile(weights_path):
            print(f"[Teacher] Caricamento pesi pretrained da file locale: {weights_path}")
            backbone = resnet50(weights=None)  # architettura senza download
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
        elif pretrained:
            print("[Teacher] Download pesi ResNet-50 da pytorch.org (solo uso locale)...")
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(weights=None)

        # ------------------------------------------------------------------
        # 2. Sostituisci il classification head ImageNet (1000 classi) con la
        #    nostra testa. Salviamo a parte la feature dimension (2048).
        # ------------------------------------------------------------------
        backbone.fc = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        # ------------------------------------------------------------------
        # 3. Congelamento parziale per il fine-tuning.
        # ------------------------------------------------------------------
        self.freeze_until = freeze_until
        if freeze_until:
            self._freeze_until(freeze_until)

    # ------------------------------------------------------------------ #
    # Setup: congelamento degli stage
    # ------------------------------------------------------------------ #
    def _freeze_until(self, stage: str) -> None:
        """Congela i parametri di tutti gli stage fino a ``stage`` (incluso).

        Gli stage successivi e il classification head restano trainabili. La
        BatchNorm congelata mantiene comunque le statistiche ImageNet (running
        mean/var) perché non viene messa in train mode per i parametri frozen.
        """
        if stage not in _RESNET_STAGES:
            raise ValueError(
                f"freeze_until={stage!r} non valido. Validi: {(None,) + _RESNET_STAGES}"
            )
        cutoff = _RESNET_STAGES.index(stage)
        frozen = set(_RESNET_STAGES[: cutoff + 1])
        # ``bn1`` è la BatchNorm di ``conv1``: la congeliamo insieme a conv1.
        if "conv1" in frozen:
            frozen.add("bn1")
        for name, module in self.backbone.named_children():
            if name in frozen:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Scongela l'intero backbone per il fine-tuning completo."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Restituisce l'embedding (vettore dopo avgpool, prima del head).

        Args:
            x: Frame ``(B, 3, H, W)`` normalizzato con le statistiche ImageNet.

        Returns:
            Embedding ``(B, 2048)``.
        """
        # ``backbone.fc`` è ``nn.Identity``, quindi il forward del ResNet-50
        # restituisce direttamente l'embedding post-avgpool appiattito.
        return self.backbone(x)  # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward completo: frame → logits.

        Args:
            x: Frame ``(B, 3, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        features = self.forward_features(x)   # (B, 2048)
        return self.classifier(features)      # (B, num_classes)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"VisionTeacher("
            f"backbone=ResNet-50 (torchvision), "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"freeze_until={self.freeze_until!r}, "
            f"params_total={total:,}, params_trainable={trainable:,})"
        )


def build_vision_teacher(cfg: dict) -> VisionTeacher:
    """Factory: costruisce un VisionTeacher da un dict di configurazione.

    Legge le chiavi ``model.*`` e ``dataset.*`` dal dict caricato da YAML.
    Le chiavi mancanti usano i default di ``VisionTeacher``.

    Esempio::

        cfg = yaml.safe_load(open("experiments/configs/teacher_vision.yaml"))
        model = build_vision_teacher(cfg)
    """
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})

    return VisionTeacher(
        num_classes=int(ds_cfg.get("num_classes", 25)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        weights_path=model_cfg.get("weights_path", None),
        freeze_until=model_cfg.get("freeze_until", "layer2"),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
    )


if __name__ == "__main__":
    # Quick smoke test: verifica shape di input/output e il congelamento parziale.
    print("Smoke test VisionTeacher (torchvision backend)...")
    model = VisionTeacher(num_classes=25, pretrained=False, freeze_until="layer2")
    print(model)

    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    features = model.forward_features(dummy)

    assert logits.shape == (2, 25), f"Logits shape errata: {logits.shape}"
    assert features.shape == (2, model.embed_dim), f"Features shape errata: {features.shape}"

    # Verifica che layer1/layer2 siano congelati e layer3/layer4/fc trainabili.
    frozen_ok = not any(p.requires_grad for p in model.backbone.layer2.parameters())
    trainable_ok = all(p.requires_grad for p in model.backbone.layer4.parameters())
    assert frozen_ok, "layer2 dovrebbe essere congelato"
    assert trainable_ok, "layer4 dovrebbe essere trainabile"

    print(f"  logits:   {tuple(logits.shape)}")
    print(f"  features: {tuple(features.shape)}")
    print("  congelamento: layer2 frozen, layer4 trainable — OK")
    print("Smoke test PASSATO.")
