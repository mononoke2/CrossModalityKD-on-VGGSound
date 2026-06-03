"""Audio Spectrogram Transformer (AST) per la classificazione audio.

``AudioSpectrogramTransformer`` adatta un ViT pre-addestrato (via ``timm``)
per accettare log-mel-spectrogram 2D estratti da audio a 16 kHz.  È il modello
usato come:

- **Baseline audio** (Fase 1): addestrato con sola cross-entropy.
- **Student distillato** (Fase 3): addestrato con DistillationLoss + CE.

Il modello espone sia ``forward(x) → logits`` che ``forward_features(x) → embedding``
per consentire la feature-level distillation nella Fase 3.

Riferimento architetturale
--------------------------
Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021.
Qui usiamo un ViT standard da ``timm`` e ne adattiamo i pos-embed anziché
ripartire da zero, il che permette di riutilizzare i pesi ImageNet-pretrained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# Dimensione di embedding usata dal backbone ViT-Base (16 teste × 64 = 768).
_VIT_BASE_EMBED_DIM = 768


class AudioSpectrogramTransformer(nn.Module):
    """ViT adattato per mel-spectrogram 2D (frequenza × tempo).

    Args:
        num_classes: Numero di classi di output (es. 25 per il nostro subset).
        backbone: Nome del modello ``timm`` da usare come backbone.
            Default: ``"vit_base_patch16_224"`` (ViT-B/16 ImageNet-pretrained).
        pretrained: Se ``True``, carica i pesi pre-addestrati dal repository
            ``timm``. Richiede accesso internet al primo avvio (poi cache locale).
        n_mels: Numero di bande Mel (asse frequenza). Deve corrispondere alla
            config del dataloader (default 128).
        target_length: Numero di frame temporali (asse tempo, default 1024).
        drop_rate: Dropout sull'output del classification head.
        freeze_backbone: Se ``True``, congela tutti i parametri tranne il head.
            Utile per il warm-up iniziale.
    """

    def __init__(
        self,
        num_classes: int = 25,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        n_mels: int = 128,
        target_length: int = 1024,
        drop_rate: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm non è installato. Esegui `pip install timm` o aggiungi "
                "timm all'environment.yml e ricrea l'ambiente."
            )

        self.num_classes = num_classes
        self.n_mels = n_mels
        self.target_length = target_length
        self.embed_dim = _VIT_BASE_EMBED_DIM  # aggiornato sotto se il backbone è diverso

        # Crea il backbone ViT senza testa di classificazione originale.
        # num_classes=0 → timm restituisce le feature dopo il pool (embedding).
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,       # rimuove il classifier originale
            in_chans=1,          # log-mel-spectrogram ha 1 canale (no RGB)
            drop_rate=drop_rate,
        )

        # Recupera la dimensione reale dell'embedding dal backbone.
        self.embed_dim = self.backbone.num_features

        # Adatta il patch embedding e i positional embedding al nostro input:
        # il ViT-B/16 si aspetta 224×224 con patch 16×16 → 14×14 = 196 patch.
        # Il nostro mel-spectrogram è (1, 128, 1024) → con patch 16×16 abbiamo
        # (128/16) × (1024/16) = 8 × 64 = 512 patch, molto più del default.
        # Adattiamo i positional embed via interpolazione bilineare.
        self._adapt_pos_embed(n_mels, target_length)

        # Nuovo classification head specifico per il nostro task.
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )

        # Inizializziamo il layer lineare con scaling ridotto (buona pratica per
        # evitare instabilità nei primi step di training).
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------ #
    # Setup: adattamento positional embedding
    # ------------------------------------------------------------------ #
    def _adapt_pos_embed(self, n_mels: int, target_length: int) -> None:
        """Interpola i positional embedding 2D per adattarli alla nuova griglia di patch.

        Il ViT-B/16 ha pos-embed shape ``(1, 196+1, 768)`` (196 patch + CLS).
        Per il nostro input (128, 1024) con patch 16×16 otteniamo 8×64=512 patch
        → pos-embed target: ``(1, 512+1, 768)``.

        L'interpolazione è bilineare nello spazio 2D (freq × time).
        """
        if not hasattr(self.backbone, 'pos_embed') or self.backbone.pos_embed is None:
            return  # alcuni backbone non usano pos_embed fisso

        old_pos_embed = self.backbone.pos_embed  # (1, N+1, D)
        cls_pos = old_pos_embed[:, :1, :]       # (1, 1, D) — CLS token
        patch_pos = old_pos_embed[:, 1:, :]     # (1, N_old, D)

        # Dimensioni originali (quadrate, ViT-B/16: 14×14=196)
        n_old = patch_pos.shape[1]
        h_old = w_old = int(math.sqrt(n_old))

        # Nuove dimensioni patch per il nostro spettrogramma
        patch_size = 16
        h_new = n_mels // patch_size          # 128//16 = 8
        w_new = target_length // patch_size   # 1024//16 = 64

        if h_new * w_new == n_old:
            return  # già compatibile, nessuna interpolazione necessaria

        # Riforma a griglia 2D e interpola
        patch_pos = patch_pos.reshape(1, h_old, w_old, -1).permute(0, 3, 1, 2)  # (1, D, H, W)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=(h_new, w_new),
            mode="bilinear",
            align_corners=False,
        )  # (1, D, h_new, w_new)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h_new * w_new, -1)  # (1, N_new, D)

        new_pos_embed = torch.cat([cls_pos, patch_pos], dim=1)  # (1, N_new+1, D)
        self.backbone.pos_embed = nn.Parameter(new_pos_embed)

    def _freeze_backbone(self) -> None:
        """Congela tutti i parametri del backbone (solo il head rimane trainable)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Scongela il backbone per il fine-tuning completo."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Restituisce l'embedding globale prima del classification head.

        Args:
            x: Mel-spectrogram ``(B, 1, n_mels, target_length)``.

        Returns:
            Embedding ``(B, embed_dim)``.
        """
        return self.backbone(x)  # backbone ha num_classes=0 → restituisce features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward completo: mel-spectrogram → logits.

        Args:
            x: Mel-spectrogram ``(B, 1, n_mels, target_length)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        features = self.forward_features(x)   # (B, embed_dim)
        return self.classifier(features)      # (B, num_classes)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"AudioSpectrogramTransformer("
            f"backbone={self.backbone.__class__.__name__}, "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"params_total={total:,}, params_trainable={trainable:,})"
        )


def build_ast(cfg: dict) -> AudioSpectrogramTransformer:
    """Factory: costruisce un AST da un dict di configurazione.

    Il dict ``cfg`` dovrebbe contenere le chiavi ``model.*`` lette da
    ``baseline_audio.yaml`` (o ``distillation.yaml``).  Le chiavi mancanti
    usano i default di ``AudioSpectrogramTransformer.__init__``.

    Esempio::

        cfg = yaml.safe_load(open("experiments/configs/baseline_audio.yaml"))
        model = build_ast(cfg)
    """
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})

    return AudioSpectrogramTransformer(
        num_classes=int(ds_cfg.get("num_classes", 25)),
        backbone=model_cfg.get("backbone", "vit_base_patch16_224"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        n_mels=int(ds_cfg.get("n_mels", 128)),
        target_length=int(ds_cfg.get("target_length", 1024)),
        drop_rate=float(model_cfg.get("drop_rate", 0.1)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )


if __name__ == "__main__":
    # Quick smoke test: verifica le shape di input/output e l'interpolazione dei pos-embed.
    print("Smoke test AudioSpectrogramTransformer...")
    model = AudioSpectrogramTransformer(num_classes=25, pretrained=False)
    print(model)

    dummy = torch.randn(2, 1, 128, 1024)
    logits = model(dummy)
    features = model.forward_features(dummy)

    assert logits.shape == (2, 25), f"Logits shape errata: {logits.shape}"
    assert features.shape == (2, model.embed_dim), f"Features shape errata: {features.shape}"
    print(f"  logits:   {tuple(logits.shape)}")
    print(f"  features: {tuple(features.shape)}")
    print("Smoke test PASSATO.")
