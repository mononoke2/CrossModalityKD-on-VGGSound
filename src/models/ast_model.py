"""Audio Spectrogram Transformer (AST) per la classificazione audio.

``AudioSpectrogramTransformer`` adatta un ViT-B/16 pre-addestrato da
``torchvision.models`` (già disponibile nel container Apptainer del cluster)
per accettare log-mel-spectrogram 2D estratti da audio a 16 kHz.

È il modello usato come:
- **Baseline audio** (Fase 1): addestrato con sola cross-entropy.
- **Student distillato** (Fase 3): addestrato con DistillationLoss + CE.

Il modello espone sia ``forward(x) → logits`` che ``forward_features(x) → embedding``
per consentire la feature-level distillation nella Fase 3.

Caricamento dei pesi pretrained sul cluster
-------------------------------------------
Il cluster non ha accesso a pytorch.org, ma i pesi possono essere caricati da
file locale tramite il parametro ``weights_path``. Il workflow consigliato è:

1. Scarica i pesi in locale (una tantum)::

    python -c "
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    import shutil, torch
    vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # scarica in cache
    import os; cache = os.path.join(torch.hub.get_dir(), 'checkpoints', 'vit_b_16-c867db91.pth')
    shutil.copy(cache, 'pretrained_weights/vit_b_16.pth')
    "

2. Sincronizza sul cluster con::

    CLUSTER_USER=... ./scripts/sync_to_cluster.sh --models

3. Imposta nel YAML::

    model:
      weights_path: pretrained_weights/vit_b_16.pth

Adattamenti rispetto al ViT-B/16 ImageNet-pretrained
------------------------------------------------------
1. **Patch embedding**: Conv2d(3→1, 16×16) — il canale singolo del
   mel-spectrogram sostituisce i 3 canali RGB. I pesi ImageNet vengono
   conservati mediando i 3 canali originali (standard practice in letteratura).
2. **Positional embedding**: i pos-embed originali (196 patch per 14×14)
   vengono interpolati bilinearmente per coprire la nuova griglia
   8×64 = 512 patch (mel-spectrogram 128×1024 con patch 16×16).
3. **Classification head**: sostituito con Linear(768 → num_classes).

Riferimento architetturale
--------------------------
Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Dimensione embedding di ViT-Base (12 teste × 64 = 768).
_VIT_BASE_EMBED_DIM = 768


class AudioSpectrogramTransformer(nn.Module):
    """ViT-B/16 adattato per mel-spectrogram 2D (frequenza × tempo).

    Args:
        num_classes: Numero di classi di output (es. 25 per il nostro subset).
        pretrained: Se ``True``, carica i pesi ImageNet-pretrained di ViT-B/16
            tramite torchvision. Richiede internet al primo avvio (poi cache).
        n_mels: Numero di bande Mel (asse frequenza). Default 128.
        target_length: Numero di frame temporali (asse tempo). Default 1024.
        drop_rate: Dropout applicato prima del classification head.
        freeze_backbone: Se ``True``, congela il backbone (solo il head è
            trainable). Utile per warm-up iniziale o ablation study.
    """

    def __init__(
        self,
        num_classes: int = 25,
        pretrained: bool = True,
        weights_path: str | None = None,
        n_mels: int = 128,
        target_length: int = 1024,
        drop_rate: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.n_mels = n_mels
        self.target_length = target_length
        self.embed_dim = _VIT_BASE_EMBED_DIM

        # ------------------------------------------------------------------
        # 1. Carica backbone ViT-B/16 da torchvision.
        #    Se weights_path è fornito e il file esiste, carica i pesi dal
        #    file locale (per il cluster senza accesso a pytorch.org).
        #    Altrimenti scarica automaticamente (solo uso locale).
        # ------------------------------------------------------------------
        if pretrained and weights_path and os.path.isfile(weights_path):
            print(f"[AST] Caricamento pesi pretrained da file locale: {weights_path}")
            vit = vit_b_16(weights=None)  # architettura senza download
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            vit.load_state_dict(state_dict)
        elif pretrained:
            print("[AST] Download pesi ViT-B/16 da pytorch.org (solo uso locale)...")
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            vit = vit_b_16(weights=None)

        # ------------------------------------------------------------------
        # 2. Adatta il patch embedding: Conv2d(3, 768, 16, 16) → (1, 768, 16, 16)
        #    I pesi pretrained vengono preservati sommando i 3 canali RGB e
        #    dividendo per 3 (media), che è la strategia standard per input mono.
        # ------------------------------------------------------------------
        old_conv = vit.conv_proj  # Conv2d(3, 768, kernel_size=16, stride=16)
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=16, stride=16, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        vit.conv_proj = new_conv

        # ------------------------------------------------------------------
        # 3. Interpola i positional embedding per la griglia 8×64 (512 patch)
        #    I pos-embed originali hanno shape (1, 197, 768): 196 patch + CLS.
        #    Il nostro input (1, 128, 1024) con patch 16×16 → 8×64 = 512 patch.
        # ------------------------------------------------------------------
        self._adapt_pos_embed(vit, n_mels, target_length)

        # ------------------------------------------------------------------
        # 4. Rimuovi il classification head di ImageNet (1000 classi) e salva
        #    il backbone senza il layer finale.
        #    torchvision.vit_b_16 ha: vit.heads.head = Linear(768, 1000)
        # ------------------------------------------------------------------
        vit.heads = nn.Identity()  # type: ignore[assignment]
        self.backbone = vit

        # ------------------------------------------------------------------
        # 5. Nuovo classification head per il nostro task
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------ #
    # Setup: adattamento positional embedding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _adapt_pos_embed(vit: nn.Module, n_mels: int, target_length: int) -> None:
        """Interpola i positional embedding per la nuova griglia di patch.

        torchvision ViT-B/16: ``encoder.pos_embedding`` shape (1, 197, 768).
        197 = 196 patch (14×14) + 1 CLS token.
        Nuovo: 512 patch (8×64) + 1 CLS token → shape (1, 513, 768).
        """
        pos_embed = vit.encoder.pos_embedding  # (1, 197, 768)
        cls_pos = pos_embed[:, :1, :]           # (1, 1, 768) — CLS token
        patch_pos = pos_embed[:, 1:, :]         # (1, 196, 768)

        n_old = patch_pos.shape[1]              # 196
        h_old = w_old = int(math.sqrt(n_old))  # 14 × 14

        patch_size = 16
        h_new = n_mels // patch_size            # 128 // 16 = 8
        w_new = target_length // patch_size     # 1024 // 16 = 64

        if h_new * w_new == n_old:
            return  # già compatibile

        # Riforma a griglia 2D e interpola bilinearmente
        patch_pos = patch_pos.reshape(1, h_old, w_old, -1).permute(0, 3, 1, 2)  # (1, D, H, W)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=(h_new, w_new),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h_new * w_new, -1)  # (1, 512, 768)

        new_pos_embed = torch.cat([cls_pos, patch_pos], dim=1)  # (1, 513, 768)
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embed)

    def _freeze_backbone(self) -> None:
        """Congela tutti i parametri del backbone (solo il head è trainable)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Scongela il backbone per fine-tuning completo."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Restituisce l'embedding CLS (prima del classification head).

        Args:
            x: Mel-spectrogram ``(B, 1, n_mels, target_length)``.

        Returns:
            Embedding ``(B, 768)``.
        """
        return self.backbone(x)  # backbone.heads = Identity → restituisce embedding CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward completo: mel-spectrogram → logits.

        Args:
            x: Mel-spectrogram ``(B, 1, n_mels, target_length)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        features = self.forward_features(x)   # (B, 768)
        return self.classifier(features)      # (B, num_classes)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"AudioSpectrogramTransformer("
            f"backbone=ViT-B/16 (torchvision), "
            f"num_classes={self.num_classes}, "
            f"embed_dim={self.embed_dim}, "
            f"params_total={total:,}, params_trainable={trainable:,})"
        )


def build_ast(cfg: dict) -> AudioSpectrogramTransformer:
    """Factory: costruisce un AST da un dict di configurazione.

    Legge le chiavi ``model.*`` e ``dataset.*`` dal dict caricato da YAML.
    Le chiavi mancanti usano i default di ``AudioSpectrogramTransformer``.

    Esempio::

        cfg = yaml.safe_load(open("experiments/configs/baseline_audio.yaml"))
        model = build_ast(cfg)
    """
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})

    return AudioSpectrogramTransformer(
        num_classes=int(ds_cfg.get("num_classes", 25)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        weights_path=model_cfg.get("weights_path", None),
        n_mels=int(ds_cfg.get("n_mels", 128)),
        target_length=int(ds_cfg.get("target_length", 1024)),
        drop_rate=float(model_cfg.get("drop_rate", 0.1)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )


if __name__ == "__main__":
    # Quick smoke test: verifica le shape di input/output e l'interpolazione dei pos-embed.
    print("Smoke test AudioSpectrogramTransformer (torchvision backend)...")
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
