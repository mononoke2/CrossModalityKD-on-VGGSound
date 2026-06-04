"""Loss function per la Knowledge Distillation

Questo modulo raccoglie le loss usate per addestrare lo **Student AST**
(``AudioSpectrogramTransformer``, embedding 768-d) sfruttando un **Teacher
visivo frozen** (``VisionTeacher`` ResNet-50, embedding 2048-d):

- :class:`DistillationLoss` — *response-based* (logit) distillation:
  combina la KL-divergence tra le distribuzioni soft di student e teacher
  (alla temperatura ``T``) con la cross-entropy supervisionata sulle label vere.
  È la formulazione classica di Hinton et al., "Distilling the Knowledge in a
  Neural Network" (2015)::

      L = alpha * T^2 * KL(softmax(z_s / T) || softmax(z_t / T))
          + (1 - alpha) * CE(z_s, y)

- :class:`FeatureDistillationLoss` — *feature-based* distillation:
  MSE tra l'embedding dello student e l'embedding del teacher proiettato nello
  spazio dello student tramite un projection layer trainabile (gli embedding
  hanno dimensioni diverse: AST=768, ResNet-50=2048). Ispirata a Romero et al.,
  "FitNets: Hints for Thin Deep Nets" (2015).

Note sull'integrazione nel training loop (Fase 3)
-------------------------------------------------
- Il teacher è in ``eval()`` e i suoi output vanno detachati dal grafo: queste
  loss applicano comunque ``.detach()`` difensivamente sui tensori del teacher,
  così nessun gradiente risale al teacher congelato.
- :class:`FeatureDistillationLoss` **contiene parametri trainabili** (il
  projection layer). Vanno aggiunti all'optimizer insieme a quelli dello
  student, ad es.::

      params = list(student.parameters()) + list(feat_loss.parameters())
      optimizer = torch.optim.AdamW(params, lr=...)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Logit distillation: KL soft (a temperatura T) + cross-entropy hard.

    Calcola::

        L = alpha * T^2 * KL(softmax(z_s / T) || softmax(z_t / T))
            + (1 - alpha) * CE(z_s, y)

    Il fattore ``T^2`` riscala i gradienti del termine soft per renderli
    confrontabili con quelli del termine hard, come raccomandato in Hinton et
    al. (2015), poiché i gradienti della soft-loss scalano con ``1/T^2``.

    Args:
        alpha: Peso del termine soft (KL) in ``[0, 1]``. ``alpha=1`` usa solo la
            distillation, ``alpha=0`` solo la cross-entropy. Default ``0.5``.
        temperature: Temperatura ``T > 0`` per ammorbidire le distribuzioni.
            Valori più alti (es. 4) producono distribuzioni più morbide e
            trasferiscono più "dark knowledge". Default ``4.0``.
        label_smoothing: Label smoothing per il termine di cross-entropy
            (coerente con il training del teacher). Default ``0.0``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve essere in [0, 1], ricevuto {alpha}")
        if temperature <= 0.0:
            raise ValueError(f"temperature deve essere > 0, ricevuto {temperature}")

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calcola la loss combinata.

        Args:
            student_logits: Logit dello student ``(B, num_classes)``.
            teacher_logits: Logit del teacher ``(B, num_classes)`` (frozen;
                vengono detachati internamente).
            labels: Label intere ``(B,)`` con ``0 <= label < num_classes``.
            return_components: Se ``True`` restituisce anche un dict con i
                singoli termini (``soft``, ``hard``, ``total``) per il logging.

        Returns:
            La loss scalare, oppure ``(loss, components)`` se
            ``return_components=True``.
        """
        T = self.temperature

        # Il teacher è frozen: stacchiamo i suoi logit dal grafo per sicurezza.
        teacher_logits = teacher_logits.detach()

        # Termine soft: KL-divergence tra le distribuzioni a temperatura T.
        # F.kl_div si aspetta input in log-prob e target in prob; reduction
        # "batchmean" media sul batch (corretto dimensionalmente rispetto a
        # "mean", che dividerebbe anche per num_classes).
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        # Termine hard: cross-entropy supervisionata sulle label vere.
        hard_loss = self.ce(student_logits, labels)

        total = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss

        if return_components:
            return total, {"soft": soft_loss, "hard": hard_loss, "total": total}
        return total

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, temperature={self.temperature}"


class FeatureDistillationLoss(nn.Module):
    """Feature distillation: MSE tra embedding dello student e del teacher proiettato.

    Gli embedding di student e teacher vivono in spazi di dimensione diversa
    (AST=768, ResNet-50=2048), quindi non sono direttamente confrontabili. Un
    **projection layer** trainabile mappa l'embedding del teacher nello spazio
    dello student, dove si calcola l'MSE::

        L = MSE( h_s , proj(h_t) )

    La proiezione si applica al teacher (frozen) ed è l'unica parte trainabile
    di questa loss: va quindi inclusa nei parametri passati all'optimizer.

    Args:
        student_dim: Dimensione dell'embedding dello student. Default ``768``.
        teacher_dim: Dimensione dell'embedding del teacher. Default ``2048``.
        hidden_dim: Se fornita, usa una proiezione a due strati
            ``teacher_dim → hidden_dim → student_dim`` con BatchNorm + ReLU.
            Se ``None`` (default) usa una singola ``Linear(teacher_dim, student_dim)``.
        normalize: Se ``True``, applica L2-normalization agli embedding prima
            dell'MSE (rende la loss invariante alla scala, utile se le norme di
            student e teacher differiscono molto). Default ``False``.
    """

    def __init__(
        self,
        student_dim: int = 768,
        teacher_dim: int = 2048,
        hidden_dim: int | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.normalize = bool(normalize)

        if hidden_dim is None:
            self.projection: nn.Module = nn.Linear(self.teacher_dim, self.student_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.teacher_dim, int(hidden_dim)),
                nn.BatchNorm1d(int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), self.student_dim),
            )

        self.mse = nn.MSELoss()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """Calcola l'MSE tra embedding dello student e del teacher proiettato.

        Args:
            student_features: Embedding dello student ``(B, student_dim)``.
            teacher_features: Embedding del teacher ``(B, teacher_dim)`` (frozen;
                viene detachato internamente).

        Returns:
            La loss MSE scalare.
        """
        # Il teacher è frozen: il suo embedding non deve propagare gradiente.
        # La proiezione resta trainabile (i suoi pesi sono parametri del modulo).
        teacher_features = teacher_features.detach()
        projected_teacher = self.projection(teacher_features)  # (B, student_dim)

        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=1)
            projected_teacher = F.normalize(projected_teacher, p=2, dim=1)

        return self.mse(student_features, projected_teacher)

    def extra_repr(self) -> str:
        return (
            f"student_dim={self.student_dim}, teacher_dim={self.teacher_dim}, "
            f"normalize={self.normalize}"
        )


if __name__ == "__main__":
    # Smoke test: verifica shape, range dei valori e flusso dei gradienti.
    print("Smoke test losses (DistillationLoss + FeatureDistillationLoss)...")

    torch.manual_seed(0)
    B, num_classes = 4, 25
    student_dim, teacher_dim = 768, 2048

    # --- DistillationLoss ---------------------------------------------------
    student_logits = torch.randn(B, num_classes, requires_grad=True)
    teacher_logits = torch.randn(B, num_classes)  # teacher frozen → no grad
    labels = torch.randint(0, num_classes, (B,))

    kd = DistillationLoss(alpha=0.7, temperature=4.0, label_smoothing=0.1)
    print(kd)
    loss, comps = kd(student_logits, teacher_logits, labels, return_components=True)
    assert loss.dim() == 0, "La loss deve essere uno scalare"
    assert loss.item() >= 0, "La loss combinata deve essere non negativa"
    loss.backward()
    assert student_logits.grad is not None, "Lo student deve ricevere gradiente"
    print(
        f"  KD loss={loss.item():.4f} "
        f"(soft={comps['soft'].item():.4f}, hard={comps['hard'].item():.4f})"
    )

    # Caso degenere: student == teacher e alpha=1 → soft loss ~ 0.
    z = torch.randn(B, num_classes)
    kd_only = DistillationLoss(alpha=1.0, temperature=2.0)
    soft_zero = kd_only(z.clone().requires_grad_(True), z.clone(), labels)
    assert soft_zero.item() < 1e-5, f"KL tra distribuzioni identiche ~0, got {soft_zero.item()}"
    print(f"  KD (student==teacher, alpha=1) loss={soft_zero.item():.2e} — OK")

    # --- FeatureDistillationLoss -------------------------------------------
    student_feat = torch.randn(B, student_dim, requires_grad=True)
    teacher_feat = torch.randn(B, teacher_dim)  # teacher frozen → no grad

    feat_loss = FeatureDistillationLoss(student_dim=student_dim, teacher_dim=teacher_dim)
    print(feat_loss)
    fl = feat_loss(student_feat, teacher_feat)
    assert fl.dim() == 0 and fl.item() >= 0, "MSE scalare non negativo"
    fl.backward()
    assert student_feat.grad is not None, "Lo student deve ricevere gradiente"
    # La proiezione (teacher-side) deve essere trainabile e aver ricevuto grad.
    proj_param = next(feat_loss.projection.parameters())
    assert proj_param.requires_grad and proj_param.grad is not None, (
        "Il projection layer deve essere trainabile e ricevere gradiente"
    )
    print(f"  Feature MSE loss={fl.item():.4f}")

    # Variante con proiezione a due strati + normalizzazione.
    feat_loss2 = FeatureDistillationLoss(
        student_dim=student_dim, teacher_dim=teacher_dim, hidden_dim=512, normalize=True
    )
    fl2 = feat_loss2(torch.randn(B, student_dim), torch.randn(B, teacher_dim))
    assert fl2.item() >= 0
    print(f"  Feature MSE (hidden=512, normalize) loss={fl2.item():.4f}")

    print("Smoke test PASSATO.")
