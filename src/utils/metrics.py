"""Metriche e utility di valutazione per Track 24 (Cross-Modal KD).

Raccoglie le funzioni richieste in fase di valutazione comparativa tra
Teacher (ResNet-50), Baseline audio (AST) e Student distillato:

- ``top_k_accuracy``        accuracy top-k (top-1 / top-5)
- ``compute_confusion_matrix``  matrice di confusione (num_classes x num_classes)
- ``model_size_mb``         dimensione del modello in MB (per il confronto size)
- ``measure_inference_time_ms``  latenza media di inferenza in ms (per il confronto speed)

Tutte le funzioni lavorano con tensori PyTorch e non hanno effetti collaterali.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


@torch.no_grad()
def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int] = (1, 5),
) -> dict[int, float]:
    """Calcola l'accuracy top-k per ciascun k richiesto.

    Args:
        logits: tensore ``(B, num_classes)`` con i punteggi/logits del modello.
        targets: tensore ``(B,)`` con gli indici di classe corretti.
        ks: valori di k per cui calcolare l'accuracy (default top-1 e top-5).

    Returns:
        Dizionario ``{k: accuracy_in_[0,1]}``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits deve essere 2D (B, num_classes), ricevuto {tuple(logits.shape)}")
    batch_size = targets.size(0)
    if batch_size == 0:
        return {k: 0.0 for k in ks}

    num_classes = logits.size(1)
    maxk = min(max(ks), num_classes)

    # Indici delle maxk classi a punteggio più alto, ordinati per probabilità decrescente.
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # (maxk, B) bool

    results: dict[int, float] = {}
    for k in ks:
        k_eff = min(k, num_classes)
        correct_k = correct[:k_eff].any(dim=0).float().sum().item()
        results[k] = correct_k / batch_size
    return results


@torch.no_grad()
def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Costruisce la matrice di confusione.

    Args:
        preds: tensore ``(B,)`` con le classi predette (argmax dei logits).
        targets: tensore ``(B,)`` con le classi corrette.
        num_classes: numero totale di classi.

    Returns:
        Tensore ``(num_classes, num_classes)`` di tipo ``long`` dove l'elemento
        ``[i, j]`` conta i campioni con etichetta vera ``i`` predetti come ``j``.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"preds {tuple(preds.shape)} e targets {tuple(targets.shape)} devono avere la stessa forma")
    preds = preds.flatten().long()
    targets = targets.flatten().long()

    # Indice lineare riga*num_classes + colonna, poi bincount: efficiente e vettoriale.
    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def model_size_mb(model: nn.Module, include_buffers: bool = True) -> float:
    """Stima la dimensione del modello in megabyte (MB).

    Somma i byte occupati dai parametri (e, opzionalmente, dai buffer come le
    statistiche di BatchNorm), così come finirebbero su disco/VRAM.

    Args:
        model: modulo PyTorch.
        include_buffers: se includere i buffer oltre ai parametri.

    Returns:
        Dimensione in MB (base 1024^2).
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = 0
    if include_buffers:
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


@torch.no_grad()
def measure_inference_time_ms(
    model: nn.Module,
    example_input: torch.Tensor | Iterable[torch.Tensor],
    n_runs: int = 100,
    warmup: int = 10,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Misura la latenza media di inferenza in millisecondi.

    Esegue alcune iterazioni di warmup (importanti su GPU per stabilizzare i
    clock e completare le allocazioni) e poi cronometra ``n_runs`` forward pass.
    Su CUDA usa eventi CUDA e sincronizza per una misura accurata.

    Args:
        model: modulo da valutare (verrà messo in ``eval()``).
        example_input: un singolo tensore o una tupla/lista di tensori da passare
            al ``forward`` del modello.
        n_runs: numero di forward cronometrati.
        warmup: numero di forward di warmup non cronometrati.
        device: device su cui misurare; se ``None`` usa quello del modello.

    Returns:
        Dizionario con ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model = model.to(device).eval()

    # Normalizza l'input in una tupla di argomenti posizionali.
    if isinstance(example_input, torch.Tensor):
        inputs: tuple[torch.Tensor, ...] = (example_input.to(device),)
    else:
        inputs = tuple(t.to(device) for t in example_input)

    use_cuda = device.type == "cuda"

    # Warmup.
    for _ in range(max(0, warmup)):
        model(*inputs)
    if use_cuda:
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    if use_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        for _ in range(n_runs):
            start_evt.record()
            model(*inputs)
            end_evt.record()
            torch.cuda.synchronize(device)
            times_ms.append(start_evt.elapsed_time(end_evt))  # già in ms
    else:
        import time

        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(*inputs)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times_ms)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std(unbiased=False).item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
    }
