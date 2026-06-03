"""Script di training per la Baseline Audio (Fase 1) — Track 24.

Addestra un ``AudioSpectrogramTransformer`` (AST) su log-mel-spectrogram
con sola cross-entropy loss, senza alcun meccanismo di distillazione.
Questo esperimento costituisce il punto di riferimento (baseline) con cui
il Teacher visivo e lo Student distillato saranno confrontati nella Fase 4.

Uso (locale, debug):
    python -m src.training.train_baseline_audio \\
        --config experiments/configs/baseline_audio.yaml

Uso (cluster, via SLURM):
    sbatch experiments/scripts/submit_job.sh train_baseline_audio \\
        --config experiments/configs/baseline_audio.yaml

Resume da checkpoint:
    python -m src.training.train_baseline_audio \\
        --config experiments/configs/baseline_audio.yaml \\
        --resume experiments/checkpoints/baseline_audio/best.pth

Argomenti CLI:
    --config    Path al file YAML di configurazione (default: experiments/configs/baseline_audio.yaml).
    --resume    Path a un checkpoint .pth da cui riprendere il training.
    --output-dir Override della directory di output per checkpoint e log.
    --device    Override del device (es. "cpu", "cuda:0").
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import interni al progetto
# ---------------------------------------------------------------------------
# Aggiunge la root del progetto al path così i moduli src.* sono raggiungibili
# sia da `python -m` che da `python src/training/train_baseline_audio.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.vggsound import VGGSoundDataset
from src.models.ast_model import build_ast
from src.utils.early_stopping import EarlyStopping
from src.utils.logger import ExperimentLogger
from src.utils.metrics import top_k_accuracy

import yaml


# ---------------------------------------------------------------------------
# Gestione del segnale SLURM USR1 (checkpointing d'emergenza pre-timeout)
# ---------------------------------------------------------------------------
_CHECKPOINT_REQUESTED = False

def _sigusr1_handler(signum, frame):  # noqa: ANN001
    global _CHECKPOINT_REQUESTED
    _CHECKPOINT_REQUESTED = True
    print("\n[SLURM] Ricevuto segnale SIGUSR1: checkpoint d'emergenza pianificato.")

signal.signal(signal.SIGUSR1, _sigusr1_handler)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Carica e unisce common.yaml + config specifica (supporta chiave ``extends``)."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "extends" in cfg:
        base_path = Path(config_path).parent / cfg.pop("extends")
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        # Deep merge: la config specifica sovrascrive la base
        base_cfg = _deep_merge(base_cfg, cfg)
        return base_cfg

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge ricorsivo: override sovrascrive base a livello di foglia."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: Path,
    filename: str = "last.pth",
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, optimizer, scheduler, scaler, early_stopping) -> int:
    """Carica un checkpoint e restituisce l'epoch di ripresa."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if "early_stopping" in ckpt and early_stopping is not None:
        early_stopping.load_state_dict(ckpt["early_stopping"])
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"[Resume] Checkpoint caricato da {path}. Ripresa dall'epoch {start_epoch}.")
    return start_epoch


# ---------------------------------------------------------------------------
# Training e validation loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None,
    gradient_accumulation_steps: int,
    epoch: int,
    logger: ExperimentLogger,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, (mel, labels) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=scaler is not None):
            logits = model(mel)
            loss = criterion(logits, labels)
            loss = loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            logger.log(
                f"  Epoch {epoch} [{batch_idx + 1}/{n_batches}]  "
                f"Loss: {total_loss / (batch_idx + 1):.4f}"
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Restituisce (val_loss, top1_acc, top5_acc)."""
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for mel, labels in loader:
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(mel)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    accs = top_k_accuracy(logits_cat, targets_cat, ks=(1, 5))

    return total_loss / len(loader), accs[1], accs[5]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AST Audio Baseline — Track 24")
    parser.add_argument(
        "--config",
        default="experiments/configs/baseline_audio.yaml",
        help="Path al file YAML di configurazione.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path a un checkpoint .pth da cui riprendere il training.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override della directory base per checkpoint e log.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override del device (es. 'cpu', 'cuda:0').",
    )
    return parser.parse_args()


def main() -> None:
    global _CHECKPOINT_REQUESTED
    args = parse_args()

    # -- Config ----------------------------------------------------------
    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # -- Device ----------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # -- Output dir ------------------------------------------------------
    output_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "experiments" / "checkpoints" / "baseline_audio"
    )
    log_dir = _PROJECT_ROOT / "experiments" / "logs" / "baseline_audio"

    logger = ExperimentLogger(
        name="baseline_audio",
        log_dir=str(log_dir),
        use_tensorboard=True,
        use_wandb=False,
        config=cfg,
    )
    logger.log(f"Config: {cfg}")
    logger.log(f"Output dir: {output_dir}")

    # -- Dataset e DataLoader -------------------------------------------
    common_ds_kwargs = dict(
        config=args.config,
        modality="audio",
        require_files=True,
    )
    train_dataset = VGGSoundDataset(split="train", **common_ds_kwargs)
    val_dataset = VGGSoundDataset(split="val", **common_ds_kwargs)

    logger.log(f"Train: {len(train_dataset)} campioni | Val: {len(val_dataset)} campioni")

    num_workers = int(train_cfg.get("num_workers", 4))
    batch_size = int(train_cfg.get("batch_size", 32))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # -- Modello --------------------------------------------------------
    model = build_ast(cfg).to(device)
    logger.log(str(model))

    # -- Ottimizzatore e Scheduler --------------------------------------
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 30))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler_name = train_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        # Cosine annealing con warm-up lineare manuale (prima del loop principale)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 1e-2
        )
    else:
        scheduler = None

    # -- Mixed precision ------------------------------------------------
    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # -- Loss -----------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # -- Early stopping -------------------------------------------------
    patience = int(train_cfg.get("patience", 7))
    early_stopping = EarlyStopping(
        patience=patience,
        mode="max",
        min_delta=1e-4,
        checkpoint_path=str(output_dir / "best.pth"),
    )

    # -- Resume ---------------------------------------------------------
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, early_stopping
        )

    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))

    # -- Training loop --------------------------------------------------
    logger.log(f"Inizio training: {epochs} epoche, LR={lr}, batch={batch_size}, device={device}")
    best_val_top1 = 0.0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Warm-up lineare del LR nelle prime `warmup_epochs` epoche
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            if scheduler is not None:
                scheduler.step()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, gradient_accumulation_steps, epoch, logger,
        )

        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.log(
            f"Epoch {epoch}/{epochs} — "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_top1={val_top1*100:.2f}% | val_top5={val_top5*100:.2f}% | "
            f"LR={current_lr:.2e} | {epoch_time:.0f}s"
        )

        # Logging scalars per TensorBoard
        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)
        logger.log_scalar("Acc/val_top1", val_top1, epoch)
        logger.log_scalar("Acc/val_top5", val_top5, epoch)
        logger.log_scalar("LR", current_lr, epoch)

        # Aggiorna best e salva checkpoint
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "early_stopping": early_stopping.state_dict(),
            "val_top1": val_top1,
            "cfg": cfg,
        }

        # Checkpoint periodico (ogni 5 epoche + last)
        save_checkpoint(checkpoint_state, output_dir, "last.pth")
        if epoch % 5 == 0:
            save_checkpoint(checkpoint_state, output_dir, f"epoch_{epoch:03d}.pth")

        # Early stopping (salva automaticamente best.pth tramite EarlyStopping)
        early_stopping(val_top1, checkpoint_state)
        if early_stopping.should_stop:
            logger.log(f"Early stopping attivato all'epoch {epoch}. Best val_top1: {best_val_top1*100:.2f}%")
            break

        # Checkpoint d'emergenza da segnale SLURM
        if _CHECKPOINT_REQUESTED:
            logger.log("[SLURM] Checkpoint d'emergenza salvato.")
            _CHECKPOINT_REQUESTED = False
            save_checkpoint(checkpoint_state, output_dir, "emergency.pth")

    logger.log(f"Training completato. Best val_top1: {best_val_top1 * 100:.2f}%")
    logger.close()


if __name__ == "__main__":
    main()
