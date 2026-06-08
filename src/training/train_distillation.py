"""Script di training per la Distillazione Cross-Modale 

Addestra un ``AudioSpectrogramTransformer`` (Student AST) usando la conoscenza
trasferita da un ``VisionTeacher`` (ResNet-50) pre-addestrato e congelato.

La loss totale combina tre termini:
    L_total = (1 - feature_weight) * L_distill + feature_weight * L_feat

dove:
    L_distill = alpha * T^2 * KL(softmax(z_s/T) || softmax(z_t/T))
                + (1 - alpha) * CE(z_s, y)
    L_feat    = MSE(h_s, proj(h_t))         [se use_feature_distillation=True]

Il teacher elabora il frame visivo (``modality="video"``); lo student elabora
il mel-spectrogram audio (``modality="audio"``). Entrambi i tensori vengono
restituiti dal dataset impostando ``modality="both"``.

Uso (locale, debug):
    python -m src.training.train_distillation \\
        --config experiments/configs/distillation.yaml

Uso (cluster, via SLURM):
    sbatch experiments/scripts/submit_job.sh src.training.train_distillation \\
        --config experiments/configs/distillation.yaml

Override alpha da CLI (per l'ablation senza duplicare file YAML):
    python -m src.training.train_distillation \\
        --config experiments/configs/distillation.yaml \\
        --alpha 0.3

Resume da checkpoint:
    python -m src.training.train_distillation \\
        --config experiments/configs/distillation.yaml \\
        --resume experiments/checkpoints/distillation/best.pth

Argomenti CLI:
    --config      Path al file YAML di configurazione.
    --alpha       Override del valore di alpha (distillation weight).
    --resume      Path a un checkpoint .pth da cui riprendere il training.
    --output-dir  Override della directory di output.
    --device      Override del device (es. "cpu", "cuda:0").
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
# Aggiunge la root del progetto al path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.vggsound import VGGSoundDataset
from src.models.ast_model import build_ast
from src.models.light_audio_models import build_light_audio_student
from src.models.vision_teacher import build_vision_teacher
from src.training.losses import DistillationLoss, FeatureDistillationLoss
from src.utils.early_stopping import EarlyStopping
from src.utils.logger import ExperimentLogger
from src.utils.metrics import top_k_accuracy

import yaml


# ---------------------------------------------------------------------------
# Gestione segnale SLURM USR1 (checkpointing d'emergenza pre-timeout)
# ---------------------------------------------------------------------------
_CHECKPOINT_REQUESTED = False

def _sigusr1_handler(signum, frame):  # noqa: ANN001
    global _CHECKPOINT_REQUESTED
    _CHECKPOINT_REQUESTED = True
    print("\n[SLURM] Ricevuto segnale SIGUSR1: checkpoint d'emergenza pianificato.")

signal.signal(signal.SIGUSR1, _sigusr1_handler)


# ---------------------------------------------------------------------------
# Utility: config, seed, checkpoint
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Carica e unisce ricorsivamente le config YAML (supporta catene di ``extends``)."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "extends" in cfg:
        base_path = Path(config_path).parent / cfg.pop("extends")
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        # Se anche la base ha un 'extends', lo risolviamo ricorsivamente.
        if "extends" in base_cfg:
            base_cfg = load_config(str(base_path))
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


def load_checkpoint(
    path: str | Path,
    student: nn.Module,
    feat_loss: FeatureDistillationLoss | None,
    optimizer,
    scheduler,
    scaler,
    early_stopping,
) -> int:
    """Carica un checkpoint e restituisce l'epoch di ripresa."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["student"])
    if feat_loss is not None and "feat_loss" in ckpt:
        feat_loss.load_state_dict(ckpt["feat_loss"])
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
# Factory student
# ---------------------------------------------------------------------------

def build_student(model_type: str, cfg: dict) -> nn.Module:
    """Costruisce lo student corretto in base a ``model_type``.

    ``"ast"`` usa il backbone ViT-B/16 (comportamento storico).
    ``"efficientnet_b0_audio"`` e ``"mobilenet_v3_small_audio"`` usano i
    backbone CNN leggeri definiti in ``light_audio_models.py`` (Extra 3).
    """
    if model_type == "ast":
        return build_ast(cfg)
    return build_light_audio_student(model_type, cfg)


# ---------------------------------------------------------------------------
# Caricamento Teacher frozen
# ---------------------------------------------------------------------------

def load_frozen_teacher(cfg: dict, device: torch.device) -> nn.Module:
    """Costruisce e carica il checkpoint del teacher; lo congela completamente."""
    teacher_cfg = cfg.get("teacher", {})
    checkpoint_path = teacher_cfg.get("checkpoint", "experiments/checkpoints/teacher_vision/best.pth")

    # Costruiamo il modello con le chiavi della sezione 'teacher' mappate su 'model'
    teacher_build_cfg = {
        "model": {
            "type": teacher_cfg.get("type", "resnet50"),
            "pretrained": False,            # non riscarichiamo ImageNet, carichiamo il ckpt fine-tunato
            "freeze_until": None,           # sarà congelato manualmente sotto
        },
        "dataset": cfg.get("dataset", {}),
    }
    teacher = build_vision_teacher(teacher_build_cfg).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint del teacher non trovato: {checkpoint_path}\n"
            "Assicurati di aver eseguito la Fase 2 e di aver sincronizzato i "
            "checkpoint dal cluster con sync_from_cluster.sh."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Il checkpoint può contenere 'model' (formato usato da train_teacher.py)
    state_dict = ckpt.get("model", ckpt)
    teacher.load_state_dict(state_dict)

    # Congela completamente il teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"[Teacher] Checkpoint caricato da: {checkpoint_path}")
    print(f"[Teacher] {teacher}")
    return teacher


# ---------------------------------------------------------------------------
# Training loop per la distillazione
# ---------------------------------------------------------------------------

def train_one_epoch_distill(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer,
    distill_loss: DistillationLoss,
    feat_loss: FeatureDistillationLoss | None,
    feature_weight: float,
    device: torch.device,
    scaler: GradScaler | None,
    gradient_accumulation_steps: int,
    epoch: int,
    logger: ExperimentLogger,
) -> dict[str, float]:
    """Esegue un epoch di training con distillazione.

    Restituisce un dict con le loss medie:
        {"total": ..., "distill": ..., "feat": ..., "soft": ..., "hard": ...}
    """
    student.train()
    teacher.eval()  # il teacher rimane sempre in eval

    tot_total = tot_distill = tot_soft = tot_hard = tot_feat = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, (mel, frame, labels) in enumerate(loader):
        mel    = mel.to(device, non_blocking=True)      # (B, 1, 128, 1024) — input student
        frame  = frame.to(device, non_blocking=True)    # (B, 3, 224, 224)  — input teacher
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=scaler is not None):
            # Forward teacher (no grad, già congelato)
            with torch.no_grad():
                teacher_logits   = teacher(frame)              # (B, num_classes)
                teacher_features = teacher.forward_features(frame)  # (B, 2048)

            # Forward student
            student_features = student.forward_features(mel)   # (B, 768)
            student_logits   = student.classifier(student_features)  # (B, num_classes)

            # Distillation loss (logit-based)
            distill_val, components = distill_loss(
                student_logits, teacher_logits, labels, return_components=True
            )

            # Feature distillation loss (opzionale)
            if feat_loss is not None:
                feat_val = feat_loss(student_features, teacher_features)
                loss = (1.0 - feature_weight) * distill_val + feature_weight * feat_val
                tot_feat += feat_val.item()
            else:
                loss = distill_val

            loss = loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad]
                    + (list(feat_loss.parameters()) if feat_loss else []),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad]
                    + (list(feat_loss.parameters()) if feat_loss else []),
                    max_norm=1.0,
                )
                optimizer.step()
            optimizer.zero_grad()

        tot_total   += loss.item() * gradient_accumulation_steps
        tot_distill += distill_val.item()
        tot_soft    += components["soft"].item()
        tot_hard    += components["hard"].item()

        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            logger.info(
                "  Epoch %d [%d/%d]  Loss: %.4f (distill=%.4f, feat=%.4f)",
                epoch, batch_idx + 1, n_batches,
                tot_total / (batch_idx + 1),
                tot_distill / (batch_idx + 1),
                tot_feat / (batch_idx + 1) if feat_loss else 0.0,
            )

    n = n_batches
    return {
        "total":   tot_total / n,
        "distill": tot_distill / n,
        "soft":    tot_soft / n,
        "hard":    tot_hard / n,
        "feat":    tot_feat / n if feat_loss else 0.0,
    }


@torch.no_grad()
def validate_student(
    student: nn.Module,
    loader: DataLoader,
    distill_loss: DistillationLoss,
    teacher: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Valida lo student solo su audio. Restituisce (val_loss, top1, top5).

    La val_loss è calcolata con la sola CrossEntropy (alpha=0 equivalente)
    per rendere i risultati comparabili con la baseline e il teacher.
    """
    ce = nn.CrossEntropyLoss()
    student.eval()
    teacher.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for mel, _frame, labels in loader:
        mel    = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = student(mel)                   # solo audio: nessun frame richiesto
        loss   = ce(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits_cat  = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    accs = top_k_accuracy(logits_cat, targets_cat, ks=(1, 5))

    return total_loss / len(loader), accs[1], accs[5]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Student AST con Cross-Modal KD — Track 24 Fase 3"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/distillation.yaml",
        help="Path al file YAML di configurazione.",
    )
    parser.add_argument(
        "--student-type",
        default="ast",
        choices=["ast", "efficientnet_b0_audio", "mobilenet_v3_small_audio"],
        help="Architettura dello student (default: 'ast').",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override di distillation.alpha (utile per l'ablation senza duplicare YAML).",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _CHECKPOINT_REQUESTED
    args = parse_args()

    # -- Config --------------------------------------------------------------
    cfg = load_config(args.config)
    ds_cfg       = cfg.get("dataset", {})
    train_cfg    = cfg.get("training", {})
    distill_cfg  = cfg.get("distillation", {})

    # Override alpha da CLI
    if args.alpha is not None:
        distill_cfg["alpha"] = args.alpha
        cfg["distillation"]["alpha"] = args.alpha

    alpha           = float(distill_cfg.get("alpha", 0.7))
    temperature     = float(distill_cfg.get("temperature", 4.0))
    use_feat_distil = bool(distill_cfg.get("use_feature_distillation", True))
    feature_weight  = float(distill_cfg.get("feature_weight", 0.3)) if use_feat_distil else 0.0

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # -- Device --------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # -- Output dir ----------------------------------------------------------
    student_type = args.student_type
    if student_type == "ast":
        run_name = f"distillation_alpha{alpha:.1f}".replace(".", "")
    else:
        model_suffix = student_type.replace("_audio", "")
        run_name = f"distillation_{model_suffix}"

    output_dir = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "experiments" / "checkpoints" / run_name
    )
    log_dir = _PROJECT_ROOT / "experiments" / "logs" / run_name

    logger = ExperimentLogger(
        name=run_name,
        log_dir=str(log_dir),
        use_tensorboard=True,
        use_wandb=False,
        config=cfg,
    )
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "Distillazione — alpha=%.1f | T=%.1f | feat_distil=%s | feat_weight=%.2f",
        alpha, temperature, use_feat_distil, feature_weight,
    )

    # -- Dataset e DataLoader ------------------------------------------------
    # Il dataset con modality="both" restituisce (mel, frame, label)
    common_ds_kwargs = dict(
        config=args.config,
        modality="both",
        require_files=True,
    )
    train_dataset = VGGSoundDataset(split="train", **common_ds_kwargs)
    val_dataset   = VGGSoundDataset(split="val",   **common_ds_kwargs)

    logger.info(
        "Train: %d campioni | Val: %d campioni",
        len(train_dataset), len(val_dataset),
    )

    num_workers = int(train_cfg.get("num_workers", 4))
    batch_size  = int(train_cfg.get("batch_size", 32))

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

    # -- Teacher frozen ------------------------------------------------------
    teacher = load_frozen_teacher(cfg, device)

    # -- Student (AST reinizializzato, stessa architettura della Fase 1) ------
    student_cfg = cfg.get("student", {})
    student_type = args.student_type or student_cfg.get("type", "ast")
    student_build_cfg = {
        "model": {
            "type": student_type,
            "pretrained": student_cfg.get("pretrained", True),
            "weights_path": student_cfg.get("weights_path", None),
            "drop_rate": float(student_cfg.get("drop_rate", 0.1)),
            "freeze_backbone": bool(student_cfg.get("freeze_backbone", False)),
        },
        "dataset": ds_cfg,
    }
    student = build_student(student_type, student_build_cfg).to(device)
    logger.info("Student: %s", student)

    # -- Loss ----------------------------------------------------------------
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    distill_loss = DistillationLoss(
        alpha=alpha,
        temperature=temperature,
        label_smoothing=label_smoothing,
    )

    feat_loss: FeatureDistillationLoss | None = None
    if use_feat_distil:
        feat_loss = FeatureDistillationLoss(
            student_dim=student.embed_dim,   # 768
            teacher_dim=teacher.embed_dim,   # 2048
            hidden_dim=512,                  # proiezione a due strati per migliore adattamento
            normalize=False,
        ).to(device)
        logger.info("FeatureDistillationLoss: student_dim=768, teacher_dim=2048, hidden_dim=512")

    # -- Ottimizzatore (student + projection layer di feat_loss) -------------
    lr           = float(train_cfg.get("lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))
    epochs       = int(train_cfg.get("epochs", 30))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    trainable_params = list(student.parameters())
    if feat_loss is not None:
        trainable_params += list(feat_loss.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler_name = train_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 1e-2
        )
    else:
        scheduler = None

    # -- Mixed precision -----------------------------------------------------
    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None

    # -- Early stopping ------------------------------------------------------
    patience = int(train_cfg.get("patience", 7))
    early_stopping = EarlyStopping(patience=patience, mode="max", min_delta=1e-4)

    # -- Resume --------------------------------------------------------------
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, student, feat_loss, optimizer, scheduler, scaler, early_stopping
        )

    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))

    # -- Training loop -------------------------------------------------------
    logger.info(
        "Inizio training: %d epoche, LR=%s, batch=%d, device=%s",
        epochs, lr, batch_size, device,
    )
    best_val_top1 = 0.0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Warm-up lineare del LR
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            if scheduler is not None:
                scheduler.step()

        # Train
        losses = train_one_epoch_distill(
            student, teacher, train_loader, optimizer,
            distill_loss, feat_loss, feature_weight,
            device, scaler, gradient_accumulation_steps, epoch, logger,
        )

        # Validate (solo audio)
        val_loss, val_top1, val_top5 = validate_student(
            student, val_loader, distill_loss, teacher, device
        )

        epoch_time  = time.time() - epoch_start
        current_lr  = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d — train_loss=%.4f (distill=%.4f feat=%.4f) | "
            "val_loss=%.4f | val_top1=%.2f%% | val_top5=%.2f%% | LR=%.2e | %.0fs",
            epoch, epochs,
            losses["total"], losses["distill"], losses["feat"],
            val_loss, val_top1 * 100, val_top5 * 100,
            current_lr, epoch_time,
        )

        # TensorBoard
        logger.log_scalar("Loss/train_total",   losses["total"],   epoch)
        logger.log_scalar("Loss/train_distill", losses["distill"], epoch)
        logger.log_scalar("Loss/train_soft",    losses["soft"],    epoch)
        logger.log_scalar("Loss/train_hard",    losses["hard"],    epoch)
        logger.log_scalar("Loss/train_feat",    losses["feat"],    epoch)
        logger.log_scalar("Loss/val",           val_loss,          epoch)
        logger.log_scalar("Acc/val_top1",       val_top1,          epoch)
        logger.log_scalar("Acc/val_top5",       val_top5,          epoch)
        logger.log_scalar("LR",                 current_lr,        epoch)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1

        checkpoint_state = {
            "epoch":         epoch,
            "student":       student.state_dict(),
            "feat_loss":     feat_loss.state_dict() if feat_loss else None,
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict() if scheduler else None,
            "scaler":        scaler.state_dict() if scaler else None,
            "early_stopping": early_stopping.state_dict(),
            "val_top1":      val_top1,
            "alpha":         alpha,
            "cfg":           cfg,
        }

        save_checkpoint(checkpoint_state, output_dir, "last.pth")
        if epoch % 5 == 0:
            save_checkpoint(checkpoint_state, output_dir, f"epoch_{epoch:03d}.pth")

        improved = early_stopping.step(val_top1)
        if improved:
            save_checkpoint(checkpoint_state, output_dir, "best.pth")
        if early_stopping.should_stop:
            logger.info(
                "Early stopping attivato all'epoch %d. Best val_top1: %.2f%%",
                epoch, best_val_top1 * 100,
            )
            break

        if _CHECKPOINT_REQUESTED:
            logger.info("[SLURM] Checkpoint d'emergenza salvato.")
            _CHECKPOINT_REQUESTED = False
            save_checkpoint(checkpoint_state, output_dir, "emergency.pth")

    logger.info(
        "Training completato. Best val_top1: %.2f%% (alpha=%.1f)",
        best_val_top1 * 100, alpha,
    )
    logger.close()


if __name__ == "__main__":
    main()
