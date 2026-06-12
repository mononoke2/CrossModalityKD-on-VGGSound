#!/usr/bin/env python3
"""Script per plottare e confrontare le curve di addestramento dei modelli distillati.

Legge i file run.log dalle cartelle degli esperimenti di distillazione:
- experiments/logs/distillation_alpha03/distillation_alpha03/run.log
- experiments/logs/distillation_alpha05/distillation_alpha05/run.log
- experiments/logs/distillation_alpha07/distillation_alpha07/run.log
- experiments/logs/distillation_alpha09/distillation_alpha09/run.log

Genera grafici comparativi per:
1. Loss di addestramento totale
2. Loss di validazione (Cross-Entropy supervisionata)
3. Val Top-1 Accuracy
4. Andamento delle componenti della loss (distill vs feat) per ciascun alpha.

Salva il risultato in figures/distillation_ablation_curves.png.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_distill_log(log_path: Path) -> dict[str, list[float]]:
    """Parsa il file run.log di distillazione estraendo le metriche per epoca."""
    epochs: list[float] = []
    train_losses: list[float] = []
    distill_losses: list[float] = []
    feat_losses: list[float] = []
    val_losses: list[float] = []
    val_top1: list[float] = []
    val_top5: list[float] = []
    lrs: list[float] = []

    # Regex per catturare la riga di riassunto epoca:
    # 2026-06-04 17:26:46 | INFO    | Epoch 1/30 — train_loss=0.9324 (distill=1.2683 feat=0.1486) | val_loss=0.7126 | val_top1=80.72% | val_top5=98.95% | LR=1.00e-05 | 120s
    epoch_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/\d+\s+—\s+"
        r"train_loss=(?P<train_loss>[\d\.]+)\s+"
        r"\(distill=(?P<distill>[\d\.]+)\s+feat=(?P<feat>[\d\.]+)\)\s*\|\s*"
        r"val_loss=(?P<val_loss>[\d\.]+)\s*\|\s*"
        r"val_top1=(?P<val_top1>[\d\.]+)%\s*\|\s*"
        r"val_top5=(?P<val_top5>[\d\.]+)%\s*\|\s*"
        r"LR=(?P<lr>[\d\.\-e]+)"
    )

    if not log_path.exists():
        print(f"[WARN] File di log non trovato: {log_path}")
        return {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Se ricomincia un training nello stesso log, resettiamo
            if "Inizio training:" in line or "Configurazione esperimento:" in line:
                epochs.clear()
                train_losses.clear()
                distill_losses.clear()
                feat_losses.clear()
                val_losses.clear()
                val_top1.clear()
                val_top5.clear()
                lrs.clear()

            match = epoch_pattern.search(line)
            if match:
                epochs.append(float(match.group("epoch")))
                train_losses.append(float(match.group("train_loss")))
                distill_losses.append(float(match.group("distill")))
                feat_losses.append(float(match.group("feat")))
                val_losses.append(float(match.group("val_loss")))
                val_top1.append(float(match.group("val_top1")))
                val_top5.append(float(match.group("val_top5")))
                lrs.append(float(match.group("lr")))

    return {
        "epoch": epochs,
        "train_loss": train_losses,
        "distill_loss": distill_losses,
        "feat_loss": feat_losses,
        "val_loss": val_losses,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "lr": lrs,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distillation ablation curves")
    parser.add_argument(
        "--logs-dir",
        default="experiments/logs",
        help="Cartella principale contenente i log degli esperimenti.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Cartella in cui salvare i grafici.",
    )
    args = parser.parse_args()

    logs_base = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = ["0.3", "0.5", "0.7", "0.9"]
    runs_data = {}

    for alpha in alphas:
        # Il path segue la convenzione: experiments/logs/distillation_alphaXX/distillation_alphaXX/run.log
        folder_name = f"distillation_alpha{alpha.replace('.', '')}"
        log_path = logs_base / folder_name / folder_name / "run.log"
        print(f"Lettura log per alpha={alpha} da {log_path}...")
        data = parse_distill_log(log_path)
        if data and data.get("epoch"):
            runs_data[alpha] = data
        else:
            print(f"[WARN] Dati non trovati o incompleti per alpha={alpha}")

    if not runs_data:
        print("[ERROR] Nessun dato di distillazione caricato con successo. Esco.")
        return

    # Setup matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 4 Subplot: Train Loss, Val Loss, Val Top-1 Accuracy, Feature Loss
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    colors = {"0.3": "#1f77b4", "0.5": "#ff7f0e", "0.7": "#2ca02c", "0.9": "#d62728"}
    markers = {"0.3": "o", "0.5": "s", "0.7": "^", "0.9": "D"}

    for alpha, data in runs_data.items():
        epochs = data["epoch"]
        label = f"alpha = {alpha}"
        c = colors[alpha]
        m = markers[alpha]

        # 1. Train Loss Totale
        axs[0].plot(epochs, data["train_loss"], label=label, color=c, marker=m, markersize=4, linewidth=1.5)
        
        # 2. Val Loss (Cross Entropy)
        axs[1].plot(epochs, data["val_loss"], label=label, color=c, marker=m, markersize=4, linewidth=1.5)
        
        # 3. Val Top-1 Accuracy
        axs[2].plot(epochs, data["val_top1"], label=label, color=c, marker=m, markersize=4, linewidth=1.5)
        
        # 4. Feature Distillation Loss (MSE)
        axs[3].plot(epochs, data["feat_loss"], label=label, color=c, marker=m, markersize=4, linewidth=1.5)

    # Titoli e labels
    axs[0].set_title("Train Loss Totale (Soft KD + Hard CE + Feature MSE)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Epoca")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].set_title("Val Loss (Cross-Entropy)", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Epoca")
    axs[1].set_ylabel("Loss (CE)")
    axs[1].legend()

    axs[2].set_title("Val Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    axs[2].set_xlabel("Epoca")
    axs[2].set_ylabel("Accuracy (%)")
    axs[2].legend()

    axs[3].set_title("Feature Distillation Loss (MSE)", fontsize=12, fontweight="bold")
    axs[3].set_xlabel("Epoca")
    axs[3].set_ylabel("MSE Loss")
    axs[3].legend()

    fig.suptitle("Studio di Ablazione di Alpha (Student Distillato AST)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = out_dir / "distillation_ablation_curves.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Grafico comparativo salvato in: {out_file}")

if __name__ == "__main__":
    main()
