"""Script per generare le curve di training (Loss e Accuracy) partendo dal file di log.

Legge il file di log testuale generato da ExperimentLogger e produce un grafico
PNG con due subplot (Loss e Accuracy Top-1/Top-5) per l'inclusione nel report.

Uso:
    python3 src/utils/plot_logs.py \
        --log-file experiments/logs/baseline_audio/run.log \
        --output-dir figures/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Configura matplotlib per l'uso senza interfaccia grafica (modalità headless/cluster)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib non è installato. Esegui `pip install matplotlib` per generare i grafici.")


def parse_log_file(log_path: Path) -> dict[str, list[float]]:
    """Effettua il parsing del file run.log estraendo le metriche per epoca."""
    epochs: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_top1: list[float] = []
    val_top5: list[float] = []
    lrs: list[float] = []

    # Regex per catturare la riga di riassunto epoca:
    # 2026-06-03 18:41:29 | INFO    | Epoch 1/30 — train_loss=1.2859 | val_loss=1.0842 | val_top1=80.45% | val_top5=99.19% | LR=3.33e-05 | 179s
    epoch_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/\d+\s+—\s+"
        r"train_loss=(?P<train_loss>[\d\.]+)\s*\|\s*"
        r"val_loss=(?P<val_loss>[\d\.]+)\s*\|\s*"
        r"val_top1=(?P<val_top1>[\d\.]+)%\s*\|\s*"
        r"val_top5=(?P<val_top5>[\d\.]+)%\s*\|\s*"
        r"LR=(?P<lr>[\d\.\-e]+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Inizio training:" in line or "Configurazione esperimento:" in line:
                # Reset per considerare solo l'ultimo run ed evitare la sovrapposizione in file in append
                epochs.clear()
                train_losses.clear()
                val_losses.clear()
                val_top1.clear()
                val_top5.clear()
                lrs.clear()

            match = epoch_pattern.search(line)
            if match:
                epochs.append(float(match.group("epoch")))
                train_losses.append(float(match.group("train_loss")))
                val_losses.append(float(match.group("val_loss")))
                val_top1.append(float(match.group("val_top1")))
                val_top5.append(float(match.group("val_top5")))
                lrs.append(float(match.group("lr")))

    return {
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "lr": lrs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from run.log")
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path al file run.log dell'esperimento.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Cartella in cui salvare i grafici generati (default: figures).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"File log non trovato: {log_path}")

    metrics = parse_log_file(log_path)
    if not metrics["epoch"]:
        print(f"[WARN] Nessuna metrica trovata nel file di log {log_path}. Verifica il formato.")
        return

    # Crea la cartella di output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = log_path.parent.name

    # Impostazioni estetiche per grafici professionali
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Plot della Loss
    ax1.plot(metrics["epoch"], metrics["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
    ax1.plot(metrics["epoch"], metrics["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, marker="s")
    ax1.set_xlabel("Epoca", fontsize=11)
    ax1.set_ylabel("Loss (Cross-Entropy)", fontsize=11)
    ax1.set_title("Curve di Loss", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Plot dell'Accuracy
    ax2.plot(metrics["epoch"], metrics["val_top1"], label="Val Top-1 Acc", color="#2ca02c", linewidth=2, marker="^")
    ax2.plot(metrics["epoch"], metrics["val_top5"], label="Val Top-5 Acc", color="#d62728", linewidth=2, marker="v")
    ax2.set_xlabel("Epoca", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Accuratezza di Validazione", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"Esperimento: {run_name}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = out_dir / f"training_curves_{run_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f">> Grafico salvato con successo in: {save_path}")


if __name__ == "__main__":
    main()
