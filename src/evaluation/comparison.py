"""Grafici comparativi — Track 24: Cross-Modal Knowledge Distillation.

Legge il file ``experiments/logs/comparison/comparison_test.json`` prodotto da
``src.evaluation.evaluate --compare`` e genera i seguenti grafici in ``figures/``:

1. **accuracy_comparison.png** — Bar plot Top-1 e Top-5 Test Accuracy per tutti i modelli.
2. **efficiency_comparison.png** — Bar plot Model Size (MB) e Inference Latency (ms).
3. **ablation_alpha_test.png** — Curva Top-1 Test Accuracy vs α ∈ {0.3, 0.5, 0.7, 0.9}.
4. **confusion_matrix_comparison.png** — Confusion matrix side-by-side Baseline vs Best Student.

Uso:
    python -m src.evaluation.comparison
    python -m src.evaluation.comparison --json experiments/logs/comparison/comparison_test.json \\
                                         --output figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Palette e stile
# ---------------------------------------------------------------------------
PALETTE = {
    "teacher":    "#6c757d",   # grigio
    "baseline":   "#4361ee",   # blu
    "alpha03":    "#f72585",   # rosa
    "alpha05":    "#7209b7",   # viola
    "alpha07":    "#3a0ca3",   # viola scuro
    "alpha09":    "#b5179e",   # magenta
}

_EXP_COLOR = {
    "EXP-002": PALETTE["teacher"],
    "EXP-001": PALETTE["baseline"],
    "EXP-003": PALETTE["alpha03"],
    "EXP-004": PALETTE["alpha05"],
    "EXP-005": PALETTE["alpha07"],
    "EXP-006": PALETTE["alpha09"],
}

_ALPHA_COLOR = {
    0.3: PALETTE["alpha03"],
    0.5: PALETTE["alpha05"],
    0.7: PALETTE["alpha07"],
    0.9: PALETTE["alpha09"],
}

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#f0f6fc",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pct(v: float) -> float:
    """Converte da [0,1] a percentuale."""
    return round(v * 100, 2)


def _load(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["models"]


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy Bar Chart (Top-1 e Top-5)
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(models: list[dict], output_dir: Path) -> Path:
    labels  = [m["label"] for m in models]
    top1    = [_pct(m["top1_acc"]) for m in models]
    top5    = [_pct(m["top5_acc"]) for m in models]
    colors  = [_EXP_COLOR.get(m["exp_id"], "#adb5bd") for m in models]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0d1117")

    bars1 = ax.bar(x - width / 2, top1, width, color=colors, alpha=0.92, label="Top-1", zorder=3)
    bars5 = ax.bar(x + width / 2, top5, width, color=colors, alpha=0.50, label="Top-5",
                   edgecolor=colors, linewidth=1.2, zorder=3)

    # Etichette sulle barre
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9,
                color="#f0f6fc", fontweight="bold")
    for bar in bars5:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5,
                color="#8b949e")

    # Linea baseline
    baseline_top1 = next(m["top1_acc"] for m in models if m["exp_id"] == "EXP-001")
    ax.axhline(_pct(baseline_top1), linestyle="--", linewidth=1.2, color=PALETTE["baseline"],
               alpha=0.6, label=f"Baseline Top-1 ({_pct(baseline_top1):.2f}%)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test Set Accuracy — Confronto Multi-Modello (Top-1 / Top-5)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    legend_patch = [
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.92, label="Top-1 (pieno)"),
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.45, label="Top-5 (trasparente)"),
    ]
    ax.legend(handles=legend_patch + [plt.Line2D([0], [0], linestyle="--",
              color=PALETTE["baseline"], alpha=0.7,
              label=f"Baseline Top-1 ({_pct(baseline_top1):.2f}%)")],
              loc="lower right", framealpha=0.3, fontsize=9)

    fig.tight_layout()
    out = output_dir / "accuracy_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvato: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 2 — Efficiency (Model Size + Latency)
# ---------------------------------------------------------------------------

def plot_efficiency_comparison(models: list[dict], output_dir: Path) -> Path:
    labels   = [m["label"] for m in models]
    sizes    = [m["model_size_mb"] for m in models]
    latency  = [m["inference_latency_ms"] for m in models]
    colors   = [_EXP_COLOR.get(m["exp_id"], "#adb5bd") for m in models]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Efficienza dei Modelli — Model Size e Inference Latency", fontsize=14,
                 fontweight="bold", color="#f0f6fc")

    # --- Model Size ---
    bars = ax1.bar(x, sizes, color=colors, alpha=0.88, zorder=3)
    for bar, val in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val:.1f} MB", ha="center", va="bottom", fontsize=9, color="#f0f6fc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Model Size (MB)")
    ax1.set_title("Dimensione del Modello")
    ax1.set_ylim(0, max(sizes) * 1.2)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    # --- Latency ---
    bars2 = ax2.bar(x, latency, color=colors, alpha=0.88, zorder=3)
    for bar, val in zip(bars2, latency):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f} ms", ha="center", va="bottom", fontsize=9, color="#f0f6fc")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Latenza Inferenza (ms)")
    ax2.set_title("Latenza di Inferenza (avg/100 run)")
    ax2.set_ylim(0, max(latency) * 1.25)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    out = output_dir / "efficiency_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvato: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 3 — Ablation α: Top-1 Test Accuracy vs α
# ---------------------------------------------------------------------------

def plot_ablation_alpha(models: list[dict], output_dir: Path) -> Path:
    distill = sorted(
        [m for m in models if "alpha" in m],
        key=lambda m: m["alpha"]
    )
    alphas  = [m["alpha"] for m in distill]
    top1    = [_pct(m["top1_acc"]) for m in distill]
    colors  = [_ALPHA_COLOR.get(a, "#adb5bd") for a in alphas]

    baseline_top1 = next((_pct(m["top1_acc"]) for m in models if m["exp_id"] == "EXP-001"), None)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")

    # Curva con punti colorati
    ax.plot(alphas, top1, color="#c9d1d9", linewidth=2, zorder=2, marker="")
    for a, t, c in zip(alphas, top1, colors):
        ax.scatter(a, t, color=c, s=120, zorder=4, edgecolors="#f0f6fc", linewidths=1.5)
        ax.text(a, t + 0.25, f"{t:.2f}%", ha="center", va="bottom",
                fontsize=10, color=c, fontweight="bold")

    # Riempi area sotto la curva
    ax.fill_between(alphas, top1, min(top1) - 1, alpha=0.12, color="#7209b7")

    # Linea baseline
    if baseline_top1 is not None:
        ax.axhline(baseline_top1, linestyle="--", linewidth=1.5, color=PALETTE["baseline"],
                   alpha=0.7, label=f"Baseline Audio ({baseline_top1:.2f}%)", zorder=3)

    # Annotazione del best alpha
    best_idx = int(np.argmax(top1))
    ax.annotate(f"Best α={alphas[best_idx]}\n{top1[best_idx]:.2f}%",
                xy=(alphas[best_idx], top1[best_idx]),
                xytext=(alphas[best_idx] + 0.04, top1[best_idx] - 0.8),
                fontsize=9, color="#f0f6fc",
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.2))

    ax.set_xlabel("Peso di Distillazione α")
    ax.set_ylabel("Top-1 Test Accuracy (%)")
    ax.set_title("Ablation Study — Top-1 Test Accuracy vs α")
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ylim_min = min(min(top1), baseline_top1 if baseline_top1 else min(top1)) - 2
    ylim_max = max(max(top1), baseline_top1 if baseline_top1 else max(top1)) + 2
    ax.set_ylim(ylim_min, ylim_max)
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0.3)

    fig.tight_layout()
    out = output_dir / "ablation_alpha_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvato: {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 4 — Confusion Matrix Side-by-Side (Baseline vs Best Student)
# ---------------------------------------------------------------------------

def _load_confusion_matrix(models: list[dict], exp_id: str) -> np.ndarray | None:
    """Cerca la confusion matrix dal JSON (campo 'confusion_matrix')."""
    for m in models:
        if m.get("exp_id") == exp_id and "confusion_matrix" in m:
            return np.array(m["confusion_matrix"])
    return None


def plot_confusion_matrix_comparison(
    models_raw: list[dict],
    output_dir: Path,
    class_names: list[str] | None = None,
) -> Path | None:
    cm_baseline = _load_confusion_matrix(models_raw, "EXP-001")
    cm_best     = _load_confusion_matrix(models_raw, "EXP-003")   # α=0.3 è il best su test

    if cm_baseline is None or cm_best is None:
        print("[WARN] Confusion matrix non disponibile nel JSON — salto il plot side-by-side.")
        return None

    n = cm_baseline.shape[0]
    if class_names is None or len(class_names) != n:
        class_names = [str(i) for i in range(n)]

    # Normalizza per riga
    def _norm(cm: np.ndarray) -> np.ndarray:
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, cm / row_sums, 0.0)

    cm_b_norm = _norm(cm_baseline)
    cm_s_norm = _norm(cm_best)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Confusion Matrix — Baseline Audio vs Student KD (α=0.3, Best)",
                 fontsize=14, fontweight="bold", color="#f0f6fc", y=1.01)

    tick_labels = [c[:18] for c in class_names]   # Tronca etichette lunghe

    for ax, cm_norm, title in [
        (axes[0], cm_b_norm, "AST Audio Baseline (EXP-001)"),
        (axes[1], cm_s_norm, "Student KD α=0.3 (EXP-003)"),
    ]:
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="#f0f6fc", fontsize=12)
        ax.set_xlabel("Classe Predetta", color="#c9d1d9")
        ax.set_ylabel("Classe Reale", color="#c9d1d9")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)

        # Annotazioni numeriche solo se il numero di classi è gestibile
        if n <= 25:
            thresh = 0.5
            for i in range(n):
                for j in range(n):
                    color = "white" if cm_norm[i, j] > thresh else "#adb5bd"
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                            ha="center", va="center", fontsize=5.5, color=color)

    fig.tight_layout()
    out = output_dir / "confusion_matrix_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvato: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera grafici comparativi — Track 24")
    p.add_argument(
        "--json",
        default="experiments/logs/comparison/comparison_test.json",
        help="Path al JSON prodotto da evaluate.py --compare (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default="figures",
        help="Directory di output per i grafici (default: %(default)s)",
    )
    p.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Lista dei nomi delle classi (opzionale, per le confusion matrix)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_path  = Path(args.json)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[ERRORE] File JSON non trovato: {json_path}")
        sys.exit(1)

    print(f"Lettura risultati da: {json_path}")
    models = _load(str(json_path))
    print(f"Modelli caricati: {len(models)}")
    for m in models:
        print(f"  {m['exp_id']:8s} | {m['label']:30s} | Top-1={_pct(m['top1_acc']):.2f}%")

    print("\nGenerazione grafici...")
    plot_accuracy_comparison(models, output_dir)
    plot_efficiency_comparison(models, output_dir)
    plot_ablation_alpha(models, output_dir)

    # Per la confusion matrix, usa i nomi forniti o quelli eventualmente nel JSON
    class_names = args.class_names
    plot_confusion_matrix_comparison(models, output_dir, class_names=class_names)

    print(f"\n✓ Tutti i grafici salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
