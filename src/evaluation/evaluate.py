"""Script di valutazione generico — Track 24.

Carica un checkpoint salvato durante il training e calcola le metriche
di valutazione sul test set (o su un qualsiasi split specificato):
- Top-1 e Top-5 accuracy
- Matrice di confusione
- Model size in MB
- Latenza di inferenza media in ms

Uso:
    python -m src.evaluation.evaluate \\
        --config experiments/configs/baseline_audio.yaml \\
        --checkpoint experiments/checkpoints/baseline_audio/best.pth \\
        --model-type ast \\
        --split test

    python -m src.evaluation.evaluate \\
        --config experiments/configs/teacher_vision.yaml \\
        --checkpoint experiments/checkpoints/teacher_vision/best.pth \\
        --model-type resnet50 \\
        --split test

Output:
    - Risultati stampati a console e salvati in ``experiments/logs/<run_name>/eval_results.json``.
    - Matrice di confusione salvata in ``figures/confusion_matrix_<run_name>.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.vggsound import VGGSoundDataset
from src.utils.metrics import (
    top_k_accuracy,
    compute_confusion_matrix,
    model_size_mb,
    measure_inference_time_ms,
)

import yaml


def load_config(config_path: str) -> dict:
    """Carica una config YAML con supporto a ``extends``."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "extends" in cfg:
        base_path = Path(config_path).parent / cfg.pop("extends")
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        base_cfg = _deep_merge(base_cfg, cfg)
        return base_cfg
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_model(model_type: str, cfg: dict, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Carica il modello corretto in base a ``model_type`` e applica il checkpoint."""
    if model_type == "ast":
        from src.models.ast_model import build_ast
        # Per la valutazione non serve inizializzare l'architettura con i pesi
        # ImageNet (che richiedono internet sul cluster): il checkpoint caricato
        # subito dopo sovrascrive comunque tutti i parametri.
        eval_cfg = dict(cfg)
        eval_cfg["model"] = dict(cfg.get("model", {}))
        eval_cfg["model"]["pretrained"] = False
        eval_cfg["model"]["weights_path"] = None
        model = build_ast(eval_cfg)
    elif model_type == "resnet50":
        from src.models.vision_teacher import build_vision_teacher
        model = build_vision_teacher(cfg)
    else:
        raise ValueError(f"model_type non riconosciuto: {model_type!r}. Validi: 'ast', 'resnet50'.")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Supporta i diversi formati di checkpoint del progetto:
    # - train_baseline_audio.py / train_teacher.py salvano i pesi sotto "model"
    # - train_distillation.py salva lo student sotto "student"
    # - eventuale state_dict "raw" (nessun wrapping)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model") or ckpt.get("student") or ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Esegue la valutazione completa e restituisce un dict con tutte le metriche."""
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        # Il batch può essere (input, label) per modalità singola o (audio, frame, label) per "both"
        if len(batch) == 3:
            # modalità "both" (per distillation): qui usiamo solo il primo input
            inputs, _, labels = batch
        else:
            inputs, labels = batch

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)

    accs = top_k_accuracy(logits_cat, targets_cat, ks=(1, 5))
    conf_matrix = compute_confusion_matrix(
        logits_cat.argmax(dim=1), targets_cat, num_classes
    )

    return {
        "loss": total_loss / len(loader),
        "top1_acc": accs[1],
        "top5_acc": accs[5],
        "confusion_matrix": conf_matrix.tolist(),
        "num_samples": targets_cat.shape[0],
    }


def plot_confusion_matrix(
    conf_matrix_list: list,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Genera e salva la matrice di confusione come immagine PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib non disponibile: skip plot della confusion matrix.")
        return

    cm = np.array(conf_matrix_list)
    # Normalizza per riga (true label) per la leggibilità
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (row-normalized)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix salvata in: {save_path}")


def resolve_device(device_arg: str | None) -> torch.device:
    """Risolve il device richiesto, con fallback automatico su CPU."""
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_spec(
    config_path: str,
    checkpoint: str,
    model_type: str,
    modality: str | None,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[str]]:
    """Valuta un singolo modello e ne restituisce metriche + nomi delle classi.

    Riusato sia dal percorso single-model sia dalla modalità ``--compare``.
    """
    cfg = load_config(config_path)
    ds_cfg = cfg.get("dataset", {})
    num_classes = int(ds_cfg.get("num_classes", 25))

    if modality is None:
        modality = "video" if model_type == "resnet50" else "audio"
    print(f"  Modello: {model_type} | Modality: {modality} | Split: {split}")

    dataset = VGGSoundDataset(
        split=split,
        modality=modality,
        config=config_path,
        require_files=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    print(f"  Dataset {split}: {len(dataset)} campioni")

    print(f"  Caricamento checkpoint: {checkpoint}")
    model = load_model(model_type, cfg, checkpoint, device)

    size_mb = model_size_mb(model, include_buffers=True)

    # Latenza di inferenza con input sintetico della forma corretta per la modalità.
    if modality in ("audio", "both"):
        example_input = torch.randn(
            1, 1, int(ds_cfg.get("n_mels", 128)), int(ds_cfg.get("target_length", 1024))
        ).to(device)
    else:
        example_input = torch.randn(
            1, 3, int(ds_cfg.get("frame_size", 224)), int(ds_cfg.get("frame_size", 224))
        ).to(device)
    latency_stats = measure_inference_time_ms(model, example_input, n_runs=100, warmup=20, device=device)

    t0 = time.time()
    results = run_evaluation(model, loader, device, num_classes)
    results["model_size_mb"] = size_mb
    results["inference_latency_ms"] = latency_stats["mean_ms"]
    results["inference_latency_stats"] = latency_stats
    results["eval_time_s"] = time.time() - t0
    results["checkpoint"] = checkpoint
    results["split"] = split
    results["model_type"] = model_type
    results["modality"] = modality
    return results, dataset.classes


def format_markdown_table(rows: list[dict], split: str) -> str:
    """Formatta i risultati della comparison come tabella Markdown pronta per il log."""
    header = (
        f"| Exp ID | Modello | Top-1 {split} (%) | Top-5 {split} (%) | "
        "Model Size (MB) | Inf. Latency (ms) |\n"
        "| --- | --- | :---: | :---: | :---: | :---: |"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.get('exp_id', '-')} | {r.get('label', r.get('name', '-'))} | "
            f"{r['top1_acc'] * 100:.2f} | {r['top5_acc'] * 100:.2f} | "
            f"{r['model_size_mb']:.2f} | {r['inference_latency_ms']:.2f} |"
        )
    return "\n".join(lines)


def run_comparison(
    manifest_path: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> None:
    """Valuta tutti i modelli elencati nel manifest e produce la tabella comparativa."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    models = manifest.get("models", [])
    if not models:
        raise ValueError(f"Manifest {manifest_path!r} non contiene alcun modello sotto 'models'.")

    rows: list[dict] = []
    figures_dir = _PROJECT_ROOT / "figures"
    for entry in models:
        name = entry["name"]
        print("\n" + "-" * 60)
        print(f"[{name}] {entry.get('label', '')}")
        results, class_names = evaluate_spec(
            config_path=entry["config"],
            checkpoint=entry["checkpoint"],
            model_type=entry["model_type"],
            modality=entry.get("modality"),
            split=split,
            batch_size=batch_size,
            device=device,
        )
        # Metadati dal manifest (per la tabella e l'analisi ablation).
        for key in ("name", "label", "exp_id", "alpha"):
            if key in entry:
                results[key] = entry[key]
        print(
            f"  -> Top-1: {results['top1_acc'] * 100:.2f}% | "
            f"Top-5: {results['top5_acc'] * 100:.2f}% | "
            f"Size: {results['model_size_mb']:.2f} MB | "
            f"Latency: {results['inference_latency_ms']:.2f} ms"
        )
        # Confusion matrix per modello (utile a comparison.py / side-by-side).
        cm = results.pop("confusion_matrix")
        plot_confusion_matrix(
            cm, class_names, figures_dir / f"confusion_matrix_{name}_{split}.png"
        )
        results["confusion_matrix"] = cm  # reinserita per il salvataggio JSON completo
        rows.append(results)

    # -- Tabella comparativa --
    table = format_markdown_table(rows, split)
    print("\n" + "=" * 60)
    print("  TABELLA COMPARATIVA (Markdown — pronta per EXPERIMENT_LOG)")
    print("=" * 60)
    print(table)
    print("=" * 60)

    # -- Salvataggio JSON aggregato --
    out_dir = _PROJECT_ROOT / "experiments" / "logs" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"comparison_{split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"split": split, "models": rows}, f, indent=2)
    print(f"\nRisultati comparativi salvati in: {json_path}")

    table_path = out_dir / f"comparison_{split}.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"Tabella Markdown salvata in: {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model — Track 24")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Modalità confronto multi-modello: valuta tutti i modelli del manifest.",
    )
    parser.add_argument(
        "--manifest",
        default="experiments/configs/comparison_manifest.yaml",
        help="Manifest YAML dei modelli da confrontare (usato con --compare).",
    )
    parser.add_argument("--config", help="Path al file YAML di configurazione (single-model).")
    parser.add_argument("--checkpoint", help="Path al checkpoint .pth da valutare (single-model).")
    parser.add_argument(
        "--model-type",
        choices=["ast", "resnet50"],
        help="Tipo di modello: 'ast' o 'resnet50' (single-model).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split da valutare (default: test).",
    )
    parser.add_argument(
        "--modality",
        default=None,
        help="Modalità dataset: 'audio', 'video' o 'both'. Se None, inferita dal model-type.",
    )
    parser.add_argument("--device", default=None, help="Device (es. 'cuda:0', 'cpu').")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per la valutazione.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nome della run per i file di output (default: derivato da checkpoint path).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # -- Modalità confronto multi-modello (Fase 4) --
    if args.compare:
        run_comparison(args.manifest, args.split, args.batch_size, device)
        return

    # -- Modalità single-model --
    if not (args.config and args.checkpoint and args.model_type):
        raise SystemExit(
            "Modalità single-model: --config, --checkpoint e --model-type sono obbligatori "
            "(oppure usa --compare per il confronto multi-modello)."
        )

    results, class_names = evaluate_spec(
        config_path=args.config,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        modality=args.modality,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
    )

    # -- Stampa risultati --
    print("\n" + "=" * 60)
    print(f"  Top-1 Accuracy: {results['top1_acc'] * 100:.2f}%")
    print(f"  Top-5 Accuracy: {results['top5_acc'] * 100:.2f}%")
    print(f"  Val Loss:       {results['loss']:.4f}")
    print(f"  Model Size:     {results['model_size_mb']:.2f} MB")
    print(f"  Latency:        {results['inference_latency_ms']:.2f} ms")
    print(f"  Samples:        {results['num_samples']}")
    print("=" * 60)

    # -- Salvataggio risultati --
    run_name = args.run_name or Path(args.checkpoint).parent.name
    out_dir = _PROJECT_ROOT / "experiments" / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rimuove la confusion matrix dal JSON (troppo grande, salvata separatamente come immagine)
    results_to_save = {k: v for k, v in results.items() if k != "confusion_matrix"}
    json_path = out_dir / f"eval_{args.split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Risultati salvati in: {json_path}")

    # -- Plot confusion matrix --
    fig_path = _PROJECT_ROOT / "figures" / f"confusion_matrix_{run_name}_{args.split}.png"
    plot_confusion_matrix(results["confusion_matrix"], class_names, fig_path)


if __name__ == "__main__":
    main()
