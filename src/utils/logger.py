"""Logging unificato per Track 24.

Fornisce un singolo oggetto ``ExperimentLogger`` che combina:

- logging su **console** e su **file** (via il modulo standard ``logging``);
- logging di scalari/metriche su **TensorBoard** (se ``tensorboard`` è installato);
- logging opzionale su **Weights & Biases** (se ``wandb`` è installato e abilitato).

Le dipendenze di tracking sono opzionali: se non presenti, il logger continua a
funzionare scrivendo solo su console/file (con un warning una tantum). Questo
evita di bloccare i job sul cluster per una dipendenza mancante.

Esempio::

    logger = ExperimentLogger("baseline_audio", log_dir="experiments/logs", use_tensorboard=True)
    logger.info("Inizio training")
    logger.log_scalars({"train/loss": 0.7, "val/acc_top1": 0.42}, step=epoch)
    logger.close()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping


class ExperimentLogger:
    """Logger di esperimento con backend multipli (console/file/TensorBoard/W&B).

    Args:
        name: nome dell'esperimento/run (usato per i file di log e la run W&B).
        log_dir: directory radice dove scrivere log e summary TensorBoard.
            Verrà creata una sottocartella ``log_dir/name``.
        use_tensorboard: se attivare il logging su TensorBoard.
        use_wandb: se attivare il logging su Weights & Biases.
        wandb_project: nome del progetto W&B (richiesto se ``use_wandb``).
        config: dizionario di iperparametri da registrare (loggato a console e,
            se attivo, come config della run W&B).
        level: livello di logging per console/file.
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "experiments/logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        config: Mapping[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> None:
        self.name = name
        self.run_dir = os.path.join(log_dir, name)
        os.makedirs(self.run_dir, exist_ok=True)

        # --- Logger standard (console + file) ---
        self._logger = logging.getLogger(f"track24.{name}")
        self._logger.setLevel(level)
        self._logger.propagate = False
        if not self._logger.handlers:  # evita handler duplicati su re-init
            fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            self._logger.addHandler(console)

            file_handler = logging.FileHandler(os.path.join(self.run_dir, "run.log"))
            file_handler.setFormatter(fmt)
            self._logger.addHandler(file_handler)

        # --- TensorBoard (opzionale) ---
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=self.run_dir)
            except Exception as exc:  # tensorboard non installato o errore di init
                self._logger.warning("TensorBoard non disponibile (%s): logging solo su file.", exc)

        # --- Weights & Biases (opzionale) ---
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project=wandb_project or "track24-kd",
                    name=name,
                    dir=self.run_dir,
                    config=dict(config) if config else None,
                )
            except Exception as exc:
                self._logger.warning("W&B non disponibile (%s): logging disabilitato per W&B.", exc)
                self._wandb = None

        if config:
            self.info("Configurazione esperimento: %s", dict(config))

    # ------------------------------------------------------------------ #
    # Logging testuale (delega al logger standard)
    # ------------------------------------------------------------------ #
    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    # ------------------------------------------------------------------ #
    # Logging di metriche
    # ------------------------------------------------------------------ #
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logga un singolo scalare su tutti i backend attivi."""
        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: Mapping[str, float], step: int) -> None:
        """Logga più scalari in un colpo (es. tutte le metriche di un'epoca)."""
        if self._tb is not None:
            for tag, value in metrics.items():
                self._tb.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)

    # ------------------------------------------------------------------ #
    # Chiusura
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Chiude i writer e rilascia gli handler (da chiamare a fine training)."""
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
