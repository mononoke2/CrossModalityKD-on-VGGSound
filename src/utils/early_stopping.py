"""Early stopping per i training di Track 24.

Monitora una metrica di validazione e segnala quando interrompere il training
se non migliora per un numero di epoche pari a ``patience``. Supporta sia
metriche da minimizzare (es. loss) che da massimizzare (es. accuracy).

Esempio d'uso::

    stopper = EarlyStopping(patience=5, mode="max", min_delta=1e-4)
    for epoch in range(epochs):
        val_acc = validate(...)
        improved = stopper.step(val_acc)
        if improved:
            save_checkpoint(...)        # nuovo best
        if stopper.should_stop:
            break
"""

from __future__ import annotations

import math


class EarlyStopping:
    """Tiene traccia della metrica migliore e decide quando fermarsi.

    Args:
        patience: numero di epoche senza miglioramento tollerate prima dello stop.
        mode: ``"min"`` se la metrica va minimizzata (loss), ``"max"`` se va
            massimizzata (accuracy).
        min_delta: miglioramento minimo perché conti come tale (evita di
            resettare il contatore per fluttuazioni numeriche trascurabili).
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode deve essere 'min' o 'max', ricevuto {mode!r}")
        if patience < 0:
            raise ValueError("patience deve essere >= 0")
        if min_delta < 0:
            raise ValueError("min_delta deve essere >= 0")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: float = -math.inf if mode == "max" else math.inf
        self.best_epoch: int = -1
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False
        self._epoch: int = -1

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        """Aggiorna lo stato con la metrica dell'epoca corrente.

        Args:
            score: valore della metrica monitorata per l'epoca appena conclusa.

        Returns:
            ``True`` se questa epoca è un nuovo best (utile per decidere se
            salvare il checkpoint), ``False`` altrimenti.
        """
        self._epoch += 1

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = self._epoch
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            self.should_stop = True
        return False

    def state_dict(self) -> dict:
        """Stato serializzabile, da salvare nel checkpoint per il resume SLURM."""
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
            "_epoch": self._epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Ripristina lo stato da un checkpoint (per riprendere il training)."""
        self.patience = state["patience"]
        self.mode = state["mode"]
        self.min_delta = state["min_delta"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.num_bad_epochs = state["num_bad_epochs"]
        self.should_stop = state["should_stop"]
        self._epoch = state["_epoch"]

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(mode={self.mode!r}, patience={self.patience}, "
            f"best_score={self.best_score:.4f}@epoch{self.best_epoch}, "
            f"bad_epochs={self.num_bad_epochs}, should_stop={self.should_stop})"
        )
