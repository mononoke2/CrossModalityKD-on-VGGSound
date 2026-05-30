"""Dataloader PyTorch per il subset VGGSound (Track 24 — Cross-Modal KD).

``VGGSoundDataset`` legge il CSV pre-filtrato ``data/vggsound/subset.csv`` (le 25
classi selezionate) e restituisce, a seconda della modalità:

- ``"audio"``  -> ``(mel_spectrogram, label)``         input per lo student AST
- ``"video"``  -> ``(frame_image, label)``             input per il teacher ResNet-50
- ``"both"``   -> ``(mel_spectrogram, frame_image, label)``  per la distillation

Punti chiave (vedi implementation_plan.md / EXPERIMENT_LOG.md):

- **Split onesto**: il CSV originale definisce solo ``train``/``test``. Da
  ``train`` ricaviamo un validation set con uno split **stratificato per classe**
  (default 85% train / 15% val), deterministico tramite ``seed``. Lo split
  ``test`` resta **blindato** per la valutazione finale.
- **Robustezza al link-rot**: alcune clip possono non essere state scaricate
  (YouTube non più disponibile). Le righe i cui file richiesti non esistono
  vengono scartate (``require_files=True``).
- **Config-driven**: parametri audio/immagine letti da ``common.yaml`` (o da un
  dict passato direttamente).

Layout file atteso (prodotto da ``download_vggsound.py``)::

    data/vggsound/subset.csv                 # header: youtube_id,start_seconds,label,split
    data/vggsound/audio/{id}_{sec}.wav       # mono 16 kHz, 10 s
    data/vggsound/video_frames/{id}_{sec}.jpg
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset


# Statistiche ImageNet per la normalizzazione dei frame (teacher ResNet-50 pretrained).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_VALID_MODALITIES = ("audio", "video", "both")
_VALID_SPLITS = ("train", "val", "test")


def _load_config(config: str | Mapping[str, Any] | None) -> dict:
    """Normalizza il parametro config in un dict.

    Accetta: ``None`` (usa default), un path a un file YAML, oppure un dict già
    caricato. Restituisce la sezione utile (l'intero dict di config).
    """
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    # path a file YAML
    import yaml

    with open(config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class VGGSoundDataset(Dataset):
    """Dataset VGGSound multimodale (audio + frame video) con split onesto.

    Args:
        root: cartella radice del dataset (default da config ``dataset.root``,
            tipicamente ``data/vggsound``). Deve contenere ``subset.csv``,
            ``audio/`` e ``video_frames/``.
        split: ``"train"``, ``"val"`` o ``"test"``.
        modality: ``"audio"``, ``"video"`` o ``"both"``.
        config: path a ``common.yaml`` o dict di config; da qui si leggono
            ``sample_rate``, ``n_mels``, ``target_length``, ``frame_size``,
            ``selected_classes``, ``num_classes`` e ``training.seed``.
        val_ratio: frazione del train originale destinata a validation (per classe).
        seed: seme per lo split stratificato train/val (riproducibilità).
        require_files: se ``True`` scarta le righe i cui file richiesti (audio e/o
            frame, secondo la modalità) non esistono su disco.
        audio_transform: trasformazione opzionale applicata al mel-spectrogram
            ``(1, n_mels, target_length)``; se ``None`` nessuna (oltre alla
            normalizzazione per-campione integrata).
        image_transform: trasformazione torchvision opzionale per il frame; se
            ``None`` viene costruita una pipeline di default (augmentation in
            train, deterministica in val/test).
    """

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        modality: str = "both",
        config: str | Mapping[str, Any] | None = None,
        val_ratio: float = 0.15,
        seed: int | None = None,
        require_files: bool = True,
        audio_transform: Any = None,
        image_transform: Any = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split deve essere uno di {_VALID_SPLITS}, ricevuto {split!r}")
        if modality not in _VALID_MODALITIES:
            raise ValueError(f"modality deve essere una di {_VALID_MODALITIES}, ricevuto {modality!r}")
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio deve essere in (0, 1)")

        cfg = _load_config(config)
        ds_cfg = cfg.get("dataset", {})
        train_cfg = cfg.get("training", {})

        self.root = root or ds_cfg.get("root", "data/vggsound")
        self.split = split
        self.modality = modality
        self.val_ratio = val_ratio
        self.seed = seed if seed is not None else int(train_cfg.get("seed", 42))
        self.require_files = require_files

        # Parametri audio/immagine (config -> default).
        self.sample_rate = int(ds_cfg.get("sample_rate", 16000))
        self.n_mels = int(ds_cfg.get("n_mels", 128))
        self.target_length = int(ds_cfg.get("target_length", 1024))
        self.frame_size = int(ds_cfg.get("frame_size", 224))

        # Mappatura classe -> indice, deterministica e indipendente dallo split.
        selected = ds_cfg.get("selected_classes")
        self.audio_dir = os.path.join(self.root, "audio")
        self.frame_dir = os.path.join(self.root, "video_frames")
        self.csv_path = os.path.join(self.root, "subset.csv")

        df = self._read_subset_csv()

        if selected is None:
            selected = sorted(df["label"].unique().tolist())
        self.classes = sorted(selected)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = int(ds_cfg.get("num_classes", len(self.classes)))

        # Tieni solo le classi selezionate (difensivo: il CSV dovrebbe già esserlo).
        df = df[df["label"].isin(self.class_to_idx)].reset_index(drop=True)

        # Costruisci lo split richiesto.
        self.samples = self._build_split(df)

        # Trasformazioni.
        self.audio_transform = audio_transform
        self.image_transform = image_transform if image_transform is not None else self._default_image_transform()
        self._mel_transform = None  # lazy init (torchaudio) per non rallentare l'import

    # ------------------------------------------------------------------ #
    # Costruzione del dataset
    # ------------------------------------------------------------------ #
    def _read_subset_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"subset.csv non trovato in {self.csv_path}. "
                "Esegui prima src/datasets/download_vggsound.py per generarlo."
            )
        df = pd.read_csv(self.csv_path)
        expected = {"youtube_id", "start_seconds", "label", "split"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Colonne mancanti in {self.csv_path}: {missing}")
        return df

    def _paths_for(self, youtube_id: str, start_seconds: Any) -> tuple[str, str]:
        base = f"{youtube_id}_{int(float(start_seconds))}"
        return (
            os.path.join(self.audio_dir, f"{base}.wav"),
            os.path.join(self.frame_dir, f"{base}.jpg"),
        )

    def _files_present(self, audio_path: str, frame_path: str) -> bool:
        need_audio = self.modality in ("audio", "both")
        need_frame = self.modality in ("video", "both")
        if need_audio and not os.path.exists(audio_path):
            return False
        if need_frame and not os.path.exists(frame_path):
            return False
        return True

    def _build_split(self, df: pd.DataFrame) -> list[dict]:
        """Seleziona le righe dello split richiesto, con split train/val stratificato."""
        if self.split == "test":
            rows = df[df["split"] == "test"]
            wanted = rows
        else:
            # train/val derivano dallo split 'train' originale, stratificato per classe.
            train_rows = df[df["split"] == "train"]
            keep_idx: list[int] = []
            generator = torch.Generator().manual_seed(self.seed)
            for cls in self.classes:
                cls_idx = train_rows.index[train_rows["label"] == cls].tolist()
                if not cls_idx:
                    continue
                # Permutazione deterministica per classe.
                perm = torch.randperm(len(cls_idx), generator=generator).tolist()
                n_val = int(round(len(cls_idx) * self.val_ratio))
                val_pos = set(perm[:n_val])
                for pos, original_index in enumerate(cls_idx):
                    in_val = pos in val_pos
                    if (self.split == "val") == in_val:
                        keep_idx.append(original_index)
            wanted = train_rows.loc[keep_idx]

        samples: list[dict] = []
        skipped = 0
        for _, row in wanted.iterrows():
            audio_path, frame_path = self._paths_for(row["youtube_id"], row["start_seconds"])
            if self.require_files and not self._files_present(audio_path, frame_path):
                skipped += 1
                continue
            samples.append(
                {
                    "audio_path": audio_path,
                    "frame_path": frame_path,
                    "label": self.class_to_idx[row["label"]],
                }
            )

        if self.require_files and skipped:
            # Informativo: utile per accorgersi di download incompleti.
            print(
                f"[VGGSoundDataset] split={self.split} modality={self.modality}: "
                f"{len(samples)} campioni, {skipped} scartati (file mancanti)."
            )
        return samples

    # ------------------------------------------------------------------ #
    # Trasformazioni
    # ------------------------------------------------------------------ #
    def _default_image_transform(self):
        from torchvision import transforms

        if self.split == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.frame_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
                ]
            )
        # val / test: deterministico.
        resize = int(self.frame_size * 1.14)  # ~256 per frame_size 224
        return transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(self.frame_size),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

    def _get_mel_transform(self):
        if self._mel_transform is None:
            import torchaudio

            # n_fft=1024 (> win_length, finestra zero-paddata): dà 513 bin di
            # frequenza, sufficienti per 128 bande mel senza filtri vuoti. La
            # risoluzione temporale dipende solo da hop_length (=160 -> ~10ms).
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                win_length=400,
                hop_length=160,
                n_mels=self.n_mels,
            )
        return self._mel_transform

    # ------------------------------------------------------------------ #
    # Caricamento campioni
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_wav(path: str) -> tuple[torch.Tensor, int]:
        """Legge un WAV come tensore float32 ``(channels, samples)`` in [-1, 1].

        Non dipende dal backend I/O di torchaudio (assente in alcuni container):
        usa ``soundfile`` se disponibile, altrimenti ``scipy.io.wavfile`` (i WAV
        prodotti da download_vggsound.py sono PCM 16-bit, gestiti da entrambi).
        """
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
            wav = torch.from_numpy(data.T.copy())  # (channels, samples)
            return wav, int(sr)
        except ImportError:
            import numpy as np
            from scipy.io import wavfile

            sr, data = wavfile.read(path)
            data = np.asarray(data)
            # Normalizza i tipi interi PCM in float [-1, 1].
            if np.issubdtype(data.dtype, np.integer):
                max_val = float(np.iinfo(data.dtype).max)
                data = data.astype("float32") / max_val
            else:
                data = data.astype("float32")
            if data.ndim == 1:
                data = data[:, None]  # (samples, 1)
            wav = torch.from_numpy(data.T.copy())  # (channels, samples)
            return wav, int(sr)

    def _load_audio(self, path: str) -> torch.Tensor:
        """Carica un WAV e restituisce un log-mel-spectrogram ``(1, n_mels, target_length)``."""
        import torchaudio

        waveform, sr = self._read_wav(path)  # (channels, samples)
        if waveform.size(0) > 1:  # mix down a mono (sicurezza: dovrebbe già esserlo)
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        mel = self._get_mel_transform()(waveform)  # (1, n_mels, time)
        log_mel = torch.log(mel + 1e-6)

        # Pad/troncamento sull'asse temporale a target_length.
        time = log_mel.size(-1)
        if time < self.target_length:
            log_mel = torch.nn.functional.pad(log_mel, (0, self.target_length - time))
        elif time > self.target_length:
            log_mel = log_mel[..., : self.target_length]

        # Normalizzazione per-campione (zero mean / unit std): stabile quando le
        # statistiche globali del dataset non sono note a priori.
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        if self.audio_transform is not None:
            log_mel = self.audio_transform(log_mel)
        return log_mel

    def _load_image(self, path: str) -> torch.Tensor:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.image_transform(img)

    # ------------------------------------------------------------------ #
    # API Dataset
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        label = sample["label"]

        if self.modality == "audio":
            return self._load_audio(sample["audio_path"]), label
        if self.modality == "video":
            return self._load_image(sample["frame_path"]), label
        # both
        return (
            self._load_audio(sample["audio_path"]),
            self._load_image(sample["frame_path"]),
            label,
        )

    def __repr__(self) -> str:
        return (
            f"VGGSoundDataset(split={self.split!r}, modality={self.modality!r}, "
            f"num_classes={self.num_classes}, n_samples={len(self)})"
        )


if __name__ == "__main__":
    # Self-check leggero: stampa le dimensioni degli split e la forma di un campione.
    # Richiede dati scaricati; con require_files=True gli split possono essere vuoti
    # se le clip non sono ancora presenti.
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test VGGSoundDataset")
    parser.add_argument("--root", default="data/vggsound")
    parser.add_argument("--config", default="experiments/configs/common.yaml")
    parser.add_argument("--modality", default="both", choices=_VALID_MODALITIES)
    args = parser.parse_args()

    for sp in _VALID_SPLITS:
        ds = VGGSoundDataset(root=args.root, split=sp, modality=args.modality, config=args.config)
        print(ds)
        if len(ds) > 0:
            item = ds[0]
            shapes = [tuple(x.shape) if hasattr(x, "shape") else x for x in item]
            print(f"  esempio[0]: {shapes}")
