# Cross-Modal Knowledge Distillation (Audio → Vision) on VGGSound

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Group and Project Information
- **Group ID**: Zero e Uno
- **Project ID**: Track 24
- **Members**: Denise Cilia, Eleonora Giuffrida

## 📝 Project Description
This repository implements **Cross-Modal Knowledge Distillation (KD)** from a frozen visual teacher (**ResNet-50**) to an audio spectrogram transformer (**AST** student) on a 25-class subset of **VGGSound**. The goal is *modality hallucination*: training an audio classifier that implicitly benefits from visual context during training, without requiring any visual frames during inference.

> 📖 **Official Report**: For full theoretical details, link-rot data analysis, ablation details, and qualitative/quantitative discussions, please refer to our formal paper: **[REPORT.md](docs/REPORT.md)**.

---

## 🛠 Technical Reproducibility

### 1. Data and Environment Setup

#### Environment installation
Create the conda environment containing PyTorch, torchaudio, timm, and other audio processing packages:

```bash
conda env create -f environment.yml
conda activate dl-project
```

Alternatively, you can install the required packages using pip:
```bash
pip install -r requirements.txt
```

Or if you prefer not to use conda, you can create the python3 virtual environment using the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Dataset preparation
The dataset consists of a 25-class subset extracted from VGGSound. To download the files and preprocess them (downsampling audio to 16 kHz and extracting central video frames):

1. You can download the csv file from [here](https://github.com/hche11/VGGSound/tree/master/data). Then, place `vggsound.csv` in `data/`.
2. Run the offline downloader script (handles YouTube video scraping):
   ```bash
   python -m src.datasets.download_vggsound
   ```

This will populate the directory `data/vggsound/` with raw `.wav` audio tracks and `.jpg` frames.

#### Jupyter Notebooks (Data Exploration & Inspection)

We provide three interactive Jupyter Notebooks in the `notebooks/` directory to document our data analysis, subset selection criteria, and audio-visual preprocessing:
* **[01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb)**: Analyzes the raw full VGGSound dataset (199k clips, 310 classes), outlines the selection process of the 25 target classes, and details the semantic grouping logic.
* **[02_downloaded_subset_exploration.ipynb](notebooks/02_downloaded_subset_exploration.ipynb)**: Evaluates the filtered `subset.csv` (26,250 clips) and investigates the impact of YouTube link-rot. It documents download success rates, identifies outliers (like *chicken crowing*), and projects the stratified 85/15 train/val split.
* **[03_audio_visual_inspection.ipynb](notebooks/03_audio_visual_inspection.ipynb)**: Visualizes Mel-spectrograms (128 bins) and audio waveforms using `librosa`, checks the 1-to-1 audio-visual pairing on disk, and estimates pixel-level statistics of the $224 \times 224$ central video frames to validate ImageNet normalization.

You can launch and view these notebooks locally by activating your environment and running:

```bash
jupyter notebook
```

---

### 2. Network Training

Training script automatically detect available GPUs and support SLURM checkpoint pre-emption/resume (if you run them on the DMI cluster).

#### Baseline Audio (AST) Training:
```bash
python -m src.training.train_baseline_audio --config experiments/configs/baseline_audio.yaml
```

#### Teacher Vision (ResNet-50) Training:
```bash
python -m src.training.train_teacher --config experiments/configs/teacher_vision.yaml
```

#### Distilled Student AST Training:
```bash
# Default distillation (ablation values overridable via CLI --alpha)
python -m src.training.train_distillation --config experiments/configs/distillation.yaml --alpha 0.3
```

#### Lightweight CNN Student Benchmark (Extra Objective 3):

```bash
# 1. Train EfficientNet-B0 & MobileNetV3-Small Baselines
python -m src.training.train_baseline_audio --config experiments/configs/baseline_audio_efficientnet_b0.yaml

python -m src.training.train_baseline_audio --config experiments/configs/baseline_audio_mobilenet_v3_small.yaml

# 2. Train Distilled Students (KD with alpha=0.3)
python -m src.training.train_distillation --config experiments/configs/distillation_efficientnet_b0.yaml --student-type efficientnet_b0_audio

python -m src.training.train_distillation --config experiments/configs/distillation_mobilenet_v3_small.yaml --student-type mobilenet_v3_small_audio
```

---

#### 3. Evaluation and Plots

To run the complete multi-model comparative evaluation on the blind test set (933 samples) across all 10 experiments:

```bash
python -m src.evaluation.evaluate --compare --manifest experiments/configs/comparison_manifest.yaml --split test
```

To regenerate the comparative figures (accuracy comparison, efficiency, ablation curve, and the accuracy/latency/size trade-off plot):

```bash
python -m src.evaluation.comparison \
    --json experiments/logs/comparison/comparison_test.json \
    --output figures/
```

Figures will be saved in the `figures/` directory, including:
* `accuracy_comparison.png` - Top-1/Top-5 accuracy.
* `efficiency_comparison.png` - Latency vs size.
* `tradeoff_comparison.png` - Scatter plot trade-off.
* `ablation_alpha_test.png` - Ablation study.
* `confusion_matrix_comparison.png` - Confusion matrix side-by-side.

To regenerate the training curves (Loss and Accuracy) directly from the text log files:

```bash
# Plot training curves for a single run (e.g., AST baseline)
python -m src.utils.plot_logs --log-file experiments/logs/baseline_audio/run.log

# Plot comparative training curves for distillation ablation across alphas
python -m src.utils.plot_distillation --logs-dir experiments/logs
```

Note that you can also use TensorBoard to visualize the training process:

```bash
tensorboard --logdir experiments/logs
```

If you have an account, you can use Weights & Biases:

```bash
wandb login
wandb sync experiments/logs
```

To view all local runs:
```bash
wandb ui experiments/logs
```

---

*For detailed task distributions and the declaration of AI tools, please refer to [docs/REPORT.md](docs/REPORT.md).*
