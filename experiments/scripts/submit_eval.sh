#!/bin/bash
#SBATCH --account=dl-course-q2               # account assegnato al corso
#SBATCH --partition=dl-course-q2             # partizione assegnata al corso
#SBATCH --qos=gpu-medium                      # QoS: gpu-medium (6h, 8G RAM, 5632 VRAM)
#SBATCH --mem=8G                            # RAM massima
#SBATCH --cpus-per-task=2                    # CPU cores
#SBATCH --gres=gpu:1 --gres=shard:5632      # GPU e VRAM in MB
#SBATCH --time=00:30:00                      # La valutazione è breve: 30 min bastano
#SBATCH --output=experiments/logs/eval-%j.log

# Submit di una valutazione (src.evaluation.evaluate) sul cluster via Apptainer.
# Complementare a submit_job.sh, che lancia invece i moduli src.training.* del training.
#
# Uso:
#   sbatch experiments/scripts/submit_eval.sh \
#       --config experiments/configs/teacher_vision.yaml \
#       --checkpoint experiments/checkpoints/teacher_vision/best.pth \
#       --model-type resnet50 --split test

echo "Data inizio job: $(date)"
echo "Host: $(hostname)"
echo "ID Job SLURM: $SLURM_JOB_ID"
echo "Esecuzione in corso tramite Apptainer..."

# Tutti gli argomenti passati allo script vengono inoltrati a evaluate.py.
apptainer run --nv /shared/sifs/latest.sif python3 -m src.evaluation.evaluate "$@"
EXIT_CODE=$?

echo "Job completato con codice d'uscita: $EXIT_CODE"
echo "Data fine job: $(date)"
exit $EXIT_CODE
