#!/bin/bash
#SBATCH --account=dl-course-q2               # account assegnato al corso (es. dl-course-q2)
#SBATCH --partition=dl-course-q2             # partizione assegnata al corso (coincide con account)
#SBATCH --qos=gpu-large                      # QoS: gpu-medium (6h, 8G RAM, 5632 VRAM) o gpu-large (12h, 16G RAM, 11264 VRAM)
#SBATCH --mem=16G                            # RAM massima (Max per gpu-large: 16G, gpu-medium: 8G, gpu-xlarge: 48G)
#SBATCH --cpus-per-task=4                    # CPU cores (Max per gpu-large: 4, gpu-medium: 2, gpu-xlarge: 8)
#SBATCH --gres=gpu:1 --gres=shard:11264      # GPU e VRAM in MB (Max per gpu-large: 11264, gpu-medium: 5632, gpu-xlarge: 22528)
#SBATCH --time=12:00:00                      # Limite temporale (Max per gpu-large/xlarge: 12h, gpu-medium: 6h)
#SBATCH --output=experiments/logs/job-%j.log
#SBATCH --signal=USR1@90                     # Segnale SIGUSR1 90s prima del timeout per checkpointing

# Gestore del segnale SIGUSR1 inviato da SLURM prima della fine del tempo limite
cleanup_handler() {
    echo "Ricevuto segnale SIGUSR1 da SLURM. Esecuzione checkpointing d'emergenza..."
    # Qui il codice Python intercetterà a sua volta il segnale o salverà automaticamente.
    # Se desiderato, si può risottomettere il job automaticamente:
    # sbatch "$0" "$@"
    exit 99
}
trap 'cleanup_handler' USR1

# Verifica argomenti
if [ -z "$1" ]; then
    echo "Errore: specificare lo script Python da eseguire."
    echo "Uso: sbatch experiments/scripts/submit_job.sh <script_name.py> [args...]"
    exit 1
fi

SCRIPT_NAME=$1
shift # Rimuove il primo parametro (nome script), lasciando gli altri argomenti in $@

# Stampa info utili sul job
echo "Data inizio job: $(date)"
echo "Host: $(hostname)"
echo "ID Job SLURM: $SLURM_JOB_ID"
echo "Esecuzione in corso tramite Apptainer..."

# Esecuzione del progetto dentro il container Apptainer fornito dal cluster DMI
# L'opzione --nv abilita l'integrazione con i driver NVIDIA CUDA
apptainer run --nv /shared/sifs/latest.sif python3 -m "$SCRIPT_NAME" "$@" &
PID=$!

# Attesa del processo in background per poter intercettare i segnali inviati a questo script bash
wait $PID
EXIT_CODE=$?

echo "Job completato con codice d'uscita: $EXIT_CODE"
echo "Data fine job: $(date)"
exit $EXIT_CODE
