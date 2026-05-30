#!/usr/bin/env bash
#
# sync_to_cluster.sh — Deploy del codice locale verso il cluster DMI via rsync.
#
# Sincronizza SOLO il codice e le configurazioni (sorgenti, script, configs),
# escludendo dati pesanti e artefatti generati (dataset, checkpoint, log, figure,
# virtualenv, cache). Questi ultimi vivono solo sul cluster e si scaricano in
# locale con il complementare ./scripts/sync_from_cluster.sh.
#
# Prerequisito: alias SSH "gcluster" configurato in ~/.ssh/config con accesso
# passwordless (vedi docs / setup chiave). Override possibili via variabili
# d'ambiente:
#   REMOTE_HOST   host/alias ssh           (default: gcluster)
#   REMOTE_DIR    path progetto sul cluster (default: dl26-projects)
#
# Uso:
#   ./scripts/sync_to_cluster.sh            # sincronizza
#   ./scripts/sync_to_cluster.sh -n         # dry-run (mostra cosa farebbe)
#   ./scripts/sync_to_cluster.sh --delete   # rimuove sul cluster i file non piu presenti in locale
#   REMOTE_DIR=other ./scripts/sync_to_cluster.sh
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gcluster}"
REMOTE_DIR="${REMOTE_DIR:-dl26-projects}"

# Radice del progetto = directory padre di questo script (così è lanciabile da ovunque).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Flag opzionali.
DRY_RUN=""
DELETE=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY_RUN="--dry-run" ;;
    --delete)     DELETE="--delete" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Argomento sconosciuto: $arg (usa -h per l'aiuto)" >&2
      exit 1 ;;
  esac
done

# Esclusioni: tutto ciò che è pesante, generato o specifico dell'ambiente.
EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude 'venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.ipynb_checkpoints/'
  --exclude '.DS_Store'
  --exclude 'data/'                      # dataset (gestito separatamente / già sul cluster)
  --exclude 'experiments/checkpoints/'   # output del training
  --exclude 'experiments/logs/'          # log/tensorboard
  --exclude 'figures/'                   # grafici generati
  --exclude 'wandb/'
)

echo ">> Deploy codice: ${PROJECT_ROOT}/  ->  ${REMOTE_HOST}:${REMOTE_DIR}/"
[ -n "$DRY_RUN" ] && echo ">> DRY-RUN: nessuna modifica verrà applicata."
[ -n "$DELETE" ]  && echo ">> DELETE attivo: i file rimossi in locale verranno rimossi anche sul cluster."

# Crea la directory di destinazione sul cluster (idempotente).
ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_DIR}'"

rsync -avz --progress --human-readable \
  $DRY_RUN $DELETE \
  "${EXCLUDES[@]}" \
  "${PROJECT_ROOT}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo ">> Fatto."
