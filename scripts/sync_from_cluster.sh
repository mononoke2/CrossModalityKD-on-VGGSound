#!/usr/bin/env bash
#
# sync_from_cluster.sh — Scarica in locale i risultati prodotti sul cluster DMI.
#
# Complementare di ./scripts/sync_to_cluster.sh: recupera SOLO gli artefatti
# generati dai job (checkpoint, log/tensorboard, figure) senza ri-scaricare il
# codice. Utile per ispezionare risultati, generare grafici o scrivere il report
# in locale.
#
# Prerequisito: alias SSH "gcluster" con accesso passwordless. Override via env:
#   REMOTE_HOST   host/alias ssh           (default: gcluster)
#   REMOTE_DIR    path progetto sul cluster (default: dl26-projects)
#
# Uso:
#   ./scripts/sync_from_cluster.sh                 # scarica tutti gli artefatti
#   ./scripts/sync_from_cluster.sh -n              # dry-run
#   ./scripts/sync_from_cluster.sh checkpoints     # solo i checkpoint
#   ./scripts/sync_from_cluster.sh logs figures    # subset di artefatti
#
# Artefatti supportati (path relativi al progetto): checkpoints, logs, figures.
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-gcluster}"
REMOTE_DIR="${REMOTE_DIR:-dl26-projects}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Mappa: nome-artefatto -> path relativo nel progetto.
declare -A ARTIFACTS=(
  [checkpoints]="experiments/checkpoints"
  [logs]="experiments/logs"
  [figures]="figures"
)

DRY_RUN=""
SELECTED=()
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY_RUN="--dry-run" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    checkpoints|logs|figures) SELECTED+=("$arg") ;;
    *)
      echo "Artefatto/argomento sconosciuto: $arg (validi: checkpoints logs figures; -h per aiuto)" >&2
      exit 1 ;;
  esac
done

# Se non è stato selezionato nulla, scarica tutti gli artefatti.
if [ "${#SELECTED[@]}" -eq 0 ]; then
  SELECTED=(checkpoints logs figures)
fi

echo ">> Download risultati: ${REMOTE_HOST}:${REMOTE_DIR}/  ->  ${PROJECT_ROOT}/"
echo ">> Artefatti: ${SELECTED[*]}"
[ -n "$DRY_RUN" ] && echo ">> DRY-RUN: nessuna modifica verrà applicata."

for name in "${SELECTED[@]}"; do
  rel="${ARTIFACTS[$name]}"
  remote_path="${REMOTE_DIR}/${rel}/"
  local_path="${PROJECT_ROOT}/${rel}/"

  # Salta se la cartella non esiste ancora sul cluster (nessun risultato prodotto).
  if ! ssh "$REMOTE_HOST" "test -d '${REMOTE_DIR}/${rel}'"; then
    echo ">> [skip] ${name}: '${remote_path}' non esiste ancora sul cluster."
    continue
  fi

  echo ">> [${name}] ${remote_path} -> ${local_path}"
  mkdir -p "$local_path"
  rsync -avz --progress --human-readable \
    $DRY_RUN \
    "${REMOTE_HOST}:${remote_path}" \
    "$local_path"
done

echo ">> Fatto."
