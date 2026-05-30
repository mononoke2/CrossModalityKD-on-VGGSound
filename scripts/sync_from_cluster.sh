#!/usr/bin/env bash
#
# sync_from_cluster.sh — Scarica in locale i risultati prodotti sul cluster DMI.
#
# Complementare di ./scripts/sync_to_cluster.sh: recupera SOLO gli artefatti
# generati dai job (checkpoint, log/tensorboard, figure) senza ri-scaricare il
# codice. Utile per ispezionare risultati, generare grafici o scrivere il report
# in locale.
#
# Prerequisito: alias SSH 'gcluster' in ~/.ssh/config oppure impostazione della
# variabile d'ambiente CLUSTER_USER con il proprio username di ateneo (codice fiscale).
#
# Variabili d'ambiente configurabili:
#   CLUSTER_USER  username del cluster     (es. CLUSTER_USER=codicefiscale)
#   REMOTE_HOST   host/alias ssh alternativo (default: autodetect alias 'gcluster')
#   REMOTE_DIR    path progetto sul cluster  (default: dl26-projects)
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

# Determinazione dinamica e generale di REMOTE_HOST
if [ -n "${REMOTE_HOST:-}" ]; then
  # Se impostato esplicitamente via env, usiamo quello
  :
elif [ -n "${CLUSTER_USER:-}" ]; then
  # Se è specificato l'username del cluster, costruiamo l'host reale
  REMOTE_HOST="${CLUSTER_USER}@gcluster.dmi.unict.it"
else
  # Controlliamo se l'alias 'gcluster' è configurato in ~/.ssh/config
  # 'ssh -G gcluster' restituisce la configurazione espansa. Se l'alias esiste,
  # l'hostname conterrà 'gcluster.dmi.unict.it'.
  if ssh -G gcluster 2>/dev/null | grep -qi '^hostname gcluster.dmi.unict.it'; then
    REMOTE_HOST="gcluster"
  else
    echo "Errore: l'alias SSH 'gcluster' non è configurato e la variabile CLUSTER_USER non è definita." >&2
    echo "Poiché non hai un file ~/.ssh/config, specifica il tuo username del cluster (codice fiscale)." >&2
    echo "Esempio d'uso:" >&2
    echo "  CLUSTER_USER=il_tuo_codice_fiscale $0 [opzioni]" >&2
    echo "Oppure configura un alias SSH 'gcluster' in ~/.ssh/config." >&2
    exit 1
  fi
fi

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
