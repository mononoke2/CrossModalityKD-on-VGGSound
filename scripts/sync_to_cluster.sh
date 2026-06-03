#!/usr/bin/env bash
#
# sync_to_cluster.sh — Deploy del codice locale verso il cluster DMI via rsync.
#
# Sincronizza SOLO il codice e le configurazioni (sorgenti, script, configs),
# escludendo dati pesanti e artefatti generati (dataset, checkpoint, log, figure,
# virtualenv, cache). Questi ultimi vivono solo sul cluster e si scaricano in
# locale con il complementare ./scripts/sync_from_cluster.sh.
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
#   ./scripts/sync_to_cluster.sh            # sincronizza SOLO il codice (data/ e pretrained_weights/ esclusi)
#   ./scripts/sync_to_cluster.sh -n         # dry-run (mostra cosa farebbe)
#   ./scripts/sync_to_cluster.sh --data     # include anche data/ (carica il dataset sul cluster)
#   ./scripts/sync_to_cluster.sh --models   # include pretrained_weights/ (pesi ViT-B/16 per il cluster)
#   ./scripts/sync_to_cluster.sh --delete   # rimuove sul cluster i file non piu presenti in locale
#   REMOTE_DIR=other ./scripts/sync_to_cluster.sh
#
# Nota: il cluster DMI non ha internet in uscita, quindi il dataset va scaricato
# in locale e poi caricato qui con --data (rsync e' resumable: utile per i ~65GB).
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

# Radice del progetto = directory padre di questo script (così è lanciabile da ovunque).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Flag opzionali.
DRY_RUN=""
DELETE=""
WITH_DATA=""
WITH_MODELS=""
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY_RUN="--dry-run" ;;
    --delete)     DELETE="--delete" ;;
    --data)       WITH_DATA=1 ;;
    --models)     WITH_MODELS=1 ;;
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
  --exclude 'docs/EXPERIMENT_LOG.md'
  --exclude 'venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.ipynb_checkpoints/'
  --exclude '.DS_Store'
  --exclude 'experiments/checkpoints/'   # output del training
  --exclude 'experiments/logs/'          # log/tensorboard
  --exclude 'figures/'                   # grafici generati
  --exclude 'wandb/'
)
# data/ e pretrained_weights/ sono esclusi di default.
if [ -z "$WITH_DATA" ]; then
  EXCLUDES+=(--exclude 'data/')
fi
if [ -z "$WITH_MODELS" ]; then
  EXCLUDES+=(--exclude 'pretrained_weights/')
fi

SCOPE="codice"
[ -n "$WITH_DATA" ]   && SCOPE="${SCOPE} + data"
[ -n "$WITH_MODELS" ] && SCOPE="${SCOPE} + pretrained_weights"
echo ">> Deploy ${SCOPE}: ${PROJECT_ROOT}/  ->  ${REMOTE_HOST}:${REMOTE_DIR}/"
[ -n "$WITH_DATA" ] && echo ">> --data attivo: verrà caricata anche la cartella data/ (può essere molto pesante)."
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
