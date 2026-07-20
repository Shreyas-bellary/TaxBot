#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-gen-lang-client-0055378858}"
ENV_FILE="${1:-.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

command -v gcloud >/dev/null 2>&1 || {
  echo "gcloud is required" >&2
  exit 1
}
command -v poetry >/dev/null 2>&1 || {
  echo "poetry is required to parse ${ENV_FILE}" >&2
  exit 1
}

# env_name:secret_id:required(1|0)
SECRET_ENTRIES=(
  "TAXBOT_POSTGRES_DSN:taxbot-postgres-dsn:1"
  "TAXBOT_GEMINI_API_KEY:taxbot-gemini-api-key:1"
  "TAXBOT_HUGGINGFACE_API_TOKEN:taxbot-huggingface-api-token:1"
  "TAXBOT_QDRANT_API_KEY:taxbot-qdrant-api-key:1"
  "TAXBOT_OPENROUTER_API_KEY:taxbot-openrouter-api-key:0"
  "TAXBOT_UNSTRUCTURED_API_KEY:taxbot-unstructured-api-key:0"
)

read_env_value() {
  local env_name="$1"
  ENV_FILE="${ENV_FILE}" ENV_NAME="${env_name}" poetry run python - <<'PY'
import os
from dotenv import dotenv_values

value = dotenv_values(os.environ["ENV_FILE"]).get(os.environ["ENV_NAME"])
if value:
    print(value, end="")
PY
}

for entry in "${SECRET_ENTRIES[@]}"; do
  env_name="${entry%%:*}"
  rest="${entry#*:}"
  secret_id="${rest%%:*}"
  required="${rest##*:}"

  value="$(read_env_value "${env_name}")"

  if [[ -z "${value}" ]]; then
    if [[ "${required}" == "1" ]]; then
      echo "Required value ${env_name} is missing from ${ENV_FILE}" >&2
      exit 1
    fi
    echo "Skipping optional ${env_name}"
    continue
  fi

  printf '%s' "${value}" |
    gcloud secrets versions add "${secret_id}" \
      --project="${PROJECT_ID}" \
      --data-file=- \
      --quiet >/dev/null
  unset value
  echo "Updated ${secret_id}"
done
