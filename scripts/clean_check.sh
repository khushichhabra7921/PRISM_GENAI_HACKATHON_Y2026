#!/usr/bin/env bash
# Clean-machine check: copy exactly what a fresh `git clone` contains (tracked + non-ignored files; no .venv,
# cache/, .env, bench caches) into a temp dir, then `docker compose up --build` there, wait for /health = 200
# and send sample requests.
#
#   bash scripts/clean_check.sh            # fresh Ollama model volume (downloads ~1 GB)
#   REUSE_OLLAMA=1 bash scripts/clean_check.sh   # mount the existing sgte_ollama model volume
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_PORT="${API_PORT:-8001}"
DIR="$(mktemp -d)/sgte_clean"
mkdir -p "$DIR"
cd "$ROOT"
git ls-files -co --exclude-standard -z | xargs -0 cp --parents -t "$DIR"
echo "[clean] copied $(git ls-files -co --exclude-standard | wc -l) files to $DIR"
cd "$DIR"
[ -e .env ] && { echo "[clean] ERROR: .env leaked into the clone"; exit 1; }
if [ "${REUSE_OLLAMA:-0}" = "1" ]; then
  printf 'volumes:\n  ollama:\n    external: true\n    name: sgte_ollama\n' > docker-compose.override.yml
fi
export SGTE_API_PORT="$API_PORT" OLLAMA_PORT="${OLLAMA_PORT:-11435}" SGTE_LLM_FALLBACK=""
T0=$(date +%s)
docker compose -p sgte_clean up -d --build
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "localhost:$API_PORT/health" || true)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "[clean] /health -> $code after $(( $(date +%s) - T0 )) s"
[ "$code" = "200" ] || { docker compose -p sgte_clean logs --tail 50 api; exit 1; }
curl -s "localhost:$API_PORT/health"; echo
curl -s "localhost:$API_PORT/v1/info"; echo
# official contract: the first input.txt complaint with its SIIS payload, sample_output.json shape
python -c "
import json; r=json.load(open('data/siis_responses.json',encoding='utf-8'))['responses'][0]
q=open('data/input.txt',encoding='utf-8').readline().strip()
print(json.dumps({'query': q, 'siis_response': r['siis_response']}))" > /tmp/sgte_payload.json
curl -s -X POST "localhost:$API_PORT/v1/troubleshoot?view=contract" -H "Content-Type: application/json" \
     -d @/tmp/sgte_payload.json | python -c "
import json,sys; r=json.load(sys.stdin); g=r['response']['contexts']
print(f\"[clean] official row_1 with payload: keys={list(r)} goals={len(g)} actions={[a['actionName'] for a in g[0]['actions']][:4] if g else []}\")"
for q in "my touchscreen is laggy and slow to respond" "my screen went completely black" \
         "my nexa watch wont sync steps"; do
  curl -s -X POST "localhost:$API_PORT/v1/troubleshoot" -H "Content-Type: application/json" \
       -d "{\"query\": \"$q\"}" | python -c "
import json,sys; r=json.load(sys.stdin); m=r['meta']
print(f\"[clean] {r['query'][:50]!r}: goals={len(r['response']['contexts'])} cache_hit={m['cache_hit']} \"
      f\"fallback={m['fallback']} latency_ms={m['latency_ms']} actions={[a['actionName'] for g in r['response']['contexts'] for a in g['actions']][:4]}\")"
done
echo "[clean] OK. Stop with: docker compose -p sgte_clean down -v   (in $DIR)"
