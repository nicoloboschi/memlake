#!/usr/bin/env bash
# Deploy memlake to the hindsight-dev cluster in an isolated namespace, over REAL AWS S3.
#
# Idempotent: safe to re-run. Reads AWS S3 creds from the repo-root .env (gitignored) and the ghcr
# pull token from `gh auth token`. Nothing secret is written to the repo or the chart.
#
#   ./deploy/deploy-dev.sh
#
# Cleanup: kubectl delete namespace "$NS"
set -euo pipefail

NS="${MEMLAKE_NS:-memlake-dev}"
RELEASE="${MEMLAKE_RELEASE:-memlake}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

command -v kubectl >/dev/null || { echo "kubectl not found"; exit 1; }
command -v helm    >/dev/null || { echo "helm not found"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (AWS S3 creds)"; exit 1; }

echo "context: $(kubectl config current-context)"
kubectl get nodes >/dev/null || { echo "cannot reach cluster — run: gcloud auth login"; exit 1; }

echo "==> namespace $NS"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

echo "==> ghcr image pull secret"
kubectl -n "$NS" create secret docker-registry ghcr \
  --docker-server=ghcr.io \
  --docker-username="$(gh api user --jq .login)" \
  --docker-password="$(gh auth token)" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> AWS S3 secret (memlake-s3) from .env (QUERY + INDEXER blocks, uncommented only)"
grep -E '^MEMLAKE_(QUERY|INDEXER)_S3_' "$ENV_FILE" \
  | kubectl -n "$NS" create secret generic memlake-s3 --from-env-file=/dev/stdin \
      --dry-run=client -o yaml | kubectl apply -f -

echo "==> helm upgrade --install"
helm upgrade --install "$RELEASE" "$REPO_ROOT/deploy/helm/memlake" \
  --namespace "$NS" \
  "$@"

echo "==> rollout"
# serve is a StatefulSet (stable pod identities for the consistent-hash proxy); the rest are
# Deployments. Resolve serve's kind so the deploy works whichever it is.
serve_kind="$(kubectl -n "$NS" get statefulset "$RELEASE"-serve >/dev/null 2>&1 && echo statefulset || echo deploy)"
kubectl -n "$NS" rollout status "$serve_kind"/"$RELEASE"-serve --timeout=180s
kubectl -n "$NS" rollout status deploy/"$RELEASE"-indexer --timeout=180s
kubectl -n "$NS" rollout status deploy/"$RELEASE"-proxy   --timeout=180s || true

echo "done. Pods:"
kubectl -n "$NS" get pods -o wide
