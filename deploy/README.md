# Deploying memlake

Two Deployments (stateless `serve`, async `indexer`) plus an Envoy L7 proxy that consistent-hashes
gRPC by the `x-memlake-namespace` header — so one serve pod owns a namespace (cache + commit
affinity) and pinned namespaces route to dedicated pods. Object storage (real AWS S3) is the only
stateful dependency.

## Images

Built for `linux/amd64` and pushed to ghcr:

```bash
SHA=$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 -f Dockerfile \
  -t ghcr.io/nicoloboschi/memlake-server:dev -t ghcr.io/nicoloboschi/memlake-server:$SHA --push .
docker buildx build --platform linux/amd64 -f admin/Dockerfile \
  -t ghcr.io/nicoloboschi/memlake-admin:dev  -t ghcr.io/nicoloboschi/memlake-admin:$SHA  --push .
```

## Dev cluster (hindsight-dev, isolated namespace)

Prereqs: `kubectl` context on the target cluster (`gcloud auth login` for GKE), `helm`, and `gh`
logged in with `read:packages` (for the ghcr pull secret). AWS S3 creds come from the repo-root
`.env` (gitignored) — the `MEMLAKE_QUERY_S3_*` and `MEMLAKE_INDEXER_S3_*` blocks, no endpoint set
(so it talks to real AWS).

One command (idempotent, creates the namespace + secrets + release):

```bash
./deploy/deploy-dev.sh
# extra helm flags pass through, e.g.:  ./deploy/deploy-dev.sh --set serve.replicas=5
```

It deploys into namespace **`memlake-dev`** so cleanup is a single:

```bash
kubectl delete namespace memlake-dev
```

## Reaching it

```bash
# gRPC entrypoint (through the proxy). Clients send x-memlake-namespace; the memlake client does it.
kubectl -n memlake-dev port-forward svc/memlake-proxy 50050:50050
# admin UI
kubectl -n memlake-dev port-forward svc/memlake-admin 3000:3000   # http://localhost:3000
```

## Namespace pinning (compute isolation)

Give a namespace its own serve pods and route only its traffic there. Add a pin in values:

```yaml
proxy:
  pins:
    - namespace: acme-corp
      service: memlake-serve-acme   # a headless Service selecting the dedicated pods
```

The proxy grows a header-match route (`x-memlake-namespace: acme-corp` → the dedicated cluster) in
front of the consistent-hash default; everything else stays on the shared pool. (The dedicated
serve Deployment/Service for a pin is deployed separately — same serve template, distinct labels.)

## Production path (Gateway API)

The standalone Envoy here needs no CRDs. On a cluster with the Gateway API + Envoy Gateway
installed, the same routing is an `HTTPRoute` (header match for pins) + a `BackendTrafficPolicy`
(`ConsistentHash` on the header) — see `docs/` / the Envoy Gateway load-balancing task.
