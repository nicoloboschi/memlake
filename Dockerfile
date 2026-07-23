# syntax=docker/dockerfile:1.7
#
# One image, two deployment modes (mirrors the README topology): the same binary runs
#   mlake-server serve ...    # stateless gRPC API pod
#   mlake-server index ...    # the async indexer Deployment
# The entrypoint is the binary, so compose/k8s supplies `serve`/`index` + flags as args.
#
# Multi-stage + cache-friendly: manifests are copied first so the dependency *download*
# layer is cached independently of source edits; BuildKit cache mounts keep the compiled
# `target/` and cargo registry warm across rebuilds. Requires BuildKit (docker >= 23).

# ---- build ----------------------------------------------------------------
FROM rust:1-bookworm AS builder
WORKDIR /app

# 1) Manifests only. `cargo fetch` resolves + downloads the whole workspace's deps from
#    these + Cargo.lock, so this layer is reused until a Cargo.toml or the lock changes.
COPY Cargo.toml Cargo.lock ./
COPY crates/mlake-core/Cargo.toml   crates/mlake-core/Cargo.toml
COPY crates/mlake-store/Cargo.toml  crates/mlake-store/Cargo.toml
COPY crates/mlake-wal/Cargo.toml    crates/mlake-wal/Cargo.toml
COPY crates/mlake-ivf/Cargo.toml    crates/mlake-ivf/Cargo.toml
COPY crates/mlake-fts/Cargo.toml    crates/mlake-fts/Cargo.toml
COPY crates/mlake-graph/Cargo.toml  crates/mlake-graph/Cargo.toml
COPY crates/mlake-index/Cargo.toml  crates/mlake-index/Cargo.toml
COPY crates/mlake-bench/Cargo.toml  crates/mlake-bench/Cargo.toml
COPY crates/mlake-perf/Cargo.toml   crates/mlake-perf/Cargo.toml
COPY crates/mlake-server/Cargo.toml crates/mlake-server/Cargo.toml
# Stub targets so the workspace manifests parse for `cargo fetch` (real sources arrive next).
RUN set -eux; \
    for c in core store wal ivf fts graph index; do \
      mkdir -p "crates/mlake-$c/src"; : > "crates/mlake-$c/src/lib.rs"; \
    done; \
    mkdir -p crates/mlake-index/benches; echo 'fn main() {}' > crates/mlake-index/benches/query.rs; \
    for c in bench perf server; do \
      mkdir -p "crates/mlake-$c/src"; echo 'fn main() {}' > "crates/mlake-$c/src/main.rs"; \
    done
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo fetch

# 2) Sources, then a release build of just the server crate (protoc is vendored by build.rs,
#    so no system protobuf-compiler is needed).
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release -p mlake-server && \
    cp target/release/mlake-server /usr/local/bin/mlake-server

# ---- runtime --------------------------------------------------------------
FROM debian:bookworm-slim AS runtime
# ca-certificates: rustls needs a trust store to reach real AWS S3 over TLS (MinIO over
# http does not, but the same image must serve both). netcat: the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/mlake-server /usr/local/bin/mlake-server

# Args pass through: `docker run memlake serve --addr ...` or `... index --namespaces ...`.
ENTRYPOINT ["mlake-server"]
CMD ["serve"]
