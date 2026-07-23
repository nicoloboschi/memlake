"""memlake compute-vs-SLA load harness.

Drives concurrent gRPC query load against the REAL serve+index topology (see
`docker-compose.perf.yml`) under container CPU/memory limits, and accounts the serve
container's CPU per achieved QPS — the question the in-process `mlake-perf` cannot answer.
"""
