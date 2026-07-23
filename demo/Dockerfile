# The demo trickle writer: a tiny Python image that installs the memlake python client and runs
# `demo/writer.py` forever. Build context is the REPO ROOT (it needs clients/python too):
#
#   docker build -f demo/Dockerfile -t memlake-demo-writer .
#
FROM python:3.12-slim

WORKDIR /app

# The client package first (its own deps: grpcio, protobuf), then the writer. Copying the client
# separately keeps the pip layer cached until the client changes.
COPY clients/python /app/clients/python
RUN pip install --no-cache-dir /app/clients/python

COPY demo/writer.py /app/writer.py

# Config comes from the environment (see writer.py): MEMLAKE_ADDR, MEMLAKE_DEMO_NAMESPACE,
# MEMLAKE_DEMO_INTERVAL, MEMLAKE_DEMO_BATCH. Overridable as CLI args too.
ENTRYPOINT ["python", "/app/writer.py"]
