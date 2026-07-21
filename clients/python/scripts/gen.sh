#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from the canonical proto. grpcio-tools bundles its own
# protoc, so no system protoc is needed. Run from anywhere:
#   uv run --project clients/python --extra dev clients/python/scripts/gen.sh
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"        # clients/python
proto_root="$(cd "$here/../../proto" && pwd)"   # repo proto/
out="$here/memlake_client/v1"
mkdir -p "$out"

python -m grpc_tools.protoc \
  -I "$proto_root" \
  --python_out="$out" \
  --grpc_python_out="$out" \
  "$proto_root/memlake/v1/memlake.proto"

# The generated files import each other by proto path (memlake.v1.*); rewrite those to the
# flat package layout we ship so `from memlake_client.v1 import ...` works.
gen_dir="$out/memlake/v1"
mv "$gen_dir/memlake_pb2.py" "$out/memlake_pb2.py"
mv "$gen_dir/memlake_pb2_grpc.py" "$out/memlake_pb2_grpc.py"
rm -rf "$out/memlake"

# Fix the cross-import in the grpc stub to the flat module.
sed -i.bak 's/from memlake\.v1 import memlake_pb2/from memlake_client.v1 import memlake_pb2/' \
  "$out/memlake_pb2_grpc.py"
sed -i.bak 's/^import memlake_pb2/from memlake_client.v1 import memlake_pb2/' \
  "$out/memlake_pb2_grpc.py"
rm -f "$out/memlake_pb2_grpc.py.bak"

echo "generated stubs in $out"
