# The admin inspection console (Next.js). Build context is the REPO ROOT, because the app loads
# the memlake gRPC contract from proto/ at runtime:
#
#   docker build -f admin/Dockerfile -t memlake-admin .
#
# See admin/Dockerfile.dockerignore — it overrides the repo-root .dockerignore (which is tuned for
# the server image and excludes admin/) so this build can actually see admin/ and proto/.
FROM node:22-bookworm-slim

WORKDIR /app/admin

# Install deps first (cached until the lockfile changes). NODE_ENV is left unset here so devDeps
# (typescript, tailwind, eslint-config-next) are installed — `next build` needs them.
COPY admin/package.json admin/package-lock.json ./
RUN npm ci

# The gRPC contract, loaded at runtime by @grpc/proto-loader (no generated stubs). Copy it before
# the build so a static trace, if any, resolves; the runtime path is set below.
COPY proto /app/proto

# App source, then a production build.
COPY admin/ ./
RUN npm run build

# Runtime config. MEMLAKE_ADDR points at the serve service on the compose network; the proto path
# is absolute so it does not depend on the process cwd.
ENV NODE_ENV=production \
    MEMLAKE_ADDR=serve:50051 \
    MEMLAKE_PROTO_PATH=/app/proto/memlake/v1/memlake.proto \
    PORT=3000

EXPOSE 3000
CMD ["npm", "start"]
