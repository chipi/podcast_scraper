# Consumer Learning Player — its OWN static image (RFC-099 §10), independent of the API
# image. Build the Vite bundle, serve it from nginx as an SPA. The container serves UI +
# proxies /api to the backend; it proxies NO audio (bridge-never-rehost) — the client plays
# origin media URLs directly. One API → this web PWA + a future native mobile client.

# ---- build ----
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
# A baked GlitchTip DSN, for a build that runs without this nginx (a local test image). The
# deployed web player takes its DSN at container start instead (PLAYER_SENTRY_DSN_WEB, below),
# so CI passes nothing here. src/main.ts reports nothing when neither is set.
ARG VITE_SENTRY_DSN_PLAYER
ENV VITE_SENTRY_DSN_PLAYER=${VITE_SENTRY_DSN_PLAYER}
# Umami analytics (optional, cookieless). main.ts injects the tracking script only
# when BOTH are set, so a build without them is a no-op. Baked at `npm run build`.
#   docker buildx build --build-arg VITE_UMAMI_WEBSITE_ID=... --build-arg VITE_UMAMI_SRC=...
ARG VITE_UMAMI_WEBSITE_ID
ENV VITE_UMAMI_WEBSITE_ID=${VITE_UMAMI_WEBSITE_ID}
ARG VITE_UMAMI_SRC
ENV VITE_UMAMI_SRC=${VITE_UMAMI_SRC}
# The build's commit, for the GlitchTip release tag (`sha-<7>`, ADR-0010) and the build info.
# The build context has no .git, so without it vite.config.ts falls back to "unknown" and every
# player event carried release `sha-unknown` (homelab check 2026-10-27).
#   docker buildx build --build-arg BUILD_SHA=<full sha>
ARG BUILD_SHA
ENV BUILD_SHA=${BUILD_SHA}
RUN npm run build

# ---- serve ----
FROM nginx:1.27-alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
# nginx.conf is a template: the image's entrypoint renders ${PLAYER_*} into it at start. The
# output goes to /tmp, not conf.d, because the deployed container runs with a read-only root
# and conf.d is deliberately not a tmpfs (an empty tmpfs there left nginx with no server block,
# 2026-07-23). conf.d holds only an include of that exact file: if rendering ever fails, nginx
# refuses to start instead of serving nothing.
COPY nginx.conf /etc/nginx/templates/player.conf.template
RUN printf 'include /tmp/player.conf;\n' > /etc/nginx/conf.d/default.conf
ENV NGINX_ENVSUBST_OUTPUT_DIR=/tmp
ENV NGINX_ENVSUBST_FILTER=^PLAYER_
ENV PLAYER_SENTRY_DSN_WEB=
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1
