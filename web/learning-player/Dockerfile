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
# GlitchTip/Sentry DSN for the player's browser error reporting. Optional —
# src/main.ts only initialises Sentry when VITE_SENTRY_DSN_PLAYER is set, so a
# build with no DSN is silent (dev / CI). Must be present at `npm run build`
# time (VITE_* is baked into the bundle). Pass via:
#   docker buildx build --build-arg VITE_SENTRY_DSN_PLAYER=...
ARG VITE_SENTRY_DSN_PLAYER
ENV VITE_SENTRY_DSN_PLAYER=${VITE_SENTRY_DSN_PLAYER}
RUN npm run build

# ---- serve ----
FROM nginx:1.27-alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1
