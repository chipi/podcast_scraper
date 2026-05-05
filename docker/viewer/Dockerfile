# GitHub #659: production viewer — Vue (Vite) static build + Nginx reverse proxy to API.
# Build from repository root: docker build -f docker/viewer/Dockerfile .

# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS build
WORKDIR /viewer

# Optional build-time override for the Graph time lens default. Empty
# string = "all time" (no lower bound); positive integer N = "last N
# days". Used by stack-test / smoke builds where fixture dates are
# static and must not depend on wall-clock time. Production keeps the
# code default (``last 7 days``).
ARG VITE_DEFAULT_GRAPH_LENS_DAYS
ENV VITE_DEFAULT_GRAPH_LENS_DAYS=${VITE_DEFAULT_GRAPH_LENS_DAYS}

# Sentry DSN + env tags for the viewer's runtime error reporting.
# All three are optional — Vue main.ts only initialises Sentry when
# VITE_SENTRY_DSN_VIEWER is set, so a build with no DSN is silent.
# See RFC-081 §Layer 2 + issue #681. Pass via:
#   docker buildx build --build-arg VITE_SENTRY_DSN_VIEWER=...
ARG VITE_SENTRY_DSN_VIEWER
ENV VITE_SENTRY_DSN_VIEWER=${VITE_SENTRY_DSN_VIEWER}
# VITE_PODCAST_ENV deliberately removed: env tag is now injected at
# request time by nginx ``sub_filter`` reading the container's
# ``PODCAST_ENV`` env var (see docker/viewer/default.conf.template).
# This lets ONE viewer image serve every environment with the right
# Sentry env tag — no double-publish, no per-env image tags.
ARG VITE_PODCAST_RELEASE
ENV VITE_PODCAST_RELEASE=${VITE_PODCAST_RELEASE}

COPY web/gi-kg-viewer/package.json web/gi-kg-viewer/package-lock.json ./
RUN npm ci

COPY web/gi-kg-viewer/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=build /viewer/dist/ /usr/share/nginx/html/
# COPY into /etc/nginx/templates/ so the nginx:alpine entrypoint runs
# envsubst on it at container start — needed for the PODCAST_ENV
# runtime injection in the ``sub_filter`` directive. Output lands at
# /etc/nginx/conf.d/default.conf.
COPY docker/viewer/default.conf.template /etc/nginx/templates/default.conf.template

# Defaults so envsubst on the template doesn't leave literal
# ``${PODCAST_ENV}`` placeholders if the operator forgot to set the
# env on the container (Sentry tag falls back to ``dev`` in main.ts).
ENV PODCAST_ENV=dev
ENV PODCAST_RELEASE=
# Restrict envsubst to ``${PODCAST_*}`` so nginx's own ``${host}`` /
# ``${uri}`` etc. (and any future single-dollar-style nginx vars in
# the template) are NOT mangled.
ENV NGINX_ENVSUBST_FILTER=^PODCAST_

# Healthcheck: accept ANY HTTP response (200 or 401 with auth_basic
# in prod) as proof nginx is alive. busybox wget is present on
# Alpine; ``-S`` dumps response headers to stderr.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qSO /dev/null http://127.0.0.1/ 2>&1 | grep -q 'HTTP/' || exit 1

EXPOSE 80
