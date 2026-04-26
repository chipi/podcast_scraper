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

COPY web/gi-kg-viewer/package.json web/gi-kg-viewer/package-lock.json ./
RUN npm ci

COPY web/gi-kg-viewer/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=build /viewer/dist/ /usr/share/nginx/html/
COPY docker/viewer/nginx.conf /etc/nginx/conf.d/default.conf

# BusyBox wget is present on Alpine-based nginx image
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

EXPOSE 80
