# RFC-079 / GitHub #659: production viewer — Vue (Vite) static build + Nginx reverse proxy to API.
# Build from repository root: docker build -f docker/viewer/Dockerfile .

# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS build
WORKDIR /viewer

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
