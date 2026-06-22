# Standalone prod observability control plane (#803).
#
# Intentionally LIGHT: installs ONLY podcast_obs + httpx/PyYAML/mcp — never the heavy
# podcast_scraper pipeline deps (torch/spaCy/...). The podcast_obs package has zero coupling to
# podcast_scraper, so copying just that one directory is enough. The result is a small image you
# run on your MBP, an Orb, or a Mac mini inside the Tailscale network as a mini control plane.
#
# Build (from repo root):
#   docker build -f docker/observability/Dockerfile -t podcast-obs:latest .
# One-shot probe:
#   docker run --rm -e PODCAST_OBS_API_BASE=https://prod-podcast.<tailnet> podcast-obs summary
# Run the agent-facing MCP/HTTP control plane:
#   docker run --rm -p 8848:8848 \
#     -v "$PWD/config/observability.yaml:/config/observability.yaml:ro" \
#     -e PODCAST_OBS_CONFIG=/config/observability.yaml podcast-obs
FROM python:3.12-slim

# Pinned to match the pyproject [observability] extra.
RUN pip install --no-cache-dir \
      "httpx>=0.28.1,<1.0.0" \
      "PyYAML>=6.0.3,<7.0.0" \
      "mcp>=1.2.0,<2.0.0"

WORKDIR /app
COPY src/podcast_obs /app/podcast_obs

# Run unprivileged.
RUN useradd --create-home --uid 10001 obs
USER obs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# streamable-http transport so an agent on another tailnet box can reach it. Override the CMD for
# stdio (`serve --transport stdio`) or a one-shot probe (`summary --target prod`).
EXPOSE 8848
ENTRYPOINT ["python", "-m", "podcast_obs"]
CMD ["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8848"]
