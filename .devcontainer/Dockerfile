FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Upstream image's yarnpkg.com apt source has a rotated GPG key
# (NO_PUBKEY 62D54FD4003F6525) that breaks ``apt-get update`` during
# devcontainer feature install (docker-in-docker, github-cli, node).
# yarn isn't used by this project (Vue/Vite uses npm), so just remove
# the unused source list. apt's GPG verification on every other repo
# (debian.org / debian-security) keeps working with full integrity.
#
# Once Microsoft refreshes the python:1-3.12-bookworm tag with a valid
# yarnpkg keyring this Dockerfile can be deleted and devcontainer.json
# can switch back to ``"image": "mcr.microsoft.com/devcontainers/..."``.
RUN rm -f /etc/apt/sources.list.d/yarn.list /usr/share/keyrings/yarnpkg.gpg
