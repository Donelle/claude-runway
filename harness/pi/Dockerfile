# A throwaway sandbox for running Pi (the coding agent) against the codebase-reviewer's MCP server.
# Built and run by harness/pi/pi_driver.py; see harness/pi/README.md. A development tool, not shipped.
#
# What is NOT in the image: this repository. The driver mounts the reviewer code read-only at run time, so one
# image serves any checkout and nothing here goes stale when the reviewer changes.

FROM node:24-bookworm-slim

# Pinned, so a run today and a run next month use the same agent. Override with --build-arg PI_VERSION=...
ARG PI_VERSION=1.6.0

# git: the reviewer reads the scratch repo through it. ripgrep: Pi's own search tool uses it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # The mounted repo is owned by a different user than the one in the container, and git refuses such a repo
    # ("dubious ownership") unless told otherwise. System-wide, so it holds for whichever user runs it.
    && git config --system --add safe.directory '*'

# --ignore-scripts: Pi's own recommended install; no package gets to run a script while the image is built.
RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}"

# The reviewer imports typing.override (Python 3.12), and bookworm ships 3.11, so uv builds a 3.12 venv.
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python "mcp[cli]>=2.0.0" openai requests

# Nothing about the agent is remembered between runs: the driver points PI_CODING_AGENT_DIR at a scratch mount.
ENV PI_TELEMETRY=0 \
    PI_OFFLINE=1 \
    PI_SKIP_VERSION_CHECK=1

WORKDIR /work
ENTRYPOINT ["pi"]
