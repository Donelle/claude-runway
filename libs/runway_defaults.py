"""Defaults more than one part of the toolkit needs, written once so they are not retyped in every file."""

import os

# LM Studio's OpenAI-compatible endpoint: CLAUDE_RUNWAY_LMSTUDIO_URL when set, else LM Studio's own default address.
LMSTUDIO_URL = os.environ.get("CLAUDE_RUNWAY_LMSTUDIO_URL", "http://localhost:1234/v1")
