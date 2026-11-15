"""
Credential-shape detection and redaction, shared by libs/local_compress_lib.py
(compression must never collect or restore a secret) and the codebase-reviewer
(a server's error text can echo the model's own input, so whoever shows it
redacts it first, via `display_safe`). libs/openai_lib.py deliberately does none
of this: it hands the text over as the server sent it, and what is safe to show
is decided where it is shown. Stdlib-only and importing nothing from this repo.
"""

import re
from typing import Optional

# Credential shapes, which must NEVER be collected. This is a security filter,
# not a noise filter, and it is the one place where over-collection is unsafe.
#
# The failure it prevents, reproduced before this existed: several of the
# identifier patterns in local_compress_lib.py match real secrets -- the snake_case one accepts
# "ghp_<...>" and "sk_live_<...>", the dotted one accepts a three-part JWT.
# When the summarizer omitted a secret (which it did -- dropping it was
# accidentally protective), append_missing_identifiers put it BACK, verbatim,
# and /cr-compact enables this feature unconditionally and then persists the
# result to Qdrant. So the repair step converted a transient secret in a
# session into a durable one in a vector store, and made a leak out of the
# summarizer's accidental good behavior.
#
# Prefix matching rather than entropy scoring on purpose: an entropy heuristic
# would also flag commit SHAs, build ids, and hashes -- exactly the
# identifiers worth restoring. A missed exotic credential shape is a gap; a
# dropped commit SHA is a regression in the feature's whole point.
_CREDENTIAL_PREFIXES = (
    # GitHub
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_",
    # OpenAI / Anthropic style
    "sk-", "sk-ant-", "sk-proj-",
    # Stripe
    "sk_live_", "sk_test_", "rk_live_", "rk_test_", "pk_live_",
    # Slack
    "xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-", "xapp-",
    # GitLab / npm / Docker / HuggingFace / Shopify
    "glpat-", "npm_", "dckr_pat_", "hf_", "shpat_", "shpss_",
    # AWS access key ids / Google API keys
    "AKIA", "ASIA", "AIza",
    # GCP — OAuth2 refresh tokens use the "1//" prefix (documented in Google's
    # OAuth2 reference; the body is base64url after the slashes):
    # 1//<base64url-body>
    "1//",
)

# Credential VALUES, matched wherever they appear -- not anchored to the start
# of a word.
#
# Anchoring to word-start was the second wrong answer here (the first was
# checking only the collected token, which missed fragments of hyphenated
# secrets). A secret is very often not a standalone word, and 6 of 8 realistic
# forms leaked the full value in testing:
#
#     GITHUB_TOKEN=ghp_<...>            leaked
#     export GITHUB_TOKEN=ghp_<...>     leaked
#     {"token":"ghp_<...>"}             leaked
#     --with-token=ghp_<...>            leaked
#     https://x:ghp_<...>@github.com/   leaked
#     TOKEN="ghp_<...>"                 leaked
#
# Worse, credential_token_count() returned 0 for those, so the "withheld"
# disclosure did not fire either -- silently wrong rather than visibly
# incomplete. Matching the value itself covers every wrapper without needing to
# enumerate assignment, JSON, YAML, flag, and URL syntaxes separately.
#
# The (?<![A-Za-z0-9_]) lookbehind is load-bearing: without it the short "sk-"
# prefix matches inside ordinary words ("task-oriented" -> "sk-oriented"),
# which would mark innocent text as a credential span and suppress the real
# identifiers sitting in it.
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    # A known prefix followed by enough body to be a real token. Short
    # prefixes like "sk-" still capture the whole of "sk-ant-api03-<...>",
    # because the body class includes the hyphen.
    + "|".join(re.escape(p) for p in _CREDENTIAL_PREFIXES) + r")[A-Za-z0-9_\-]{6,}"
    # base64url-encoded JSON: a JWT header and up to two following segments.
    r"|(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+){0,2}"
    # PEM private keys: the WHOLE block, header through footer.
    #
    # Matching only the header left the key body outside the credential span,
    # where it could be collected as an identifier and restored verbatim. The
    # CamelCase pattern is \b-anchored and base64 is one long word with no
    # internal boundaries, so a match can only begin at a line's first
    # character -- which makes this probabilistic rather than certain, and is
    # why a typical sample looks clean. Measured: ~1.5% of random 64-char
    # base64 lines begin with a CamelCase-shaped run, so for a 25-line 2048-bit
    # key the chance that at least one line leaks is ~31%, and ~53% for a
    # 50-line 4096-bit key. A leaked line is 64 characters of key material.
    #
    # The \Z fallback covers an unterminated block (a truncated summary), where
    # everything after the header is treated as key material. That
    # over-suppresses -- identifiers past the header go unverified -- which is
    # the correct direction: an unverified identifier is a gap, leaked key
    # material is a breach.
    r"|-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END[A-Z ]*PRIVATE KEY-----|\Z)"
    # Azure Databricks PATs: "dapi" followed by EXACTLY 32 hex characters.
    # Excluded from _CREDENTIAL_PREFIXES (which appends a generic
    # [A-Za-z0-9_\-]{6,} body) because "dapi" alone would then match
    # ordinary identifiers like dapi_request, dapi_handler, etc. -- names
    # common in API code. The documented format is strict hex only, so the
    # dedicated pattern is both more accurate and false-positive-free.
    # The right-side boundary (?![A-Za-z0-9_]) is load-bearing: without it a
    # 40-char git-style hash beginning with "dapi" would match its first 32
    # characters and be silently redacted even though it's not a real token.
    r"|(?<![A-Za-z0-9_])dapi[0-9a-fA-F]{32}(?![A-Za-z0-9_])"
)


def looks_like_credential(token: str) -> bool:
    """True if `token` contains something shaped like a secret.

    Deliberately errs toward withholding: a legitimate identifier wrongly
    withheld is one unverified token, while a credential wrongly restored is a
    secret written into a durable store.
    """
    return bool(_CREDENTIAL_VALUE_RE.search(token))


def credential_spans(text: str) -> list:
    """Character spans of credential values anywhere in `text`."""
    return [m.span() for m in _CREDENTIAL_VALUE_RE.finditer(text)]


_REDACTION_PLACEHOLDER = "[REDACTED-CREDENTIAL]"


def redact_credentials(text: str):
    """
    Replace every credential value in `text` with a placeholder.
    Returns (redacted_text, number_of_replacements).

    Excluding credentials from identifier collection is not sufficient on its
    own, because that only governs what gets RE-APPENDED. Content can reach the
    output by being copied rather than regenerated -- the preamble, a verbatim
    section, a failed-call fallback -- and a model can also echo a secret out of
    its own input. Redacting the finished artifact is the only step that covers
    all of those routes at once.

    Applied only in the artifact-producing modes (preserve_identifiers /
    preserve_sections), which are the ones whose output gets persisted. A
    throwaway gist of a web page is not worth rewriting.
    """
    spans = credential_spans(text)
    if not spans:
        return text, 0
    pieces, last = [], 0
    for start, end in spans:  # finditer spans are ordered and non-overlapping
        pieces.append(text[last:start])
        pieces.append(_REDACTION_PLACEHOLDER)
        last = end
    pieces.append(text[last:])
    return "".join(pieces), len(spans)


# How much of a server's error text goes into a user-facing message. The text can echo the
# model's own input, so it is capped.
SERVER_DETAIL_MAX_CHARS = 200


def display_safe(text: Optional[str], max_chars: int = SERVER_DETAIL_MAX_CHARS) -> Optional[str]:
    """Server or model text made safe to put in a message or a log, or None if nothing is left.

    Redaction runs BEFORE whitespace-collapsing and truncation: cutting a credential in half first would
    leave a fragment the pattern no longer matches. Collapsing whitespace keeps the text on one line, which
    the outage-notice parsing (it splits on a blank line) and a one-line error message both rely on.
    """
    if not text:
        return None
    text, _ = redact_credentials(text)
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text
