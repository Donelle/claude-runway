#!/usr/bin/env python3
"""Table tests for libs/local_compress_lib.py's focus-shape detection.

Stdlib-only and no network -- both predicates under test are pure functions
over the focus string, so this needs neither LM Studio nor Qdrant:

    .venv/bin/python -m unittest discover -s tests

What these pin down is the DIRECTION of a mistake. _looks_non_selective
decides whether classify_relevant runs at all, and being wrong is silent
either way: a focus wrongly called non-selective quietly stops filtering
(the caller gets a summary of the whole document instead of the part they
asked for), and a focus wrongly called selective quietly hands a NO-biased
classifier the chance to reject everything -- which is exactly the bug this
predicate was added to fix, where DEFAULT_FOCUS caused the PostToolUse hook
to compress only output that happened to contain an error.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"))

from local_compress_lib import (  # noqa: E402
    DEFAULT_FOCUS,
    _compression_cache,
    _compression_cache_key,
    _find_first_heading_boundary,
    _looks_non_selective,
    _looks_positional,
    _normalize_focus,
    append_missing_identifiers,
    append_trailing_summary_if_missing,
    clear_compression_cache,
    compress,
    credential_token_count,
    derive_compact_label,
    find_trailing_summary_line,
    identifier_tokens,
    looks_like_credential,
    missing_identifiers,
    missing_sections,
    redact_and_disclose,
    redact_credentials,
    restore_missing_sections,
    section_has_mixed_runs,
    section_is_verbatim,
    split_prose_bullet_runs,
    split_sections,
)

# The exact template /my-compact step 3 fills in. These headings are AUTHORED by
# the skill, which is what makes section handling a guarantee rather than a
# heuristic -- there is nothing to infer.
COMPACT_TEMPLATE = """PROJECT: Acme.Billing.OrdersApi
DATE: 2026-08-11
LABEL: Align OA publisher with the provisioned topology

## What we were working on
Aligning the OA order-status-notification publisher with the provisioned
OrderNotification Service Bus topology.

## Key decisions made
- Publisher reads AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey.
- Rejected re-rendering snapshots on read; chose the stored snapshot.

## Current state / progress
PR 481200 open, four commits pushed, 732 tests green.

## Open tasks / next steps
- Grep sweep for RuleFor(...).ForEach( inside a RuleSet block.

## Important files and locations
- src/Validators/OrderStatusNotificationRequestDtoValidator.cs
- Documentation/templates/rpiv-template.md

## Unresolved questions
- Does the topic need the -dev suffix in dev, or is the namespace enough?
- What is Vendra's retry cadence on a failed notification?
"""

def _fake(prefix: str, body: str) -> str:
    """Assemble a credential-shaped test value at RUNTIME.

    The complete literal must never appear in this file. GitHub secret
    scanning push protection rejects the push outright when it does -- which
    is exactly what happened writing these tests:

        Push blocked due to secrets:
        1. Stripe API Key found at ... tests/test_local_compress_lib.py:40
        2. Slack API Token found at ... tests/test_local_compress_lib.py:42

    Splitting the prefix from the body keeps the matchable pattern out of the
    committed source while the value assembled at runtime still has the real
    shape, which is what the filter under test actually sees. Every value here
    is fabricated; none is or ever was a live credential.
    """
    return prefix + body


# Real-shaped credentials, fabricated values, real prefixes. See _fake above
# for why these are assembled rather than written out.
CREDENTIALS = [
    (_fake("ghp_", "16C7e42F292c6912E7710c838347Ae178B4a"), "GitHub PAT"),
    (_fake("github_pat_", "11ABCDE0000fFgGhH"), "GitHub fine-grained PAT"),
    (_fake("sk_live_", "51H8xQ2eZvKYlo2CkIvxyzABC"), "Stripe live key"),
    (_fake("sk-ant-", "api03-AbCdEfGhIjKlMnOpQrSt"), "Anthropic key"),
    (_fake("xoxb-", "2401234567-abcdefghijklmnop"), "Slack bot token"),
    (_fake("glpat-", "ABC123xyz456DEF789"), "GitLab PAT"),
    (_fake("npm_", "aBcDeFgHiJkLmNoPqRsTuVwXyZ"), "npm token"),
    (_fake("dckr_pat_", "AbCdEfGhIjKlMnOp"), "Docker PAT"),
    (_fake("hf_", "AbCdEfGhIjKlMnOpQrStUvWx"), "HuggingFace token"),
    (_fake("AKIA", "IOSFODNN7EXAMPLE"), "AWS access key id"),
    (_fake("AIza", "SyD-abcdefghijklmnopqrstuvwxyz12"), "Google API key"),
    (_fake("eyJ", "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3jVmNHl0w5N"), "JWT"),
    # Azure Databricks PAT (issue #62): "dapi" + exactly 32 hex chars.
    # Uses a dedicated pattern rather than _CREDENTIAL_PREFIXES to avoid
    # false-positives on ordinary identifiers that start with "dapi".
    (_fake("dapi", "abcdef1234567890abcdef1234567890"), "Azure Databricks PAT"),
    # GCP OAuth2 refresh token (issue #62)
    (_fake("1//", "0fAbCdEfGhIjKlMnOpQrStUvWxYz-AbCd"), "GCP OAuth2 refresh token"),
]

# Identifiers that must survive the credential filter -- these are the whole
# point of the feature, and an entropy-based filter would have eaten them.
NOT_CREDENTIALS = [
    ("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0", "a 40-char commit SHA"),
    ("compute_file_hashes", "an ordinary snake_case function"),
    ("Acme.Billing.OrdersApi", "a dotted namespace"),
    ("src/Validators/Order.cs", "a path"),
    ("_EXACT_CMDS", "a module constant"),
    ("skip_if_under_chars", "a parameter that merely starts with 'sk'"),
    # issue #62: "dapi" is now matched only as dapi+32hex, NOT as a generic
    # prefix -- API handler names starting with "dapi" must not be redacted.
    ("dapi_request", "API handler name starting with 'dapi'"),
    ("dapi_handler", "another ordinary dapi-prefixed name"),
    # issue #62: a 40-char git-style hex hash that begins with "dapi" must NOT
    # match -- the right-boundary (?![A-Za-z0-9_]) prevents the {32} quantifier
    # from consuming the first 32 chars of a longer hex string.
    ("dapi" + "a" * 40, "a 40-char hex identifier that happens to start with dapi"),
]


# Asks for the whole thing -> skip classification, nothing to filter for.
NON_SELECTIVE = [
    (DEFAULT_FOCUS, "the default every caller gets when passing no focus at all"),
    ("Summarize the key information.", "the default's first clause on its own"),
    ("summarize this", ""),
    ("Summarize This", "case-insensitive"),
    ("  summarize this  ", "surrounding whitespace"),
    ("give me the gist", ""),
    ("Please summarize this.", "leading filler and trailing punctuation stripped"),
    ("Just please summarize this.", "doubled filler: just + please (issue #42)"),
    ("Please just summarize this.", "doubled filler: please + just (issue #42)"),
    ("Can you please just summarize this.", "tripled filler: can you + please + just (issue #42)"),
    ("what happened", ""),
    ("did it fail?", ""),
    ("pass or fail", ""),
    ("", "empty focus -- cannot select for anything"),
    ("   ", "whitespace-only focus"),
]

# Names a target -> classification must still run and still be able to filter.
SELECTIVE = [
    ("chocolate cake recipes", "the canonical absent-topic case"),
    ("how do I enable the savings tracker", "present in the docs, must not bypass"),
    ("how auth works", ""),
    ("the retry logic", ""),
    ("which tests failed", "narrower than 'did it fail' -- names a subset"),
    ("connection string configuration", ""),
    # A generic phrase is almost always a PREFIX of a selective request. These
    # each contain an entry from _GENERIC_FOCUS_PHRASES while naming a target,
    # and were all wrongly bypassed while the check was substring-based.
    ("summarize the key information about auth", "contains 'summarize the key information'"),
    ("what happened with the retry logic", "contains 'what happened'"),
    ("did it fail to connect to redis", "contains 'did it fail'"),
    ("the gist of the auth module", "contains 'the gist'"),
    ("summarize this file's error handling", "contains 'summarize this'"),
]


class NonSelectiveFocus(unittest.TestCase):
    def test_non_selective(self):
        for focus, why in NON_SELECTIVE:
            with self.subTest(focus=focus):
                self.assertTrue(
                    _looks_non_selective(focus),
                    f"{focus!r} asks for everything; classification should be skipped ({why})",
                )

    def test_selective(self):
        for focus, why in SELECTIVE:
            with self.subTest(focus=focus):
                self.assertFalse(
                    _looks_non_selective(focus),
                    f"{focus!r} names a target; classification must still filter ({why})",
                )

    def test_default_focus_is_non_selective(self):
        """Guards the specific regression: DEFAULT_FOCUS reads like an
        instruction, but its 'preserve anything that looks like an error'
        clause reads to a strict classifier as a selection criterion, which
        made it reject every chunk of any output without an error in it."""
        self.assertTrue(_looks_non_selective(DEFAULT_FOCUS))


class NormalizeFocusDoubledFiller(unittest.TestCase):
    """Issue #42: _normalize_focus's filler-stripping regex used to be
    anchored at ^ but applied only ONCE (a single re.sub, not a loop or a
    repeating group), so a doubled politeness/imperative prefix only lost
    its first layer -- 'Just please summarize this.' normalized to
    'please summarize this' instead of 'summarize this', which then failed
    to match DEFAULT_FOCUS/_GENERIC_FOCUS_PHRASES in _looks_non_selective
    and wrongly let the per-chunk classifier run against the leftover
    fragment as a selection target."""

    def test_single_filler_word_still_stripped(self):
        self.assertEqual(_normalize_focus("Please summarize this."), "summarize this")
        self.assertEqual(_normalize_focus("just summarize this"), "summarize this")

    def test_doubled_filler_fully_stripped(self):
        self.assertEqual(_normalize_focus("Just please summarize this."), "summarize this")
        self.assertEqual(_normalize_focus("Please just summarize this."), "summarize this")

    def test_tripled_filler_fully_stripped(self):
        self.assertEqual(
            _normalize_focus("Can you please just summarize this."), "summarize this"
        )

    def test_no_filler_unaffected(self):
        self.assertEqual(_normalize_focus("summarize this"), "summarize this")


class PositionalFocus(unittest.TestCase):
    """_looks_positional is the older sibling of _looks_non_selective and was
    previously untested. The two are independent: a positional focus IS
    selective (it names a part), so neither predicate should imply the other."""

    def test_positional(self):
        for focus in ["summarize the lead section", "the introduction", "the abstract",
                      "first paragraph", "tl;dr", "the beginning of the doc"]:
            with self.subTest(focus=focus):
                self.assertTrue(_looks_positional(focus))

    def test_not_positional(self):
        for focus in ["how auth works", "chocolate cake recipes", DEFAULT_FOCUS]:
            with self.subTest(focus=focus):
                self.assertFalse(_looks_positional(focus))

    def test_positional_focuses_are_still_selective(self):
        """A positional ask names a specific part, so it must NOT take the
        non-selective bypass -- it has its own auto-truncation path instead."""
        for focus in ["summarize the lead section", "the introduction", "the abstract"]:
            with self.subTest(focus=focus):
                self.assertFalse(_looks_non_selective(focus))


class FindFirstHeadingBoundary(unittest.TestCase):
    """Issue #28: _find_first_heading_boundary only checked the line
    IMMEDIATELY adjacent to a candidate heading for "is this prose" --
    missing the standard Markdown convention of a blank line on either side
    of a heading (including this repo's own README), which silently fell
    back to the fixed ~4000-char window far more often than intended."""

    def _lead(self, n=350):
        # Default 350: > 80 chars clears the prose-line length check, and
        # >= min_offset (300) means the heading line that follows is
        # actually reached with enough accumulated offset to be considered
        # at all -- a shorter lead never triggers the check in the first
        # place, regardless of blank lines.
        return "A" * n + "."

    def test_blank_line_delimited_heading_is_now_detected(self):
        # The bug case: a blank line on both sides of the heading, exactly
        # how a real Markdown document is written.
        lead = self._lead()
        rest = self._lead(95)
        text = f"{lead}\n\nSection Two\n\n{rest}"
        boundary = _find_first_heading_boundary(text)
        self.assertIsNotNone(boundary, "a blank-line-delimited heading must still be found")
        self.assertEqual(text[:boundary], lead + "\n\n")

    def test_no_blank_line_heading_still_detected(self):
        # Regression guard: the already-working case (heading directly
        # adjacent to prose, no blank line) must keep working identically.
        lead = self._lead()
        rest = self._lead(95)
        text = f"{lead}\nSection Two\n{rest}"
        boundary = _find_first_heading_boundary(text)
        self.assertIsNotNone(boundary)
        self.assertEqual(text[:boundary], lead + "\n")

    def test_multiple_blank_lines_around_heading_still_detected(self):
        lead = self._lead()
        rest = self._lead(95)
        text = f"{lead}\n\n\n\nSection Two\n\n\n{rest}"
        boundary = _find_first_heading_boundary(text)
        self.assertIsNotNone(boundary, "several consecutive blank lines must still be walked past")

    def test_blank_line_on_only_one_side_is_still_detected(self):
        # Confirms the prev/next walks are independent -- a blank line on
        # just one side must not break detection either.
        lead = self._lead()
        rest = self._lead(95)
        with self.subTest("blank before only"):
            text = f"{lead}\n\nSection Two\n{rest}"
            self.assertIsNotNone(_find_first_heading_boundary(text))
        with self.subTest("blank after only"):
            text = f"{lead}\nSection Two\n\n{rest}"
            self.assertIsNotNone(_find_first_heading_boundary(text))

    def test_heading_at_document_start_is_not_a_false_boundary(self):
        # i == 0 -- no previous line exists at all, blank or otherwise.
        # Must behave the same as before the fix: no boundary, since a
        # document-initial heading isn't a real split point in the text.
        rest = self._lead(95)
        text = f"Title\n\n{rest}"
        self.assertIsNone(_find_first_heading_boundary(text))

    def test_plain_text_with_no_heading_structure_returns_none(self):
        # Regression guard: this heuristic must remain a no-op when there's
        # genuinely nothing heading-like to find (e.g. a log file).
        text = "\n".join(f"line {i} of plain unstructured text content here" for i in range(200))
        self.assertIsNone(_find_first_heading_boundary(text))

    def test_finds_a_real_boundary_in_this_repos_own_readme(self):
        # The exact scenario the issue names: this repo's own README uses
        # standard blank-line-delimited headings throughout.
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"
        )
        with open(readme_path, encoding="utf-8") as f:
            text = f.read()
        # max_offset=len(text): the first heading this heuristic accepts in the
        # README sits after the Files table, so the default 20_000-char window
        # would tie this test to the table's length rather than to the bug.
        boundary = _find_first_heading_boundary(text, max_offset=len(text))
        self.assertIsNotNone(boundary, "the real README has headings; None means the bug regressed")


# Tokens whose whole value is being byte-exact. Each entry is (text, token).
MUST_DETECT = [
    ("see src/utils/auth.ts for the guard", "src/utils/auth.ts"),
    ("path is Documentation/templates/rpiv-template.md", "Documentation/templates/rpiv-template.md"),
    # The identifier from the real failure: the ':Options:' segment was the
    # part the compression dropped, so the whole key has to be one token.
    ("reads AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey",
     "AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey"),
    ("project Acme.Billing.OrdersApi builds", "Acme.Billing.OrdersApi"),
    ("fixed OrderStatusNotificationRequestDtoValidator today",
     "OrderStatusNotificationRequestDtoValidator"),
    # 2-segment CamelCase, common in .NET-style names -- regression case for
    # the {2,}->{1,} fix (issue #32): the old 3-hump minimum missed these.
    ("restart UserService now", "UserService"),
    ("call GetUser first", "GetUser"),
    ("call compute_file_hashes on the repo", "compute_file_hashes"),
    ("the exempt list lives in _EXACT_CMDS", "_EXACT_CMDS"),
    ("run it with --dry-run first", "--dry-run"),
    ("PR 481200 is open", "481200"),
    # Found in PR review: single-letter flags matched neither the regex (which
    # required a character after the letter) nor the length floor (which was 4
    # for everything). Dropping the -o changes what the command returns.
    ("az group show -o json", "-o"),
    ("python x.py -h", "-h"),
    # Also found in PR review: the path pattern required a forward slash before
    # the filename, so Windows-style paths went unverified -- on a repo that
    # ships docs/windows-setup.md.
    (r"edit src\Validators\Invoice.cs now", r"src\Validators\Invoice.cs"),
    # An absolute Windows path must keep its drive letter, or the repair step
    # restores a path that no longer resolves.
    (r"open C:\repo\src\Program.cs", r"C:\repo\src\Program.cs"),
]

# Ordinary prose that must NOT be collected, or every compaction grows a tail
# of noise. This is the direction that silently makes the feature useless.
MUST_IGNORE = [
    ("a well-known trade-off applies here", "well-known"),
    ("the compression is context-lossless in theory", "context-lossless"),
    ("we ran 732 tests", "732"),
    ("e.g. this one", "e.g."),
    # Hyphenation must not read as a flag now that single-letter flags match.
    # The (?<!\w) lookbehind is the only thing standing between "-o" support
    # and every hyphenated word in the summary being treated as an identifier.
    ("a state-of-the-art design", "-of"),
    ("encoded as UTF-8 text", "-8"),
    ("the trade-off is real", "-off"),
]


class IdentifierDetection(unittest.TestCase):
    """identifier_tokens must find grep-able strings and skip prose.

    Why this matters in both directions: a missed identifier is one token the
    repair step won't restore, while a false positive appends prose to EVERY
    compaction -- so the noise case is the one that quietly destroys the value
    of the feature, not the miss.
    """

    def test_detects_identifier_shapes(self):
        for text, token in MUST_DETECT:
            with self.subTest(token=token):
                self.assertIn(token, identifier_tokens(text))

    def test_ignores_prose(self):
        for text, token in MUST_IGNORE:
            with self.subTest(token=token):
                self.assertNotIn(token, identifier_tokens(text))

    def test_first_appearance_order_without_duplicates(self):
        text = "src/a/b.cs then Foo.Bar.Baz then src/a/b.cs again"
        self.assertEqual(identifier_tokens(text), ["src/a/b.cs", "Foo.Bar.Baz"])

    def test_trailing_punctuation_is_not_part_of_the_token(self):
        self.assertIn("src/a/b.cs", identifier_tokens("edit src/a/b.cs, then stop."))


class IdentifierRepair(unittest.TestCase):
    ORIGINAL = (
        "Publisher reads AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey.\n"
        "Fixed OrderStatusNotificationRequestDtoValidator in src/Validators/Order.cs.\n"
        "PR 481200 open."
    )

    def test_reports_only_what_is_missing(self):
        # The identifier is present but reworded around -- and the config key
        # has the ':Options:' segment dropped, the exact real-world corruption.
        summary = (
            "Publisher reads AppSecretKey:OrderNotificationServiceBus:ConnectionKey. "
            "Fixed OrderStatusNotificationRequestDtoValidator."
        )
        missing = missing_identifiers(self.ORIGINAL, summary)
        self.assertIn(
            "AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey",
            missing,
            "a mangled identifier must count as missing, not as present",
        )
        self.assertIn("src/Validators/Order.cs", missing)
        self.assertNotIn("OrderStatusNotificationRequestDtoValidator", missing)

    def test_backticks_around_a_token_still_count_as_present(self):
        summary = "reads `AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey`"
        self.assertNotIn(
            "AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey",
            missing_identifiers(self.ORIGINAL, summary),
        )

    def test_append_is_a_noop_when_nothing_was_dropped(self):
        self.assertEqual(
            append_missing_identifiers(self.ORIGINAL, self.ORIGINAL),
            self.ORIGINAL,
            "a summary retaining every identifier must not grow a block",
        )

    def test_append_restores_dropped_tokens_verbatim(self):
        summary = "Publisher updated. Validator fixed."
        out = append_missing_identifiers(self.ORIGINAL, summary)
        self.assertTrue(out.startswith(summary), "the summary itself must be preserved")
        self.assertIn("VERBATIM IDENTIFIERS DROPPED BY COMPRESSION", out)
        self.assertIn(
            "AppSecretKey:Options:OrderNotificationServiceBus:ConnectionKey", out
        )
        self.assertIn("src/Validators/Order.cs", out)

    def test_limit_truncates_and_says_so(self):
        original = " ".join(f"src/gen/File{i}.cs" for i in range(50))
        out = append_missing_identifiers(original, "nothing kept", limit=5)
        self.assertIn("more not shown", out, "truncation must be reported, not silent")
        self.assertEqual(out.count("src/gen/File"), 5)


class CredentialsAreNeverRestored(unittest.TestCase):
    """Found in PR review, reproduced before fixing.

    Several identifier patterns match real secrets -- snake_case accepts
    ghp_<...> and sk_live_<...>, dotted accepts a three-part JWT. The repair
    step therefore RESTORED credentials the summarizer had dropped, and
    /my-compact enables it unconditionally then persists to Qdrant. Net effect:
    a transient secret in a session became a durable one in a vector store, and
    the summarizer's accidental good behavior (omitting it) was undone.

    These tests pin both directions, because a credential filter that eats
    commit SHAs would gut the feature it protects.
    """

    def test_credential_shapes_are_recognized(self):
        for secret, label in CREDENTIALS:
            with self.subTest(label=label):
                self.assertTrue(looks_like_credential(secret), label)

    def test_ordinary_identifiers_are_not_mistaken_for_credentials(self):
        for token, label in NOT_CREDENTIALS:
            with self.subTest(label=label):
                self.assertFalse(looks_like_credential(token), label)

    def test_credentials_are_not_collected(self):
        for secret, label in CREDENTIALS:
            with self.subTest(label=label):
                self.assertEqual(
                    [t for t in identifier_tokens(f"token is {secret} ok") if t in secret],
                    [],
                    f"{label} must never be collected",
                )

    def test_guard_does_not_reinject_a_secret_the_model_dropped(self):
        secret = _fake("ghp_", "16C7e42F292c6912E7710c838347Ae178B4a")
        jwt = _fake("eyJ", "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3jVmNHl0w5N")
        original = f"Auth uses {secret} against src/Auth/Client.cs. Bearer {jwt}"
        summary = "Auth uses a GitHub token against src/Auth/Client.cs with a bearer JWT."
        self.assertNotIn(secret, summary, "precondition: the model dropped the secret")
        out = append_missing_identifiers(original, summary)
        self.assertNotIn(secret, out, "the guard must not put the secret back")
        self.assertNotIn(jwt, out, "the guard must not put the JWT back")
        # The real identifier alongside it must still be handled normally.
        self.assertIn("src/Auth/Client.cs", out)

    def test_identifier_repair_never_reappends_a_credential(self):
        original = f"token {_fake('ghp_', '16C7e42F292c6912E7710c838347Ae178B4a')} and src/A.cs"
        out = append_missing_identifiers(original, "nothing kept")
        self.assertNotIn("ghp_", out, "the secret must not be re-appended")
        self.assertIn("src/A.cs", out, "the real identifier still is")
        self.assertEqual(credential_token_count(original), 1)

    def test_embedded_credentials_are_still_caught(self):
        """Found in a second review round, reproduced first: 6 of these 8 forms
        leaked the FULL secret, because detection was anchored to the start of a
        whitespace-delimited word and a secret is usually not a standalone word.
        credential_token_count() also returned 0 for those, so the 'withheld'
        disclosure didn't fire either -- silently wrong, not visibly incomplete.
        """
        secret = _fake("ghp_", "16C7e42F292c6912E7710c838347Ae178B4a")
        forms = {
            "bare": f"the token is {secret} ok",
            "env assignment": f"GITHUB_TOKEN={secret}",
            "export line": f"export GITHUB_TOKEN={secret}",
            "json": f'{{"token":"{secret}"}}',
            "yaml": f"token: {secret}",
            "cli flag": f"gh auth login --with-token={secret}",
            "url basic auth": f"https://x:{secret}@github.com/o/r.git",
            "quoted": f'TOKEN="{secret}"',
        }
        for label, text in forms.items():
            with self.subTest(form=label):
                collected = identifier_tokens(text)
                self.assertEqual(
                    [t for t in collected if secret in t or (len(t) > 8 and t in secret)],
                    [],
                    f"{label}: no part of the secret may be collected",
                )
                self.assertNotIn(secret, append_missing_identifiers(text, "summary"))
                self.assertGreater(
                    credential_token_count(text), 0,
                    f"{label}: withholding must be disclosed, not silent",
                )

    def test_pem_private_key_body_is_excluded_not_just_the_header(self):
        """Found in a fourth review round.

        Matching only the PEM header left the key body outside the credential
        span, where it could be collected as an identifier and restored
        verbatim. It is probabilistic, not certain, which is why a typical
        sample looks clean: the CamelCase pattern is \\b-anchored and base64 is
        one long word, so a match can only begin at a line's first character.
        Measured at ~1.5% of random 64-char base64 lines, which is ~31% for a
        25-line 2048-bit key and ~53% for a 50-line 4096-bit key. The body line
        below is one that does begin with a CamelCase-shaped run.
        """
        body = "Ab3Cd4Ef5GhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMnOpQrStUvWx"
        pem = f"-----BEGIN OPENSSH PRIVATE KEY-----\n{body}\n-----END OPENSSH PRIVATE KEY-----"
        text = f"deploy key:\n{pem}\nsee src/deploy.sh"
        self.assertEqual(
            [t for t in identifier_tokens(text) if len(t) > 12 and t in body],
            [],
            "no part of the key body may be collected",
        )
        self.assertNotIn(body, append_missing_identifiers(text, "a summary"))
        self.assertGreater(credential_token_count(text), 0)
        # Real identifiers outside the block are unaffected.
        self.assertIn("src/deploy.sh", identifier_tokens(text))

    def test_unterminated_pem_block_still_withholds_the_body(self):
        """A truncated summary can carry a header with no footer. Everything
        after it is treated as key material -- over-suppressing rather than
        leaking, since an unverified identifier is a gap and leaked key
        material is a breach."""
        body = "Ab3Cd4Ef5GhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMnOpQrStUvWx"
        text = f"key follows\n-----BEGIN RSA PRIVATE KEY-----\n{body}"
        self.assertNotIn(body, append_missing_identifiers(text, "a summary"))

    def test_identifiers_before_a_pem_block_are_still_collected(self):
        body = "Ab3Cd4Ef5GhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMnOpQrStUvWx"
        text = f"edit src/deploy.sh and compute_file_hashes first\n-----BEGIN RSA PRIVATE KEY-----\n{body}"
        collected = identifier_tokens(text)
        self.assertIn("src/deploy.sh", collected)
        self.assertIn("compute_file_hashes", collected)

    def test_short_prefix_does_not_swallow_ordinary_words(self):
        """The "sk-" prefix is short enough to appear inside normal words. If it
        matched there, the surrounding text would be treated as a credential
        span and its real identifiers silently suppressed."""
        text = "a task-oriented rewrite of src/Handlers/Task.cs in compute_file_hashes"
        self.assertEqual(credential_token_count(text), 0)
        collected = identifier_tokens(text)
        self.assertIn("src/Handlers/Task.cs", collected)
        self.assertIn("compute_file_hashes", collected)

    def test_no_credential_note_when_source_has_none(self):
        original = "just src/A.cs and compute_file_hashes here"
        self.assertEqual(credential_token_count(original), 0)
        self.assertNotIn("withheld", append_missing_identifiers(original, "nothing kept"))


class CredentialRedaction(unittest.TestCase):
    """Found in a second review round on the section work, reproduced first.

    Excluding credentials from identifier COLLECTION only governs what gets
    re-appended. Once verbatim passthrough existed, content could reach the
    output by being copied rather than regenerated -- the preamble, a bullet
    section, a failed-call fallback -- so a secret was deterministically stored,
    and the artifact printed "withheld and NOT restored" right beside it. A
    false assurance is worse than none.
    """

    SECRET = _fake("ghp_", "16C7e42F292c6912E7710c838347Ae178B4a")

    def test_redacts_and_counts(self):
        out, n = redact_credentials(f"a {self.SECRET} b {self.SECRET} c")
        self.assertEqual(n, 2)
        self.assertNotIn("ghp_", out)
        self.assertEqual(out.count("[REDACTED-CREDENTIAL]"), 2)
        self.assertTrue(out.startswith("a ") and out.endswith(" c"),
                        "surrounding text is preserved")

    def test_noop_and_no_note_when_clean(self):
        text = "just src/A.cs and compute_file_hashes"
        self.assertEqual(redact_credentials(text), (text, 0))
        self.assertEqual(redact_and_disclose(text), text)
        self.assertNotIn("redacted", redact_and_disclose(text))

    def test_disclosure_counts_actual_replacements(self):
        out = redact_and_disclose(f"x {self.SECRET} y")
        self.assertIn("1 credential-shaped value(s) redacted", out)
        self.assertNotIn("ghp_", out)

    def test_redaction_survives_embedded_forms(self):
        for form in (f"GITHUB_TOKEN={self.SECRET}",
                     f'{{"token":"{self.SECRET}"}}',
                     f"https://x:{self.SECRET}@github.com/o/r.git"):
            with self.subTest(form=form):
                out, n = redact_credentials(form)
                self.assertEqual(n, 1)
                self.assertNotIn("ghp_", out)


class SectionSplitting(unittest.TestCase):
    """Section boundaries come from headings the skill itself wrote, so these
    are exact expectations rather than tolerances."""

    def test_splits_the_compact_template_into_its_six_sections(self):
        preamble, sections = split_sections(COMPACT_TEMPLATE)
        self.assertIn("PROJECT: Acme.Billing.OrdersApi", preamble)
        self.assertIn("LABEL:", preamble, "the label line must stay in the preamble")
        self.assertEqual(
            [h for h, _ in sections],
            [
                "## What we were working on",
                "## Key decisions made",
                "## Current state / progress",
                "## Open tasks / next steps",
                "## Important files and locations",
                "## Unresolved questions",
            ],
        )
        bodies = dict(sections)
        self.assertIn("Vendra", bodies["## Unresolved questions"])
        self.assertIn("rpiv-template.md", bodies["## Important files and locations"])

    def test_unstructured_text_reports_no_sections(self):
        """Signals the caller to fall back to whole-text compression rather than
        being special-cased at every call site."""
        text = "just a paragraph of prose with no headings at all"
        preamble, sections = split_sections(text)
        self.assertEqual(sections, [])
        self.assertEqual(preamble, text)

    def test_headings_inside_a_fenced_block_are_not_boundaries(self):
        text = (
            "## Real\nbody\n\n```md\n## Not a section\nsample\n```\n\n## Also Real\nmore\n"
        )
        self.assertEqual(
            [h for h, _ in split_sections(text)[1]], ["## Real", "## Also Real"]
        )

    def test_single_hash_is_not_a_section_boundary(self):
        self.assertEqual(split_sections("# Title\nbody\n## Real\nmore")[1],
                         [("## Real", "more")])

    def test_exactness_critical_sections_are_marked_verbatim(self):
        for heading in ("## Important files and locations", "## Unresolved questions",
                        "## Open tasks / next steps"):
            with self.subTest(heading=heading):
                self.assertTrue(section_is_verbatim(heading))

    def test_prose_bodies_are_marked_compressible(self):
        prose = "A paragraph of narrative text describing what happened.\nAnd a second line."
        for heading in ("## What we were working on", "## Current state / progress"):
            with self.subTest(heading=heading):
                self.assertFalse(section_is_verbatim(heading, prose))

    def test_off_template_bullet_section_is_verbatim_by_structure(self):
        """Heading names alone were not enough: "## Environment and workflow
        notes" is not in the template, was therefore compressed, and the model
        dropped "ADO has a 4000-char cap" from inside it -- the very section the
        real incident lost. A bullet list is a set of discrete facts, so
        "shorter" can only mean merging or dropping them."""
        body = ("- No CLAUDE.md exists in this repo.\n"
                "- ADO has a 4000-char cap on PR description fields.\n"
                "- Bump appVersion in Chart.yaml on every deploy.")
        self.assertTrue(section_is_verbatim("## Environment and workflow notes", body))
        self.assertTrue(section_is_verbatim("## Some Heading Nobody Anticipated", body))

    def test_numbered_lists_count_as_lists(self):
        self.assertTrue(section_is_verbatim("## Steps", "1. do this\n2. then this"))

    def test_a_paragraph_mentioning_a_dash_is_not_a_list(self):
        body = "We chose the stored snapshot - re-rendering broke reproducibility."
        self.assertFalse(section_is_verbatim("## Key decisions made", body))


class DeriveCompactLabel(unittest.TestCase):
    """Issue #72: derive_compact_label extracts a short, deterministic label
    from a structured compact summary without involving a model call."""

    def test_extracts_first_sentence_from_working_on_section(self):
        """Happy path: the template's 'What we were working on' section exists
        and has a sentence-terminated first sentence."""
        result = derive_compact_label(COMPACT_TEMPLATE)
        # COMPACT_TEMPLATE's body starts: "Aligning the OA order-status-notification
        # publisher with the provisioned\nOrderNotification Service Bus topology."
        # First sentence ends at the period after "topology".
        self.assertIn("Aligning", result)
        self.assertNotIn("Key decisions", result)

    def test_wrapped_sentence_is_not_truncated_at_first_line_break(self):
        """Regression: a sentence that word-wraps to a second line must not be
        cut at the first physical newline (Copilot review finding on PR #159).

        The sentence terminator is on the second physical line, so the result
        must contain content from that line (not just the first line's text).
        The 80-char cap may still truncate the full sentence; what matters is
        that joined content from the second line appears in the output.
        """
        text = (
            "## What we were working on\n"
            "Aligning the OA order publisher with the provisioned\n"
            "OrderNotification Service Bus topology. Other context follows.\n"
        )
        result = derive_compact_label(text)
        # "provisioned" ends the first physical line -- without the fix the
        # result would stop there. With the fix the joined text flows into
        # the second line ("OrderNotification..."), so the result must contain
        # content that only appears AFTER the first-line break.
        # "provisioned" alone does appear on line 1; "OrderNotification" is only
        # reachable by joining across lines.
        self.assertIn("OrderNotification", result)
        self.assertNotIn("Other context", result)

    def test_result_does_not_exceed_80_chars(self):
        long_body = "A" * 200
        text = f"## What we were working on\n{long_body}"
        result = derive_compact_label(text)
        self.assertLessEqual(len(result), 80)

    def test_sentence_split_stops_at_period(self):
        text = "## What we were working on\nFixed the auth bug. Then moved on to caching."
        result = derive_compact_label(text)
        self.assertEqual(result, "Fixed the auth bug")

    def test_sentence_split_stops_at_question_mark(self):
        text = "## What we were working on\nIs this working? Yes it is."
        result = derive_compact_label(text)
        self.assertEqual(result, "Is this working")

    def test_sentence_split_stops_at_exclamation(self):
        text = "## What we were working on\nShipped the feature! Continuing with cleanup."
        result = derive_compact_label(text)
        self.assertEqual(result, "Shipped the feature")

    def test_no_sentence_terminator_uses_whole_first_line(self):
        text = "## What we were working on\nRefactoring the auth middleware layer"
        result = derive_compact_label(text)
        self.assertEqual(result, "Refactoring the auth middleware layer")

    def test_returns_empty_string_when_no_content(self):
        result = derive_compact_label("")
        self.assertEqual(result, "")

    def test_falls_back_gracefully_when_section_absent(self):
        """When the structured section is missing, derive_compact_label should
        still return something non-empty rather than '' whenever there is
        preamble text -- the preamble path is a best-effort fallback."""
        text = "Aligning the OA publisher. Some more details."
        result = derive_compact_label(text)
        # No sections present, so falls back to preamble's first sentence
        self.assertEqual(result, "Aligning the OA publisher")

    def test_returns_empty_for_blank_string(self):
        result = derive_compact_label("   \n  ")
        self.assertEqual(result, "")

    def test_ignores_other_sections(self):
        """Label must come from 'What we were working on', not later sections."""
        text = (
            "## Key decisions made\nChose Redis over Memcached.\n\n"
            "## What we were working on\nMigrating the cache layer to Redis."
        )
        result = derive_compact_label(text)
        self.assertIn("Redis", result)
        self.assertNotIn("Chose", result)

    def test_strips_leading_trailing_whitespace(self):
        text = "## What we were working on\n   Trimmed label text.   "
        result = derive_compact_label(text)
        self.assertEqual(result, "Trimmed label text")


class SplitProseBulletRuns(unittest.TestCase):
    """Issue #61: split_prose_bullet_runs and section_has_mixed_runs.

    The feature compresses only the prose portions of a mixed section, keeping
    bullet runs verbatim -- recovering compression ratio without weakening the
    'a list of discrete facts is never compressed' guarantee.
    """

    def test_pure_prose_is_one_run(self):
        body = "A paragraph of narrative text.\nAnd a second prose line."
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 1)
        is_bullet, text = runs[0]
        self.assertFalse(is_bullet)
        self.assertIn("narrative", text)

    def test_pure_bullets_is_one_run(self):
        body = "- fact one\n- fact two\n- fact three"
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 1)
        is_bullet, text = runs[0]
        self.assertTrue(is_bullet)
        self.assertIn("fact one", text)

    def test_numbered_bullets_is_one_run(self):
        body = "1. step one\n2. step two"
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0][0])

    def test_mixed_prose_then_bullets(self):
        body = "Background context here.\n- fact A\n- fact B"
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 2)
        self.assertFalse(runs[0][0], "first run should be prose")
        self.assertIn("Background", runs[0][1])
        self.assertTrue(runs[1][0], "second run should be bullets")
        self.assertIn("fact A", runs[1][1])
        self.assertIn("fact B", runs[1][1])

    def test_mixed_bullets_then_prose(self):
        body = "- fact A\n- fact B\nContext after the list."
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 2)
        self.assertTrue(runs[0][0], "first run should be bullets")
        self.assertFalse(runs[1][0], "second run should be prose")
        self.assertIn("Context", runs[1][1])

    def test_prose_bullets_prose_three_runs(self):
        body = "Intro prose line.\n- fact one\n- fact two\nTrailing prose line."
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 3)
        self.assertFalse(runs[0][0])
        self.assertTrue(runs[1][0])
        self.assertFalse(runs[2][0])

    def test_blank_lines_attach_to_following_run(self):
        # A blank line between a bullet run and a prose run should travel
        # with the prose run (attach-to-follower convention).
        body = "- fact A\n\nProse after blank."
        runs = split_prose_bullet_runs(body)
        self.assertEqual(len(runs), 2)
        self.assertTrue(runs[0][0], "bullets come first")
        self.assertFalse(runs[1][0], "prose comes second")
        # The blank line attaches to the prose run, not the bullet run.
        self.assertIn("\n", runs[1][1], "blank line should be in the prose run")
        self.assertIn("Prose after blank", runs[1][1])
        self.assertNotIn("\n", runs[0][1], "bullet run should not carry the blank")

    def test_empty_body_returns_single_prose_run(self):
        runs = split_prose_bullet_runs("")
        self.assertEqual(len(runs), 1)
        self.assertFalse(runs[0][0])

    def test_section_has_mixed_runs_pure_prose(self):
        body = "Just prose lines.\nAnother prose line."
        self.assertFalse(section_has_mixed_runs(body))

    def test_section_has_mixed_runs_pure_bullets(self):
        body = "- fact one\n- fact two"
        self.assertFalse(section_has_mixed_runs(body))

    def test_section_has_mixed_runs_mixed_body(self):
        body = "Intro prose.\n- fact one\n- fact two"
        self.assertTrue(section_has_mixed_runs(body))

    def test_section_has_mixed_runs_bullets_then_prose(self):
        body = "- fact\nTrailing prose."
        self.assertTrue(section_has_mixed_runs(body))

    def test_all_content_present_in_reconstructed_runs(self):
        """Round-trip: joining all run texts should recover all non-blank content."""
        body = "Intro prose.\n- fact A\n- fact B\nConclusion prose."
        runs = split_prose_bullet_runs(body)
        joined = "\n".join(text for _, text in runs)
        for line in body.splitlines():
            if line.strip():
                self.assertIn(line, joined, f"line {line!r} must survive the split-join")

    def test_leading_blanks_preserved_in_prose_run_text(self):
        """Blank lines that split_prose_bullet_runs attaches to a prose run
        must actually appear at the start of the run's text, so the
        _compress_each_section leading_blanks extraction below works correctly:
        leading_blanks = run_text[: len(run_text) - len(run_text.lstrip('\\n'))]
        """
        body = "- fact A\n\nProse after blank."
        runs = split_prose_bullet_runs(body)
        # The prose run (second) must start with the blank line.
        _, prose_text = runs[1]
        self.assertTrue(
            prose_text.startswith("\n"),
            "blank line must be the first character of the prose run text",
        )
        # The leading_blanks extraction used in _compress_each_section must
        # recover exactly that blank prefix.
        leading_blanks = prose_text[: len(prose_text) - len(prose_text.lstrip("\n"))]
        self.assertEqual(leading_blanks, "\n")
        # And the rest (stripped content) must be the prose line itself.
        self.assertEqual(prose_text.strip(), "Prose after blank.")


class SectionsAreNeverSilentlyDropped(unittest.TestCase):
    """The reported failure: three ENTIRE sections vanished from a real handoff
    while the result still read as complete. A missing heading has to be a
    detectable condition, not something a reader has to notice."""

    def test_detects_dropped_sections(self):
        # Exactly the shape of the real incident: prose kept, whole sections gone.
        summary = (
            "## What we were working on\nAligning the publisher.\n\n"
            "## Key decisions made\nUses the AppSecret key.\n\n"
            "## Current state / progress\nPR open.\n"
        )
        missing = missing_sections(COMPACT_TEMPLATE, summary)
        self.assertEqual(
            missing,
            ["## Open tasks / next steps",
             "## Important files and locations",
             "## Unresolved questions"],
        )

    def test_no_false_positives_when_everything_survived(self):
        self.assertEqual(missing_sections(COMPACT_TEMPLATE, COMPACT_TEMPLATE), [])

    def test_restores_dropped_sections_verbatim(self):
        summary = "## What we were working on\nAligning the publisher.\n"
        out = restore_missing_sections(COMPACT_TEMPLATE, summary)
        self.assertTrue(out.startswith(summary), "the summary itself is preserved")
        self.assertIn("SECTIONS DROPPED BY COMPRESSION", out)
        # The content that was actually lost in the real incident.
        self.assertIn("Vendra", out)
        self.assertIn("-dev suffix", out)
        self.assertIn("rpiv-template.md", out)
        self.assertIn("RuleFor(...).ForEach(", out)

    def test_restore_is_a_noop_when_nothing_was_dropped(self):
        self.assertEqual(
            restore_missing_sections(COMPACT_TEMPLATE, COMPACT_TEMPLATE),
            COMPACT_TEMPLATE,
        )


class CompressEarlyReturnDisclosure(unittest.TestCase):
    """Issue #24: compress()'s early-return (text under skip_if_under_chars)
    must disclose truncation rather than silently handing back a truncated
    slice indistinguishable from "the whole document was already this
    short." None of these reach LM Studio -- the early-return path returns
    before resolve_model/client/complete are ever touched -- so no stub
    injection is needed, unlike SectionCompressionEndToEnd below."""

    def setUp(self):
        # The _run_preserving tests DO stub resolve_model/complete/client --
        # clear the cache so prior test results don't bypass those stubs.
        clear_compression_cache()

    def tearDown(self):
        clear_compression_cache()

    def _run(self, text, focus="summarize this", max_chars=None):
        import asyncio
        return asyncio.run(compress(
            text, focus=focus, skip_if_under_chars=2000, max_chars=max_chars,
        ))

    def test_untruncated_short_text_returned_unchanged(self):
        # Regression guard: no max_chars, no positional focus -- the
        # original "returns the ORIGINAL text unchanged" contract must still
        # hold with no note attached.
        text = "short input, nothing truncated"
        self.assertEqual(self._run(text), text)

    def test_explicit_max_chars_below_threshold_discloses_truncation(self):
        text = "x" * 10_000
        result = self._run(text, max_chars=500)
        self.assertTrue(result.startswith("x" * 500))
        self.assertIn("max_chars=500", result)
        self.assertIn("truncated the source", result)

    def test_positional_focus_auto_truncation_discloses_truncation(self):
        # A genuine case where the auto-detected heading boundary actually
        # drops content: a long lead paragraph, a short heading-like line,
        # then more prose after it (see _find_first_heading_boundary). The
        # detected boundary (~352 chars) lands under skip_if_under_chars, so
        # this hits the early return with real truncation to disclose.
        lead = "A" * 350 + "."
        heading = "Section Two"
        rest = "B" * 100 + "."
        text = f"{lead}\n{heading}\n{rest}"
        result = self._run(text, focus="summarize the lead section")
        self.assertTrue(result.startswith(lead))
        self.assertNotIn(heading, result)  # confirms content was actually dropped
        self.assertIn("focus looked positional", result)
        self.assertIn("pass max_chars explicitly", result)

    def test_no_false_truncation_note_when_limit_is_selected_but_not_reached(self):
        # PR #85 review (Copilot): chars_limited/auto_truncated only mean a
        # limit was SELECTED, not that text[:max_chars] actually dropped
        # anything. A short positional input, or an explicit max_chars >=
        # len(text), must return completely unchanged -- no false claim of
        # truncation attached.
        short_positional = "A short lead section.\n"
        self.assertEqual(
            self._run(short_positional, focus="summarize the lead section"),
            short_positional,
        )

        text = "x" * 100
        self.assertEqual(self._run(text, max_chars=500), text)

    def test_disclosure_note_never_looks_like_real_compression(self):
        # compress_mcp_server.py's savings-tracker footer credits savings
        # only when the result starts with "[compressed" (see
        # _append_savings_footer) -- no compression ran here, so the
        # disclosure note must never trip that detection.
        text = "x" * 10_000
        result = self._run(text, max_chars=500)
        self.assertFalse(result.startswith("[compressed"))

    def _run_preserving(self, text, preserve_identifiers, max_chars=None, preserve_sections=False):
        import asyncio
        return asyncio.run(compress(
            text, focus="summarize this", skip_if_under_chars=2000,
            max_chars=max_chars, preserve_identifiers=preserve_identifiers,
            preserve_sections=preserve_sections,
        ))

    def test_credential_under_threshold_is_still_redacted_when_preserving_identifiers(self):
        # PR #135 review (Copilot): this early return happens BEFORE the
        # main preserve_identifiers block further down in compress() ever
        # runs, so a caller that turns preserve_identifiers on specifically
        # to get credential redaction (compress_command_output's new
        # default) was leaving every input shorter than skip_if_under_chars
        # -- the common case, since most command output is short --
        # completely unredacted.
        secret = "ghp_" + "a" * 36
        text = f"TOKEN={secret}"
        self.assertLess(len(text), 2000)  # confirms this hits the early return
        result = self._run_preserving(text, preserve_identifiers=True)
        self.assertNotIn(secret, result)
        self.assertIn("credential-shaped value(s) redacted", result)

    def test_clean_short_text_still_returned_unchanged_when_preserving_identifiers(self):
        # redact_and_disclose() is a no-op when nothing needs redacting --
        # confirms the fix doesn't attach a spurious note to ordinary short
        # output just because preserve_identifiers is on.
        text = "no secrets here, just a short status line"
        self.assertEqual(self._run_preserving(text, preserve_identifiers=True), text)

    def test_credential_under_threshold_is_not_redacted_when_not_preserving_identifiers(self):
        # Confirms the fix is still gated on preserve_identifiers, matching
        # this function's documented "only in artifact-producing modes"
        # tradeoff -- unrelated to issue #46/PR #135, just guarding against
        # accidentally making this unconditional.
        secret = "ghp_" + "a" * 36
        text = f"TOKEN={secret}"
        result = self._run_preserving(text, preserve_identifiers=False)
        self.assertEqual(result, text)

    def test_truncation_note_length_is_unaffected_by_redaction(self):
        # The truncate_note reports where the SOURCE was cut, which must
        # stay accurate even though redact_and_disclose() can change the
        # returned text's length (replacing a credential with a differently
        # sized placeholder, or appending a disclosure line).
        secret = "ghp_" + "a" * 36
        text = f"{secret} " + "x" * 500
        result = self._run_preserving(text, preserve_identifiers=True, max_chars=50)
        self.assertIn("max_chars=50 truncated the source to 50 chars", result)
        self.assertNotIn(secret, result)

    def test_credential_under_threshold_is_still_redacted_when_preserving_sections_only(self):
        # PR #135 review (Copilot), round 2: preserve_sections is also an
        # artifact-producing mode -- the NORMAL (over-threshold)
        # _compress_each_section path always redacts unconditionally,
        # regardless of preserve_identifiers -- so gating this early-return
        # redaction on preserve_identifiers alone left the exact same secret
        # safe above the threshold but leaking unchanged below it, with
        # preserve_sections=True and preserve_identifiers left at its
        # default False.
        secret = "ghp_" + "a" * 36
        text = f"## Heading\nTOKEN={secret}"
        self.assertLess(len(text), 2000)
        result = self._run_preserving(text, preserve_identifiers=False, preserve_sections=True)
        self.assertNotIn(secret, result)
        self.assertIn("credential-shaped value(s) redacted", result)


class SectionCompressionEndToEnd(unittest.TestCase):
    """Drives compress(preserve_sections=True) against a STUB summarizer, so the
    orchestration is tested without LM Studio: no network, deterministic, and
    able to simulate the exact misbehaviour that caused the incident."""

    def setUp(self):
        # Each test in this class injects a different stub for complete() --
        # a prior test's result cached under the same inputs would be served
        # back on the next test's first call, bypassing the stub entirely.
        # Clear the cache before each test so every _run() call is a real
        # compression pass through the currently-injected stub.
        clear_compression_cache()

    def tearDown(self):
        clear_compression_cache()

    def _run(self, fake_complete):
        import asyncio
        import local_compress_lib as L
        real_complete, real_resolve, real_client = L.complete, L.resolve_model, L.client
        L.complete = fake_complete
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        try:
            return asyncio.run(L.compress(
                COMPACT_TEMPLATE, focus="handoff", skip_if_under_chars=0,
                preserve_identifiers=True, preserve_sections=True,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real_complete, real_resolve, real_client

    def test_all_six_headings_survive_a_destructive_summarizer(self):
        out = self._run(lambda *a, **k: "gone")
        for heading in ("## What we were working on", "## Key decisions made",
                        "## Current state / progress", "## Open tasks / next steps",
                        "## Important files and locations", "## Unresolved questions"):
            with self.subTest(heading=heading):
                self.assertIn(heading, out)

    def test_verbatim_sections_are_never_sent_to_the_model(self):
        seen = []
        def fake(client, model, system, content):
            seen.append(content)
            return "compressed"
        self._run(fake)
        joined = "\n".join(seen)
        self.assertNotIn("Vendra", joined, "unresolved questions must not be sent")
        self.assertNotIn("rpiv-template.md", joined, "file list must not be sent")
        self.assertNotIn(
            "AppSecretKey:Options:", joined,
            "'Key decisions made' is a bullet list, so it is verbatim by structure",
        )
        self.assertIn("Aligning the OA", joined, "prose sections SHOULD be sent")
        self.assertEqual(
            len(seen), 2,
            "only the two genuinely prose sections; the other four are lists",
        )

    def test_exactness_critical_content_survives_verbatim(self):
        out = self._run(lambda *a, **k: "compressed")
        for needle in ("Vendra", "-dev suffix", "rpiv-template.md",
                       "RuleFor(...).ForEach(",
                       "OrderStatusNotificationRequestDtoValidator.cs"):
            with self.subTest(needle=needle):
                self.assertIn(needle, out)

    def test_a_failed_model_call_keeps_the_section_instead_of_losing_it(self):
        out = self._run(lambda *a, **k: None)
        self.assertIn("Aligning the OA", out, "original body kept when the call fails")
        self.assertIn("failed call", out, "and the fallback is disclosed")

    def test_not_relevant_verdict_does_not_delete_a_section(self):
        out = self._run(lambda *a, **k: "[NOT RELEVANT]")
        self.assertIn("Aligning the OA", out)

    def test_all_not_relevant_is_not_mistaken_for_an_outage(self):
        # PR #134 review (Copilot): a NOT RELEVANT verdict is a valid,
        # successful response from a reachable model, not a request failure.
        # Every attempted section here gets that verdict -- compressed_count
        # is 0, same as a real outage -- but LM Studio answered every request
        # just fine, so this must NOT be reported as unreachable.
        out = self._run(lambda *a, **k: "[NOT RELEVANT]")
        self.assertNotIn(
            "LM Studio appears unreachable", out,
            "a valid NOT RELEVANT verdict must never look like a request failure",
        )
        self.assertTrue(out.startswith("[compressed"), "still the normal success-shaped string")
        self.assertIn("Aligning the OA", out, "content is still fully preserved")

    def test_all_empty_responses_are_not_mistaken_for_an_outage(self):
        # PR #134 review (Copilot): complete() returns None ONLY on a genuine
        # request failure -- "" or whitespace-only text is a real, successful
        # response the model just happened to send nothing useful in. Every
        # attempted section here gets an empty response, so compressed_count
        # is 0 just like a real outage, but no request ever actually failed.
        for empty in ("", "   ", "\n\n"):
            with self.subTest(empty=repr(empty)):
                out = self._run(lambda *a, **k: empty)
                self.assertNotIn(
                    "LM Studio appears unreachable", out,
                    "an empty-but-live response must never look like a request failure",
                )
                self.assertTrue(out.startswith("[compressed"), "still the normal success-shaped string")
                self.assertIn("Aligning the OA", out, "content is still fully preserved")

    def test_total_outage_prepends_an_unambiguous_warning(self):
        # issue #43: both prose sections in COMPACT_TEMPLATE fail (the model
        # call itself failed, not a NOT RELEVANT verdict) and nothing else was
        # ever sent to the model (the other four sections are verbatim) --
        # this is a real total outage and must be distinguishable from a
        # normal run, without losing any content.
        out = self._run(lambda *a, **k: None)
        self.assertTrue(
            out.startswith("[LM Studio appears unreachable"),
            "a total outage must prepend an unambiguous warning, not just "
            "the usual success-shaped '[compressed ...]' string",
        )
        self.assertIn("Aligning the OA", out, "content is still fully preserved")
        self.assertNotIn("Error:", out, "the content-preserving fallback is not a hard error")

    def test_total_outage_warning_does_not_falsely_claim_content_is_unchanged(self):
        # PR #134 review (Copilot), round 3: redact_and_disclose() runs over
        # the WHOLE result regardless of why a section fell back to its
        # original body, so a credential-shaped value inside a fallback
        # section during a total outage is still replaced -- the warning
        # must not claim the content was returned "unchanged" when it wasn't.
        secret = _fake("ghp_", "16C7e42F292c6912E7710c838347Ae178B4a")
        import asyncio
        import local_compress_lib as L
        text = (
            f"## What we were working on\nDeploying with token {secret} in the pipeline.\n\n"
            "## Current state / progress\nAnother prose section with narrative content.\n"
        )
        real = (L.complete, L.resolve_model, L.client)
        L.complete = lambda *a, **k: None  # total outage: every attempted call fails
        L.resolve_model = lambda m, b: ("stub", None)
        L.client = lambda b: object()
        try:
            out = asyncio.run(L.compress(
                text, focus="handoff", skip_if_under_chars=0, preserve_sections=True,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real
        self.assertTrue(out.startswith("[LM Studio appears unreachable"))
        self.assertNotIn(secret, out, "the secret must still be redacted during an outage")
        self.assertNotIn(
            "unchanged", out,
            "the warning must not claim content is unchanged when redaction altered it",
        )
        self.assertIn("credential-shaped value(s) redacted", out)

    def test_partial_failure_does_not_trigger_the_total_outage_warning(self):
        # Only the FIRST attempted call fails; the second succeeds. This must
        # read as a normal (if imperfect) run, not a total outage.
        calls = {"n": 0}

        def fake(client, model, system, content):
            calls["n"] += 1
            return None if calls["n"] == 1 else "a real compressed summary"

        out = self._run(fake)
        self.assertNotIn("LM Studio appears unreachable", out)
        self.assertTrue(out.startswith("[compressed"), "still the normal success-shaped string")

    def test_all_sections_verbatim_is_not_mistaken_for_an_outage(self):
        # A document with zero prose sections never calls the model at all --
        # compressed_count and request_failed_count are both 0. That's a
        # normal, healthy run (nothing needed compressing), not a signal LM
        # Studio is down, so the warning must not fire here either.
        import asyncio
        import local_compress_lib as L

        text = (
            "## Key decisions made\n"
            "- Publisher reads a connection key from config.\n"
            "- Chose the stored snapshot over re-rendering.\n\n"
            "## Important files and locations\n"
            "- src/Validators/OrderStatusNotificationRequestDtoValidator.cs\n"
        )

        def fail_if_called(*a, **k):
            raise AssertionError("no section here is prose -- the model must never be called")

        real_complete, real_resolve, real_client = L.complete, L.resolve_model, L.client
        L.complete = fail_if_called
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        try:
            out = asyncio.run(L.compress(
                text, focus="handoff", skip_if_under_chars=0, preserve_sections=True,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real_complete, real_resolve, real_client

        self.assertNotIn("LM Studio appears unreachable", out)
        self.assertTrue(out.startswith("[compressed"))

    def test_preamble_metadata_is_preserved(self):
        out = self._run(lambda *a, **k: "compressed")
        self.assertIn("PROJECT: Acme.Billing.OrdersApi", out)
        self.assertIn("DATE: 2026-08-11", out)

    def test_reports_what_it_did(self):
        out = self._run(lambda *a, **k: "compressed")
        self.assertIn("6 section(s)", out)
        self.assertIn("2 compressed", out)
        self.assertIn("4 kept verbatim", out)

    def test_a_secret_in_a_verbatim_section_or_the_preamble_is_redacted(self):
        """The regression preserve_sections introduced: verbatim passthrough
        copies content instead of regenerating it, so a secret in a bullet list
        or in the LABEL line reached the artifact deterministically -- twice,
        in the reproduction -- while the output claimed it had been withheld."""
        secret = _fake("ghp_", "16C7e42F292c6912E7710c838347Ae178B4a")
        import asyncio
        import local_compress_lib as L
        summary = (
            f"PROJECT: Demo\nDATE: 2026-08-11\nLABEL: deploy uses {secret}\n\n"
            "## What we were working on\n"
            "A narrative paragraph long enough to be worth compressing.\n\n"
            "## Environment and workflow notes\n"
            "- No CLAUDE.md exists here.\n"
            f"- Deploy token for the pipeline is {secret}\n"
            "- Bump appVersion in Chart.yaml.\n\n"
            "## Unresolved questions\n- Does the topic need the -dev suffix?\n"
        )
        real = (L.complete, L.resolve_model, L.client)
        L.complete = lambda *a, **k: "compressed prose"
        L.resolve_model = lambda m, b: ("stub", None)
        L.client = lambda b: object()
        try:
            out = asyncio.run(L.compress(
                summary, focus="handoff", skip_if_under_chars=0,
                preserve_identifiers=True, preserve_sections=True,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real
        self.assertNotIn(secret, out, "no copy of the secret may survive")
        self.assertNotIn("ghp_", out)
        self.assertIn("credential-shaped value(s) redacted", out,
                      "and the artifact must say so accurately")
        self.assertNotIn("withheld and NOT restored", out,
                         "the old, false assurance must be gone")
        # The non-secret content of those same sections is untouched.
        self.assertIn("appVersion", out)
        self.assertIn("-dev suffix", out)


class WholeTextFallbackRedactsWithEitherPreservationFlag(unittest.TestCase):
    """Regression coverage for PR #135 review, round 4 (Copilot):
    preserve_sections=True falls through to the ORDINARY whole-text
    compression path (chunk_text -> classify/compress -> the
    `if preserve_identifiers:`-gated block) whenever
    _compress_each_section finds no headings to split on -- unstructured
    input is exactly this case. That whole-text path's redaction used to be
    gated on preserve_identifiers alone, so preserve_sections=True with
    preserve_identifiers left at its default False let a model-echoed
    credential straight through on ANY unstructured input over
    skip_if_under_chars, despite preserve_sections being just as much an
    artifact-producing mode as preserve_identifiers.
    """

    def setUp(self):
        # Each test uses a different stub -- clear the cache so a prior test's
        # result isn't served back on the next call (same inputs, different stub).
        clear_compression_cache()

    def tearDown(self):
        clear_compression_cache()

    def _run(self, text, preserve_identifiers, preserve_sections, fake_complete):
        import asyncio
        import local_compress_lib as L
        real_complete, real_resolve, real_client = L.complete, L.resolve_model, L.client
        L.complete = fake_complete
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        try:
            return asyncio.run(L.compress(
                text, focus="summarize this", skip_if_under_chars=0,
                preserve_identifiers=preserve_identifiers, preserve_sections=preserve_sections,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real_complete, real_resolve, real_client

    def test_credential_echoed_by_the_model_is_redacted_with_preserve_sections_alone(self):
        secret = "ghp_" + "a" * 36
        # Unstructured (no '## ' headings) -- _compress_each_section returns
        # None for this, so it falls through to the whole-text path.
        text = "x" * 3000
        out = self._run(
            text, preserve_identifiers=False, preserve_sections=True,
            fake_complete=lambda *a, **k: f"Summary echoing secret: {secret}",
        )
        self.assertNotIn(secret, out)
        self.assertIn("credential-shaped value(s) redacted", out)

    def test_still_redacted_with_preserve_identifiers_alone_no_regression(self):
        # Confirms the existing preserve_identifiers-only path (already
        # covered elsewhere) still works after widening the gate to an "or".
        secret = "ghp_" + "a" * 36
        text = "x" * 3000
        out = self._run(
            text, preserve_identifiers=True, preserve_sections=False,
            fake_complete=lambda *a, **k: f"Summary echoing secret: {secret}",
        )
        self.assertNotIn(secret, out)
        self.assertIn("credential-shaped value(s) redacted", out)

    def test_not_redacted_when_neither_flag_is_set(self):
        # Confirms the fix stays gated -- gist-style compression (neither
        # flag set) is unaffected, matching this function's documented
        # "only in artifact-producing modes" tradeoff.
        secret = "ghp_" + "a" * 36
        text = "x" * 3000
        out = self._run(
            text, preserve_identifiers=False, preserve_sections=False,
            fake_complete=lambda *a, **k: f"Summary echoing secret: {secret}",
        )
        self.assertIn(secret, out)


class TrailingSummaryRepair(unittest.TestCase):
    """
    find_trailing_summary_line / append_trailing_summary_if_missing.

    Reproduced before fixing: a real pytest failure log's map-step summary
    correctly listed all 3 failures but dropped the final "3 failed, 39
    passed in 8.72s" line -- the single most decision-relevant fact in the
    whole log -- across both the default focus and a bullet-per-fact one. A
    prompt-only fix ("always keep the final totals line") stopped the drop
    but measurably made the SAME model list every individual passing test on
    an unrelated clean build instead of aggregating them, so this is a
    deterministic repair instead, same reasoning as append_missing_identifiers.
    """

    PYTEST_LOG = (
        "collected 42 items\n"
        "tests/test_a.py .........F\n\n"
        "=================== FAILURES ===================\n"
        "...\n"
        "======================= 3 failed, 39 passed in 8.72s ==========================="
    )

    def test_detects_pytest_totals_line(self):
        self.assertEqual(
            find_trailing_summary_line(self.PYTEST_LOG), "3 failed, 39 passed in 8.72s",
        )

    def test_detects_jest_totals_block(self):
        text = "...\nTests:       3 failed, 39 passed, 42 total\nTime:        4.2s"
        self.assertEqual(
            find_trailing_summary_line(text), "Tests:       3 failed, 39 passed, 42 total",
        )

    def test_detects_go_test_result(self):
        self.assertEqual(find_trailing_summary_line("--- FAIL: TestFoo\nFAIL"), "FAIL")
        self.assertEqual(
            find_trailing_summary_line("ok  \tgithub.com/x/y\t0.003s"),
            "ok  \tgithub.com/x/y\t0.003s",
        )

    def test_detects_gradle_build_result(self):
        text = "> Task :test\n...\nBUILD SUCCESSFUL in 47s\n14 actionable tasks: 14 executed"
        self.assertEqual(find_trailing_summary_line(text), "BUILD SUCCESSFUL in 47s")

    def test_detects_dotnet_test_result(self):
        text = "Starting test execution...\nPassed!  - Failed: 0, Passed: 10, Skipped: 0, Total: 10, Duration: 1 s"
        self.assertIn("Passed!  - Failed: 0, Passed: 10, Skipped: 0, Total: 10", find_trailing_summary_line(text))

    def test_detects_rspec_result(self):
        text = "..F..\n\nFailures:\n\n1) ...\n\n10 examples, 1 failure"
        self.assertEqual(find_trailing_summary_line(text), "10 examples, 1 failure")

    def test_no_totals_line_returns_none(self):
        self.assertIsNone(find_trailing_summary_line("just some ordinary prose with no result line"))

    def test_totals_shaped_text_outside_window_is_ignored(self):
        # A totals-shaped line from an early retry, buried well before the
        # end -- not the actual final result, and must not be mistaken for it.
        # "padding\n" is 8 chars; 200 repetitions = 1600 chars of noise, which
        # exceeds the 1500-char window so this still returns None after #47.
        text = "3 failed, 39 passed in 8.72s\n" + ("padding\n" * 200) + "retrying..."
        self.assertIsNone(find_trailing_summary_line(text))

    def test_result_line_now_found_within_widened_window(self):
        # Issue #47: the old 500-char window missed a result line pushed back
        # by trailing CI noise (coverage tables, upload confirmations). With
        # the new 1500-char window the line is found even when ~910 chars of
        # trailing content follow it -- 900 chars of noise plus "retrying..."
        # (11 chars) -- a realistic coverage-table size.
        noise = "coverage-noise-line\n" * 45  # 20 * 45 = 900 chars
        text = "3 failed, 39 passed in 8.72s\n" + noise + "retrying..."
        self.assertEqual(find_trailing_summary_line(text), "3 failed, 39 passed in 8.72s")

    def test_append_is_a_noop_when_line_already_present(self):
        summary = "All 3 failures were in test_a.py.\n\n3 failed, 39 passed in 8.72s"
        self.assertEqual(append_trailing_summary_if_missing(self.PYTEST_LOG, summary), summary)

    def test_append_is_a_noop_when_no_totals_line_exists(self):
        summary = "A gist of some ordinary text."
        self.assertEqual(
            append_trailing_summary_if_missing("just some ordinary prose with no result line", summary),
            summary,
        )

    def test_append_restores_the_dropped_line_verbatim(self):
        summary = "3 failures found in test_a.py, all AssertionErrors."
        out = append_trailing_summary_if_missing(self.PYTEST_LOG, summary)
        self.assertTrue(out.startswith(summary), "the summary itself must be preserved")
        self.assertIn("3 failed, 39 passed in 8.72s", out)


class TrailingSummaryPatternCoverage(unittest.TestCase):
    """Issue #62: extend _TRAILING_SUMMARY_PATTERNS with Playwright, Cypress,
    and JUnit 5 console-launcher output shapes (same structural-evidence
    requirement as all the existing patterns -- keyword + digits, not prose).

    Each test confirms find_trailing_summary_line() extracts the exact summary
    line, which is also what append_trailing_summary_if_missing() restores
    verbatim when the compressor drops it.
    """

    # --- Playwright -------------------------------------------------------

    def test_detects_playwright_passed_seconds(self):
        text = "Running 10 tests using 4 workers\n\n  10 passed (4.2s)"
        self.assertEqual(find_trailing_summary_line(text), "10 passed (4.2s)")

    def test_detects_playwright_failed_minutes_seconds(self):
        # Playwright uses "Nm Ys" for runs longer than a minute.
        text = "  ...\n  1 failed (1m 2.3s)"
        self.assertEqual(find_trailing_summary_line(text), "1 failed (1m 2.3s)")

    def test_detects_playwright_passed_whole_seconds(self):
        text = "  12 passed (30s)"
        self.assertEqual(find_trailing_summary_line(text), "12 passed (30s)")

    def test_detects_playwright_passed_milliseconds(self):
        # Playwright also uses "Xms" for very fast test runs.
        text = "  5 passed (487ms)"
        self.assertEqual(find_trailing_summary_line(text), "5 passed (487ms)")

    def test_detects_playwright_passed_fractional_minutes(self):
        # For very long runs Playwright emits a fractional-minute form "X.Ym".
        text = "  20 passed (1.7m)"
        self.assertEqual(find_trailing_summary_line(text), "20 passed (1.7m)")

    def test_detects_playwright_failed_bare_no_duration(self):
        # An all-failing run emits a bare "N failed" line with no duration.
        # Confirmed in Playwright's own CLI output (see issue #62).
        text = "Running 2 tests\n\n  2 failed"
        self.assertEqual(find_trailing_summary_line(text), "2 failed")

    def test_playwright_line_is_restored_when_dropped(self):
        log = "Running 5 tests\n\n  ...\n\n  5 passed (2.1s)"
        summary = "All 5 tests in the suite ran without issues."
        out = append_trailing_summary_if_missing(log, summary)
        self.assertIn("5 passed (2.1s)", out)
        self.assertIn("[RESULT]", out)

    # --- Cypress ----------------------------------------------------------

    def test_detects_cypress_passing_milliseconds(self):
        text = "  3 passing (487ms)"
        self.assertEqual(find_trailing_summary_line(text), "3 passing (487ms)")

    def test_detects_cypress_passing_seconds(self):
        text = "  3 passing (4s)"
        self.assertEqual(find_trailing_summary_line(text), "3 passing (4s)")

    def test_detects_cypress_passing_minutes_only(self):
        # Mocha's duration formatter emits "Xm" (minute-only) for runs >= 60 s.
        text = "  3 passing (1m)"
        self.assertEqual(find_trailing_summary_line(text), "3 passing (1m)")

    def test_detects_cypress_failing_no_duration(self):
        text = "  3 passing (4s)\n  1 failing"
        self.assertEqual(find_trailing_summary_line(text), "1 failing")

    def test_detects_cypress_pending(self):
        text = "  2 passing (1s)\n  1 pending"
        self.assertEqual(find_trailing_summary_line(text), "1 pending")

    def test_cypress_line_is_restored_when_dropped(self):
        log = "Running Cypress tests\n  2 passing (3s)\n  1 failing"
        summary = "One test failed in the login suite."
        out = append_trailing_summary_if_missing(log, summary)
        self.assertIn("1 failing", out)
        self.assertIn("[RESULT]", out)

    # --- JUnit 5 console launcher -----------------------------------------

    def test_detects_junit5_tests_found(self):
        text = "[         3 tests found           ]"
        self.assertEqual(
            find_trailing_summary_line(text), "[         3 tests found           ]"
        )

    def test_detects_junit5_tests_failed(self):
        text = "...\n[         1 tests failed          ]"
        self.assertEqual(
            find_trailing_summary_line(text), "[         1 tests failed          ]"
        )

    def test_detects_junit5_tests_successful(self):
        # JUnit 5 ConsoleLauncher uses "successful" not "succeeded" --
        # real output: "[         5 tests successful       ]"
        text = "[         5 tests successful       ]"
        self.assertIsNotNone(find_trailing_summary_line(text))
        self.assertIn("5 tests successful", find_trailing_summary_line(text))

    def test_junit5_line_is_restored_when_dropped(self):
        log = "JUnit Platform Launcher\n[         2 tests started         ]\n[         1 tests failed          ]"
        summary = "One test failed."
        out = append_trailing_summary_if_missing(log, summary)
        self.assertIn("1 tests failed", out)
        self.assertIn("[RESULT]", out)

    # --- Negative cases (must not false-positive) -------------------------

    def test_plain_prose_with_numbers_is_not_matched(self):
        # Ordinary sentences that contain the same keywords but lack the
        # structural shape (parenthesised duration / bracket+keyword).
        self.assertIsNone(
            find_trailing_summary_line("We had 3 failing builds last month")
        )
        self.assertIsNone(
            find_trailing_summary_line("The 5 passing criteria were reviewed")
        )


class ResolveModelStaleEnvCheck(unittest.TestCase):
    """Issue #41: resolve_model() used to check stale_env_warning() only on
    the auto-detect branch -- reached only when neither explicit_model nor
    DEFAULT_MODEL short-circuited first. That meant a partial env-var
    migration (e.g. LMSTUDIO_BASE_URL never renamed to
    CLAUDE_RUNWAY_LMSTUDIO_URL) went undiagnosed whenever the model itself
    resolved fine via the new CLAUDE_RUNWAY_LMSTUDIO_MODEL var or an explicit
    param -- the caller just silently got DEFAULT_BASE_URL's hardcoded
    default instead of their real LM Studio URL, with no explanation.

    These stub stale_env_warning() directly (same style as
    test_redirect_webfetch_to_fetch_url.py) rather than manipulating real env
    vars, since resolve_model() only calls it -- it doesn't inspect
    individual var names itself."""

    def setUp(self):
        import local_compress_lib as L
        self.L = L
        self._real_stale_env_warning = L.stale_env_warning
        self._real_default_model = L.DEFAULT_MODEL

    def tearDown(self):
        self.L.stale_env_warning = self._real_stale_env_warning
        self.L.DEFAULT_MODEL = self._real_default_model

    def test_stale_env_surfaces_even_with_explicit_model(self):
        self.L.stale_env_warning = lambda: "LMSTUDIO_BASE_URL is set but no longer read"
        model, error = self.L.resolve_model("some-explicit-model", None)
        self.assertIsNone(model)
        self.assertEqual(error, "LMSTUDIO_BASE_URL is set but no longer read")

    def test_stale_env_surfaces_even_with_pinned_default_model(self):
        self.L.DEFAULT_MODEL = "pinned-model"
        self.L.stale_env_warning = lambda: "LMSTUDIO_BASE_URL is set but no longer read"
        model, error = self.L.resolve_model(None, None)
        self.assertIsNone(model)
        self.assertEqual(error, "LMSTUDIO_BASE_URL is set but no longer read")

    def test_explicit_model_still_resolves_when_env_is_clean(self):
        self.L.stale_env_warning = lambda: None
        model, error = self.L.resolve_model("some-explicit-model", None)
        self.assertEqual(model, "some-explicit-model")
        self.assertIsNone(error)

    def test_pinned_default_model_still_resolves_when_env_is_clean(self):
        self.L.DEFAULT_MODEL = "pinned-model"
        self.L.stale_env_warning = lambda: None
        model, error = self.L.resolve_model(None, None)
        self.assertEqual(model, "pinned-model")
        self.assertIsNone(error)

    def test_auto_detect_branch_still_hard_fails_on_stale_env(self):
        # Pre-existing behavior (the only branch the old code covered) must
        # still hold: no explicit_model, no DEFAULT_MODEL, stale env present.
        self.L.DEFAULT_MODEL = None
        self.L.stale_env_warning = lambda: "LMSTUDIO_MODEL is set but no longer read"
        model, error = self.L.resolve_model(None, None)
        self.assertIsNone(model)
        self.assertEqual(error, "LMSTUDIO_MODEL is set but no longer read")


class ConcurrentClassification(unittest.TestCase):
    """Issue #59: classify_relevant calls must run concurrently for multi-chunk
    input with a selective focus, not sequentially.

    All tests use stub injection (same pattern as SectionCompressionEndToEnd)
    -- no LM Studio needed.
    """

    def setUp(self):
        # Each test stubs classify_relevant and complete differently -- clear
        # the cache so prior-test results don't bypass those stubs.
        clear_compression_cache()

    def tearDown(self):
        clear_compression_cache()

    def _run_compress(self, text, focus, fake_classify, fake_complete,
                      skip_if_under_chars=0):
        import asyncio
        import local_compress_lib as L

        real_classify = L.classify_relevant
        real_complete = L.complete
        real_resolve = L.resolve_model
        real_client = L.client

        L.classify_relevant = fake_classify
        L.complete = fake_complete
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        try:
            return asyncio.run(L.compress(
                text,
                focus=focus,
                skip_if_under_chars=skip_if_under_chars,
                chunk_chars=len(text) // 3 + 1,  # force 3 chunks for a 3x text
            ))
        finally:
            L.classify_relevant = real_classify
            L.complete = real_complete
            L.resolve_model = real_resolve
            L.client = real_client

    def test_all_chunks_classified_with_selective_focus(self):
        """Every chunk must be presented to classify_relevant, regardless of
        order, when the focus is selective."""
        seen_chunks = []

        def fake_classify(client, model, focus, chunk, chunk_index, chunk_count,
                          truncated=False):
            seen_chunks.append(chunk_index)
            return True  # all relevant

        # 3-chunk text: each word padded to fill the chunk_chars window
        text = ("A" * 100 + " " + "B" * 100 + " " + "C" * 100)
        self._run_compress(
            text,
            focus="chocolate cake recipes",  # selective, not in text -> no-op for summaries
            fake_classify=fake_classify,
            fake_complete=lambda *a, **k: "chunk summary",
        )
        # All three chunk indices must have been classified.
        self.assertEqual(sorted(seen_chunks), [1, 2, 3])

    def test_classification_skipped_for_non_selective_focus(self):
        """With a non-selective focus (e.g. 'summarize this'), classify_relevant
        must never be called -- all chunks pass straight to extraction."""
        classify_called = []

        def fake_classify(*a, **k):
            classify_called.append(True)
            return True

        text = "A" * 100 + " " + "B" * 100 + " " + "C" * 100
        self._run_compress(
            text,
            focus="summarize this",
            fake_classify=fake_classify,
            fake_complete=lambda *a, **k: "chunk summary",
        )
        self.assertEqual(classify_called, [],
                         "classify_relevant must not be called for a non-selective focus")

    def test_summaries_are_in_document_order(self):
        """Even though classification runs concurrently, chunk summaries must
        appear in document order in the final output."""
        # Stagger the chunk labels so out-of-order results would be visible.
        def fake_classify(client, model, focus, chunk, chunk_index, chunk_count,
                          truncated=False):
            return True

        def fake_complete(client, model, system, content):
            # Return a distinct, identifiable summary per chunk.
            # content is the chunk itself (no focus prefix in the system arg
            # changes the chunk content).
            if "CHUNK1" in content:
                return "summary-of-chunk-1"
            if "CHUNK2" in content:
                return "summary-of-chunk-2"
            if "CHUNK3" in content:
                return "summary-of-chunk-3"
            # Reduce step -- return a combined summary.
            return content  # pass through so all three appear in output

        # Three distinct chunks, padded to ensure they're not merged.
        chunk_body = "X" * 50
        text = f"CHUNK1 {chunk_body} CHUNK2 {chunk_body} CHUNK3 {chunk_body}"
        result = self._run_compress(
            text,
            focus="CHUNK content",
            fake_classify=fake_classify,
            fake_complete=fake_complete,
        )
        # The order of "part 1/2/3" labels from the reduce step confirms ordering.
        idx1 = result.find("summary-of-chunk-1")
        idx2 = result.find("summary-of-chunk-2")
        idx3 = result.find("summary-of-chunk-3")
        self.assertGreater(idx1, -1, "summary for chunk 1 must appear in output")
        self.assertGreater(idx2, -1, "summary for chunk 2 must appear in output")
        self.assertGreater(idx3, -1, "summary for chunk 3 must appear in output")
        self.assertLess(idx1, idx2, "chunk 1 summary must precede chunk 2")
        self.assertLess(idx2, idx3, "chunk 2 summary must precede chunk 3")

    def test_classification_failure_on_any_chunk_returns_error(self):
        """If classify_relevant returns None for any chunk (request failure),
        compress must return an Error string and not silently skip the chunk."""
        call_count = [0]

        def fake_classify(client, model, focus, chunk, chunk_index, chunk_count,
                          truncated=False):
            call_count[0] += 1
            # Fail the second chunk.
            if chunk_index == 2:
                return None
            return True

        text = "A" * 100 + " " + "B" * 100 + " " + "C" * 100
        result = self._run_compress(
            text,
            focus="chocolate cake recipes",
            fake_classify=fake_classify,
            fake_complete=lambda *a, **k: "summary",
        )
        self.assertTrue(result.startswith("Error:"),
                        f"a classification failure must produce an Error string, got: {result!r}")
        self.assertIn("LM Studio request failed while checking relevance", result)

    def test_irrelevant_chunks_excluded_from_summaries(self):
        """A chunk classified as False (not relevant) must not appear in the
        final reduce step, same as in the old sequential path."""
        def fake_classify(client, model, focus, chunk, chunk_index, chunk_count,
                          truncated=False):
            # Only chunk 2 is relevant.
            return chunk_index == 2

        def fake_complete(client, model, system, content):
            if "CHUNK2" in content:
                return "only-relevant-summary"
            # Reduce step (only one summary, so this branch is never reached).
            return content

        chunk_body = "X" * 50
        text = f"CHUNK1 {chunk_body} CHUNK2 {chunk_body} CHUNK3 {chunk_body}"
        result = self._run_compress(
            text,
            focus="CHUNK content",
            fake_classify=fake_classify,
            fake_complete=fake_complete,
        )
        # Only the relevant chunk's summary should appear.
        self.assertIn("only-relevant-summary", result)
        # The irrelevant chunks' identifiers must not appear in any summary
        # (since compress() never called complete() on them).
        self.assertNotIn("CHUNK1", result)
        self.assertNotIn("CHUNK3", result)


class CompressionCache(unittest.TestCase):
    """Issue #60: in-session compression cache that skips redundant LM Studio
    round-trips for the same content compressed more than once.

    All tests use stub injection (same pattern as SectionCompressionEndToEnd)
    -- no LM Studio needed. Each test explicitly clears the cache on setUp and
    tearDown so test isolation is guaranteed regardless of run order.
    """

    def setUp(self):
        clear_compression_cache()

    def tearDown(self):
        clear_compression_cache()

    def _stub_compress(self, text, focus="summarize this",
                       model=None, max_chars=None,
                       preserve_identifiers=False, preserve_sections=False,
                       call_counter=None):
        """Drive compress() with a fake model that returns a deterministic
        summary, counting how many times the model is actually called."""
        import asyncio
        import local_compress_lib as L

        real_complete = L.complete
        real_resolve = L.resolve_model
        real_client = L.client

        def counting_complete(client, model_id, system, content):
            if call_counter is not None:
                call_counter.append(1)
            return "STUB_SUMMARY"

        L.complete = counting_complete
        L.resolve_model = lambda m, b: (m or "stub-model", None)
        L.client = lambda b: object()
        try:
            return asyncio.run(L.compress(
                text, focus=focus, skip_if_under_chars=0,
                model=model, max_chars=max_chars,
                preserve_identifiers=preserve_identifiers,
                preserve_sections=preserve_sections,
            ))
        finally:
            L.complete = real_complete
            L.resolve_model = real_resolve
            L.client = real_client

    # ----- _compression_cache_key unit tests -----

    def test_same_inputs_produce_same_key(self):
        k1 = _compression_cache_key("text", "focus", "model", None, False, False)
        k2 = _compression_cache_key("text", "focus", "model", None, False, False)
        self.assertEqual(k1, k2)

    def test_different_text_produces_different_key(self):
        k1 = _compression_cache_key("text-A", "focus", "model", None, False, False)
        k2 = _compression_cache_key("text-B", "focus", "model", None, False, False)
        self.assertNotEqual(k1, k2)

    def test_different_focus_produces_different_key(self):
        k1 = _compression_cache_key("text", "focus-A", "model", None, False, False)
        k2 = _compression_cache_key("text", "focus-B", "model", None, False, False)
        self.assertNotEqual(k1, k2)

    def test_different_model_produces_different_key(self):
        k1 = _compression_cache_key("text", "focus", "model-A", None, False, False)
        k2 = _compression_cache_key("text", "focus", "model-B", None, False, False)
        self.assertNotEqual(k1, k2)

    def test_different_max_chars_produces_different_key(self):
        k1 = _compression_cache_key("text", "focus", "model", 100, False, False)
        k2 = _compression_cache_key("text", "focus", "model", 200, False, False)
        self.assertNotEqual(k1, k2)

    def test_different_preserve_identifiers_produces_different_key(self):
        k1 = _compression_cache_key("text", "focus", "model", None, True, False)
        k2 = _compression_cache_key("text", "focus", "model", None, False, False)
        self.assertNotEqual(k1, k2)

    def test_different_preserve_sections_produces_different_key(self):
        k1 = _compression_cache_key("text", "focus", "model", None, False, True)
        k2 = _compression_cache_key("text", "focus", "model", None, False, False)
        self.assertNotEqual(k1, k2)

    def test_different_chunk_chars_produces_different_key(self):
        # chunk_chars changes chunk boundaries, model call count, and the
        # returned "across N chunk(s)" metadata string -- not just a throughput
        # knob (Copilot PR review, round 1).
        k1 = _compression_cache_key("text", "focus", "model", None, False, False, chunk_chars=100)
        k2 = _compression_cache_key("text", "focus", "model", None, False, False, chunk_chars=500)
        self.assertNotEqual(k1, k2)

    def test_auto_truncated_flag_produces_different_key(self):
        # By the time _compression_cache_key is called, max_chars may have been
        # mutated from None to the auto-detected boundary integer -- so two
        # calls with the same effective boundary but different origin (one
        # auto-detected, one explicit) would produce the same max_chars value
        # in the key. auto_truncated distinguishes them because their result
        # metadata differs (Copilot PR review, rounds 2-4).
        k_auto = _compression_cache_key("text", "focus", "model", 500, False, False, auto_truncated=True)
        k_explicit = _compression_cache_key("text", "focus", "model", 500, False, False, auto_truncated=False)
        self.assertNotEqual(k_auto, k_explicit)

    def test_different_effective_url_produces_different_key(self):
        # base_url identifies the physical LM Studio server. Model IDs are not
        # globally unique across instances -- two different servers can expose
        # the same ID for different weights. Including the effective URL keeps
        # MCP per-call base_url requests from sharing cache entries across
        # servers (Copilot PR review, rounds 1 and 4).
        k1 = _compression_cache_key("text", "focus", "model", None, False, False,
                                    effective_url="http://localhost:1234/v1")
        k2 = _compression_cache_key("text", "focus", "model", None, False, False,
                                    effective_url="http://remote-server:1234/v1")
        self.assertNotEqual(k1, k2)

    def test_none_model_and_empty_string_differ_from_named_model(self):
        # None and '' both normalize to '' in the key (both mean "no model
        # pinned") -- but a real model id must never match either.
        k_none = _compression_cache_key("text", "focus", None, None, False, False)
        k_real = _compression_cache_key("text", "focus", "gemma-3-4b", None, False, False)
        self.assertNotEqual(k_none, k_real)

    # ----- cache hit / miss via compress() integration -----

    def test_second_call_with_identical_inputs_is_a_cache_hit(self):
        """The model must be called exactly once; the second call must return
        the cached result without invoking the model again."""
        text = "x" * 500  # over skip_if_under_chars=0
        calls = []
        first = self._stub_compress(text, call_counter=calls)
        # Cache is now populated from the first call.
        calls_after_first = len(calls)

        second = self._stub_compress(text, call_counter=calls)
        calls_after_second = len(calls)

        self.assertEqual(first, second, "cached result must equal the live result")
        # The model was invoked at least once for the first call.
        self.assertGreater(calls_after_first, 0, "first call must invoke the model")
        # No additional model calls on the second call.
        self.assertEqual(
            calls_after_second, calls_after_first,
            "second call with identical inputs must not invoke the model",
        )

    def test_different_text_is_a_cache_miss(self):
        calls = []
        self._stub_compress("text-A", call_counter=calls)
        after_first = len(calls)
        self._stub_compress("text-B", call_counter=calls)
        after_second = len(calls)
        self.assertGreater(after_second, after_first,
                           "different text must be a cache miss and invoke the model")

    def test_different_focus_is_a_cache_miss(self):
        calls = []
        self._stub_compress("text", focus="focus-A", call_counter=calls)
        after_first = len(calls)
        self._stub_compress("text", focus="focus-B", call_counter=calls)
        after_second = len(calls)
        self.assertGreater(after_second, after_first,
                           "different focus must be a cache miss")

    def test_different_model_is_a_cache_miss(self):
        calls = []
        self._stub_compress("text", model="model-A", call_counter=calls)
        after_first = len(calls)
        self._stub_compress("text", model="model-B", call_counter=calls)
        after_second = len(calls)
        self.assertGreater(after_second, after_first,
                           "different model must be a cache miss")

    def test_different_max_chars_is_a_cache_miss(self):
        # A long text that would actually be truncated differently.
        text = "x" * 1000
        calls = []
        self._stub_compress(text, max_chars=500, call_counter=calls)
        after_first = len(calls)
        self._stub_compress(text, max_chars=800, call_counter=calls)
        after_second = len(calls)
        self.assertGreater(after_second, after_first,
                           "different max_chars must be a cache miss")

    def test_auto_truncated_vs_explicit_same_boundary_is_a_cache_miss(self):
        """Copilot PR review, rounds 2-4: by the time the cache key is computed,
        max_chars may have been mutated from None to the auto-detected boundary
        integer. A positional call (auto-truncated) and an explicit call with
        the same boundary value would produce the same max_chars in the key
        -- but their results differ because auto-truncation adds a '[note:
        focus looked positional...]' suffix while an explicit call adds a
        '[note: max_chars=N truncated...]' suffix. The auto_truncated flag in
        the key separates these two cases."""
        import asyncio
        import local_compress_lib as L

        # A short text under skip_if_under_chars to exercise the truncated
        # early-return path without needing LM Studio at all: the positional
        # auto-detected boundary lands under skip_if_under_chars, so both
        # calls return the truncated text (with different disclosure notes)
        # without ever calling the model. This is the exact path where the
        # auto_truncated / explicit collision could produce the wrong note.

        # Build a text with a clear positional boundary that the heuristic
        # will detect -- same structure as FindFirstHeadingBoundary tests.
        lead = "A" * 350 + "."
        heading = "Section Two"
        rest = "B" * 100 + "."
        text = f"{lead}\n{heading}\n{rest}"

        # First call: positional focus, no explicit max_chars -- auto-truncates.
        result_auto = asyncio.run(L.compress(
            text, focus="summarize the lead section", skip_if_under_chars=10_000,
        ))
        # Auto-truncated result must have the positional note.
        self.assertIn("focus looked positional", result_auto, "auto-truncated call must report positional note")

        # Figure out which boundary was detected so we can call explicitly.
        # The note says "only the first N chars" -- extract N.
        import re as _re
        match = _re.search(r"first (\d+) chars", result_auto)
        self.assertIsNotNone(match, f"could not extract truncation size from: {result_auto!r}")
        detected_boundary = int(match.group(1))

        # Second call: same text, same focus, explicit max_chars equal to the
        # auto-detected boundary -- must be a cache MISS (different key).
        result_explicit = asyncio.run(L.compress(
            text, focus="summarize the lead section", skip_if_under_chars=10_000,
            max_chars=detected_boundary,
        ))
        # Explicit result must NOT have the positional note.
        self.assertNotIn("focus looked positional", result_explicit,
                         "explicit max_chars call must not get the positional note")
        # And the two results must be different strings (different metadata).
        self.assertNotEqual(result_auto, result_explicit,
                            "auto-truncated and explicit same-boundary results must differ")

    def test_different_chunk_chars_is_a_cache_miss(self):
        """Copilot PR review, round 1: chunk_chars changes chunk boundaries,
        model call count, and the 'across N chunk(s)' metadata string in the
        returned result -- caching across chunk_chars values would return a
        result with metadata that is false for the new invocation."""
        import asyncio
        import local_compress_lib as L

        real_complete, real_resolve, real_client = L.complete, L.resolve_model, L.client
        calls = []

        def counting_complete(client, model_id, system, content):
            calls.append(1)
            return "STUB_SUMMARY"

        L.complete = counting_complete
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        try:
            # Use a text large enough to produce multiple chunks at chunk_chars=50.
            text = "word " * 200  # 1000 chars
            asyncio.run(L.compress(text, skip_if_under_chars=0, chunk_chars=50))
            after_first = len(calls)
            # Different chunk_chars must be a cache miss and invoke the model again.
            asyncio.run(L.compress(text, skip_if_under_chars=0, chunk_chars=500))
            after_second = len(calls)
        finally:
            L.complete, L.resolve_model, L.client = real_complete, real_resolve, real_client

        self.assertGreater(after_first, 0, "first call must invoke the model")
        self.assertGreater(after_second, after_first,
                           "different chunk_chars must be a cache miss")

    def test_under_threshold_text_is_not_cached(self):
        """compress() returns the original text unchanged for under-threshold
        input, before model resolution or the cache is ever consulted. Calling
        it again must still return the original unchanged -- the cache must
        not interfere with or 'cache' an under-threshold no-op path."""
        import asyncio
        import local_compress_lib as L

        short_text = "just a few words"
        # skip_if_under_chars is large so this is definitely under-threshold.
        result = asyncio.run(L.compress(short_text, skip_if_under_chars=10_000))
        self.assertEqual(result, short_text,
                         "under-threshold text must be returned unchanged, not compressed")
        # Cache must be empty -- the no-op path must never call _cache_put.
        self.assertEqual(len(_compression_cache), 0,
                         "under-threshold result must not be stored in the cache")

    def test_error_returns_are_not_cached(self):
        """Error strings (from a failed LM Studio call) must not be cached --
        a transient failure should NOT prevent future calls from succeeding."""
        import asyncio
        import local_compress_lib as L

        real_resolve = L.resolve_model
        real_client = L.client
        real_complete = L.complete

        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        # complete() returning None simulates a failed LM Studio call.
        L.complete = lambda *a, **k: None
        try:
            result = asyncio.run(L.compress("x" * 500, skip_if_under_chars=0))
        finally:
            L.resolve_model = real_resolve
            L.client = real_client
            L.complete = real_complete

        self.assertTrue(result.startswith("Error:"),
                        f"a failed call must return an Error string, got: {result!r}")
        self.assertEqual(len(_compression_cache), 0,
                         "Error returns must not be stored in the cache")

    def test_section_partial_failure_result_is_not_cached(self):
        """Copilot PR review, round 2: a section result reporting 'kept verbatim
        after a failed call' means at least one prose section had a transient
        model failure. Caching it would permanently suppress retries for those
        sections on every subsequent call. The cache must stay empty when any
        section call failed, not only when ALL of them failed."""
        import asyncio
        import local_compress_lib as L

        real_complete, real_resolve, real_client = L.complete, L.resolve_model, L.client
        calls = {"n": 0}

        def partial_fail(*a, **k):
            # First call fails (None), subsequent calls succeed -- simulates a
            # transient failure on one section while others complete normally.
            calls["n"] += 1
            return None if calls["n"] == 1 else "compressed"

        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        L.complete = partial_fail
        try:
            result = asyncio.run(L.compress(
                COMPACT_TEMPLATE, focus="handoff", skip_if_under_chars=0,
                preserve_sections=True,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real_complete, real_resolve, real_client

        # Confirm the partial-failure indicator is present (precondition).
        self.assertIn(
            "kept verbatim after a failed call", result,
            f"precondition: expected partial-failure note in result, got: {result[:120]!r}",
        )
        # The partial-failure result must NOT have been stored in the cache.
        self.assertEqual(
            len(_compression_cache), 0,
            "a section result with any failed call must not be cached; "
            "a repeated call must retry LM Studio for the failed section",
        )

    def test_section_outage_result_is_not_cached(self):
        """Copilot PR review, round 1: a section-mode result prepended with
        '[LM Studio appears unreachable ...]' means every model call that was
        SENT failed outright. Caching it would cause repeated calls to keep
        returning the degraded preserved-content output instead of retrying
        once LM Studio recovers."""
        import asyncio
        import local_compress_lib as L

        real_complete, real_resolve, real_client = L.complete, L.resolve_model, L.client
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        # complete() returning None triggers the total-outage path in
        # _compress_each_section (every sent request fails outright).
        L.complete = lambda *a, **k: None
        try:
            result = asyncio.run(L.compress(
                COMPACT_TEMPLATE, focus="handoff", skip_if_under_chars=0,
                preserve_sections=True,
            ))
        finally:
            L.complete, L.resolve_model, L.client = real_complete, real_resolve, real_client

        # Confirm the total-outage prefix is actually present (precondition).
        self.assertTrue(
            result.startswith("[LM Studio appears unreachable"),
            f"precondition: expected outage prefix, got: {result[:80]!r}",
        )
        # The outage result must NOT have been stored in the cache.
        self.assertEqual(
            len(_compression_cache), 0,
            "a section-mode outage result must not be stored in the cache; "
            "a repeated call must retry LM Studio rather than serving the degraded result",
        )

    # ----- cache size cap / eviction -----

    def test_oversized_value_is_not_cached(self):
        """Copilot PR review, round 5: _cache_put must silently skip entries
        whose value exceeds _CACHE_MAX_VALUE_CHARS to prevent the long-lived
        MCP server from holding hundreds of MB of nearly-input-sized results
        (a preserve_sections run with mostly verbatim sections can produce a
        result close to the 2,000,000-char input limit)."""
        import local_compress_lib as L

        key = _compression_cache_key("text", "focus", "model", None, False, False)
        oversized_value = "x" * (L._CACHE_MAX_VALUE_CHARS + 1)
        L._cache_put(key, oversized_value)
        self.assertEqual(
            len(_compression_cache), 0,
            "a result exceeding _CACHE_MAX_VALUE_CHARS must not be stored",
        )

    def test_normal_sized_value_is_still_cached(self):
        """Regression guard: the byte cap must not affect normally sized results."""
        import local_compress_lib as L

        key = _compression_cache_key("text", "focus", "model", None, False, False)
        normal_value = "x" * (L._CACHE_MAX_VALUE_CHARS - 1)
        L._cache_put(key, normal_value)
        self.assertEqual(
            len(_compression_cache), 1,
            "a result under _CACHE_MAX_VALUE_CHARS must be stored normally",
        )

    def test_oversized_value_does_not_evict_existing_entry(self):
        """An oversized result that is skipped must not evict an existing cached
        entry to make room -- it was never going to be stored."""
        import local_compress_lib as L

        existing_key = _compression_cache_key("text-A", "focus", "model", None, False, False)
        existing_value = "small result"
        L._cache_put(existing_key, existing_value)
        self.assertEqual(len(_compression_cache), 1, "precondition: one entry in cache")

        oversized_key = _compression_cache_key("text-B", "focus", "model", None, False, False)
        oversized_value = "x" * (L._CACHE_MAX_VALUE_CHARS + 1)
        L._cache_put(oversized_key, oversized_value)

        # Still exactly one entry: the oversized result was dropped, and the
        # existing entry was not evicted.
        self.assertEqual(len(_compression_cache), 1, "oversized value must not evict existing entry")
        self.assertIn(existing_key, _compression_cache, "existing entry must still be present")

    def test_cache_size_cap_evicts_oldest_entry(self):
        """When the cache is full, the oldest entry must be evicted to make
        room for a new one -- the total size must never exceed _CACHE_MAX_ENTRIES."""
        import local_compress_lib as L

        cap = L._CACHE_MAX_ENTRIES
        # Fill the cache to exactly the cap by adding distinct keys directly.
        for i in range(cap):
            key = _compression_cache_key(
                f"text-{i}", "focus", "model", None, False, False
            )
            L._cache_put(key, f"result-{i}")

        self.assertEqual(len(_compression_cache), cap,
                         "cache must be exactly at capacity before the eviction")

        # The first key is the oldest -- capture it before inserting another.
        oldest_key = _compression_cache_key(
            "text-0", "focus", "model", None, False, False
        )
        self.assertIn(oldest_key, _compression_cache,
                      "oldest key must be in the cache before eviction")

        # Insert one more to trigger eviction.
        new_key = _compression_cache_key(
            "text-new", "focus", "model", None, False, False
        )
        L._cache_put(new_key, "result-new")

        self.assertEqual(len(_compression_cache), cap,
                         "cache size must stay at the cap after eviction")
        self.assertNotIn(oldest_key, _compression_cache,
                         "oldest entry must have been evicted")
        self.assertIn(new_key, _compression_cache,
                      "newly inserted entry must be present")

    def test_clear_compression_cache_empties_the_cache(self):
        import local_compress_lib as L
        # Put something in the cache.
        key = _compression_cache_key("text", "focus", "model", None, False, False)
        L._cache_put(key, "result")
        self.assertEqual(len(_compression_cache), 1, "precondition: cache must be non-empty")

        clear_compression_cache()
        self.assertEqual(len(_compression_cache), 0, "cache must be empty after clearing")


class WholeTxtMapReducePath(unittest.TestCase):
    """Regression pins for the ordinary whole-text map-reduce compress() path
    (issue #40): chunking, classify_relevant integration, the chars_limited /
    non_selective bypass, and append_trailing_summary_if_missing.

    These complement the ConcurrentClassification tests (which pin ordering
    and the classify-all bypass for non_selective) by covering the
    remaining control-flow branches in the same pipeline:

    1. chars_limited bypass -- when max_chars is set AND the text is long
       enough to go through the full pipeline, classify_relevant must be
       skipped.  The existing tests patch around the bypass condition; this
       pins the branch explicitly.

    2. "All chunks irrelevant" error path -- when a selective focus causes
       every chunk to be classified NOT relevant, compress() must return an
       Error: string naming the chunk count and focus rather than silently
       returning an empty result.

    3. append_trailing_summary_if_missing integration -- the unconditional
       trailing-summary restoration step runs AFTER the model call; this
       pins that the final result actually contains the restored line when
       the model dropped it.

    All tests use stub injection (same pattern as ConcurrentClassification
    and SectionCompressionEndToEnd) -- no LM Studio needed.
    """

    def setUp(self):
        # Clear the cache so prior-test results never bypass the stubs.
        clear_compression_cache()

    def tearDown(self):
        clear_compression_cache()

    def _run_compress(self, text, focus, fake_classify, fake_complete,
                      skip_if_under_chars=0, max_chars=None):
        import asyncio
        import local_compress_lib as L

        real_classify = L.classify_relevant
        real_complete = L.complete
        real_resolve = L.resolve_model
        real_client = L.client

        L.classify_relevant = fake_classify
        L.complete = fake_complete
        L.resolve_model = lambda m, b: ("stub-model", None)
        L.client = lambda b: object()
        try:
            return asyncio.run(L.compress(
                text,
                focus=focus,
                skip_if_under_chars=skip_if_under_chars,
                max_chars=max_chars,
                chunk_chars=len(text) // 2 + 1,  # two chunks for a 2-unit text
            ))
        finally:
            L.classify_relevant = real_classify
            L.complete = real_complete
            L.resolve_model = real_resolve
            L.client = real_client

    def test_chars_limited_bypasses_classify_relevant(self):
        # When max_chars is set and the text is long enough to compress,
        # every chunk is treated as relevant by definition (chars_limited=True)
        # and classify_relevant must never be called -- asking it to re-judge
        # content already deterministically scoped adds pure risk.
        classify_called = []

        def fake_classify(*a, **k):
            classify_called.append(True)
            return True

        # A selective focus that would normally trigger classification --
        # but max_chars scopes the window, so the bypass must fire instead.
        # Text is long enough to exceed skip_if_under_chars and to produce
        # at least one real chunk via chunk_chars=len(text)//2+1 above.
        text = "A" * 200
        result = self._run_compress(
            text,
            focus="chocolate cake recipes",  # selective, never in text
            fake_classify=fake_classify,
            fake_complete=lambda *a, **k: "summary",
            skip_if_under_chars=0,
            max_chars=100,  # scopes to the first 100 chars
        )
        self.assertEqual(
            classify_called, [],
            "classify_relevant must not be called when max_chars scopes the input",
        )
        # The pipeline must have produced a real result (not an error), confirming
        # the bypass correctly treated the chunk as relevant and passed it through.
        self.assertIn("[compressed", result,
                      "chars_limited bypass must still produce a compressed result")

    def test_all_chunks_irrelevant_returns_error(self):
        # When a selective focus causes every chunk to be classified NOT
        # relevant, compress() must return an Error: string rather than
        # silently producing an empty result. The error must name both the
        # chunk count and the focus so the caller can report what happened.
        text = "A" * 100 + " " + "B" * 100  # two chunks via chunk_chars

        def fake_classify(*a, **k):
            return False  # every chunk is irrelevant

        result = self._run_compress(
            text,
            focus="chocolate cake recipes",
            fake_classify=fake_classify,
            fake_complete=lambda *a, **k: "summary",
            skip_if_under_chars=0,
        )
        self.assertTrue(
            result.startswith("Error:"),
            f"all-irrelevant result must be an Error string, got: {result!r}",
        )
        self.assertIn(
            "chunk(s)", result,
            "the error must state how many chunks were checked",
        )
        self.assertIn(
            "chocolate cake recipes", result,
            "the error must name the focus so the caller can adjust it",
        )

    def test_trailing_summary_line_restored_when_dropped_by_model(self):
        # append_trailing_summary_if_missing runs unconditionally after the
        # model call and restores a recognized test-result line if the model
        # dropped it. This pins the integration at the compress() boundary:
        # the final returned result must contain the restored line, tagged
        # with [RESULT], even though the stub model's return value omitted it.
        trailing = "3 failed, 39 passed in 8.72s"
        text = "some build output\n" + trailing  # source ends with the result line

        def fake_complete(client, model, system, content):
            # Simulate a model that tightened the prose but dropped the
            # final tally -- the exact failure mode documented in the
            # compress() docstring and that motivated this guard.
            return "some compressed output"  # trailing line intentionally omitted

        result = self._run_compress(
            text,
            focus="summarize this",
            fake_classify=lambda *a, **k: True,  # not used (non_selective focus)
            fake_complete=fake_complete,
            skip_if_under_chars=0,
        )
        self.assertIn(
            trailing, result,
            "the trailing result line dropped by the model must be restored by compress()",
        )
        self.assertIn(
            "[RESULT]", result,
            "the restored line must be tagged so it is distinguishable from model output",
        )


if __name__ == "__main__":
    unittest.main()
