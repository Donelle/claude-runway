#!/usr/bin/env python3
"""Tests for tools/compress_mcp_server.py.

Stdlib-only (unittest, no pytest) and no network -- needs neither a live model
nor a real database:

    .venv/bin/python -m unittest discover -s tests

Originally written for issue #22 (the startup schema-overhead estimate
enumerated a hardcoded tool tuple that silently went stale); that estimate is
gone (issue #439) and RegisteredToolSet keeps the live-enumeration guard.
"""

import codecs
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _load_compress_mcp_server():
    """
    Loads a FRESH copy of the module per call (its own MCPServer instance, its
    own registered tools) so tests that register an extra ad-hoc tool can't
    leak that tool into other tests. savings_ledger is still the one shared
    module instance every load imports (Python caches by module name), which
    is what makes mock.patch.object(mod.savings_ledger, "set_meta", ...)
    reliably patch the real thing regardless of which load called it.
    """
    spec = importlib.util.spec_from_file_location(
        "compress_mcp_server", REPO_ROOT / "tools" / "compress_mcp_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(coro):
    import asyncio
    return asyncio.run(coro)


class RegisteredToolSet(unittest.TestCase):
    """Pins the server's registered tool set and checks the shared live count
    sees a tool added later. This used to be tested through
    _log_fixed_overhead() (issue #22: a hardcoded 5-tool tuple silently went
    stale); that startup estimate was removed in issue #439, but the failure
    mode it guarded -- a tool list that drifts from what is really registered --
    is still real for tools/report_tool_counts.py, which uses the same
    libs/mcp_tool_introspect.py enumeration."""

    def test_the_registered_tool_set_is_the_documented_one(self):
        mod = _load_compress_mcp_server()
        tool_names = {info.name for info in mod.mcp._tool_manager.list_tools()}
        # Pinned to the module's actual current tool set, not just a count, so a
        # rename/removal is caught too, not just a count drift. The three
        # conversation-compact tools live on the qdrant server now (see
        # tests/test_qdrant_mcp_server.py), and this server must not serve them
        # a second time under a different prefix.
        self.assertEqual(
            tool_names,
            {
                "list_local_models",
                "compress_file",
                "compress_command_output",
                "fetch_url",
                "compress_text",
                "savings_summary",
                "savings_detail",
                "savings_trend",
                "get_metrics",
                "record_metric",
            },
        )

    def test_new_tool_is_picked_up_without_code_changes(self):
        import mcp_tool_introspect

        mod = _load_compress_mcp_server()
        before = mcp_tool_introspect.tool_count(mod.mcp)

        @mod.mcp.tool()
        def _newly_added_probe_tool() -> str:
            """A tool added after the enumeration code was last edited."""
            return "x"

        self.assertEqual(mcp_tool_introspect.tool_count(mod.mcp), before + 1)
        self.assertIn("_newly_added_probe_tool", mcp_tool_introspect.describe_tools(mod.mcp))

    def test_no_schema_overhead_estimate_is_computed_or_stored(self):
        # Issue #439: the chars / 3.5 estimate and its meta row are gone.
        mod = _load_compress_mcp_server()
        self.assertFalse(hasattr(mod, "_log_fixed_overhead"))
        self.assertFalse(hasattr(mod.savings_ledger, "get_schema_overhead_tokens"))


class CompressCommandOutputDefaultsToRedaction(unittest.TestCase):
    """Regression coverage for issue #46: compress_command_output never
    exposed preserve_identifiers or preserve_sections, so its call into the
    shared compress() pipeline always used both flags' defaults
    (preserve_identifiers=False, preserve_sections=False) -- and
    redact_and_disclose() (libs/local_compress_lib.py) only runs credential
    redaction when at least one of those two is True (PR #135 review, round
    4 -- initially gated on preserve_identifiers alone, widened to either
    flag once round 4 caught preserve_sections=True falling through to the
    same ungated whole-text path on unstructured input). That left the tool
    most likely to see a raw secret (curl output, printenv, config dumps)
    never redacting one.

    Mocks at the _compress_impl boundary (the local_compress_lib.compress()
    function, imported under that alias) rather than running a real LM
    Studio call -- same reasoning as this repo's other MCP-server tests: no
    network needed to verify what kwargs get passed through.
    """

    def _capture_compress_impl_kwargs(self, mod):
        captured = {}

        async def _fake_compress_impl(
            text, focus, skip_if_under_chars, chunk_chars, max_total_chars, max_chars,
            model, base_url, preserve_identifiers=False, preserve_sections=False, ctx=None,
            client=None,
        ):
            captured["client"] = client
            captured["preserve_identifiers"] = preserve_identifiers
            captured["preserve_sections"] = preserve_sections
            return "[compressed 10 -> 5 chars across 1 chunk(s), ~50% smaller]\n\nfake summary"

        mod._compress_impl = _fake_compress_impl
        return captured

    def test_default_passes_preserve_identifiers_true(self):
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        _run(mod.compress_command_output(command="echo hello"))
        self.assertTrue(captured["preserve_identifiers"])

    def test_the_server_builds_and_passes_the_model_client(self):
        # The compression library no longer builds a client; this server owns
        # it and hands it over on every call, built for that call's base_url.
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        sentinel = object()
        built = []
        mod.build_client = lambda base_url: built.append(base_url) or sentinel
        _run(mod.compress_command_output(command="echo hello", base_url="http://other:1/v1"))
        self.assertIs(captured["client"], sentinel)
        self.assertEqual(built, ["http://other:1/v1"])

    def test_caller_can_still_opt_out(self):
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        _run(mod.compress_command_output(command="echo hello", preserve_identifiers=False))
        self.assertFalse(captured["preserve_identifiers"])

    def test_preserve_sections_still_defaults_off_and_is_passed_through(self):
        # preserve_sections isn't part of issue #46's finding, but since this
        # tool now exposes both flags (matching compress_text's pattern),
        # confirm the second flag's own default/pass-through isn't broken by
        # the same change.
        mod = _load_compress_mcp_server()
        captured = self._capture_compress_impl_kwargs(mod)
        _run(mod.compress_command_output(command="echo hello"))
        self.assertFalse(captured["preserve_sections"])
        _run(mod.compress_command_output(command="echo hello", preserve_sections=True))
        self.assertTrue(captured["preserve_sections"])

    def test_redaction_actually_runs_end_to_end_with_the_real_pipeline(self):
        # Goes one level deeper than the kwarg-capture tests above: uses the
        # REAL local_compress_lib.compress() (not a fake) with a fake LM
        # Studio client, so this proves the default genuinely results in a
        # secret being redacted from compress_command_output's output, not
        # just that a flag was threaded through correctly. The server owns
        # the model client now (it builds one per call with build_client and
        # passes it to the library), so the fake is patched in at the
        # server's own `build_client` name.
        mod = _load_compress_mcp_server()
        secret = "ghp_" + ("a" * 36)
        # The secret comes from the command's STDOUT here, not its own
        # command line text, to keep this test focused on the compressed
        # BODY's redaction specifically -- the savings footer's "source"
        # field (a separate leak vector: the raw command line itself, which
        # is also now redacted -- see SavingsFooterRedactsSourceCredentials
        # below, PR #135 review round 3) is exercised by its own dedicated
        # test instead. Passing it via an env var keeps the command line
        # itself secret-free while still producing the secret in stdout.
        os.environ["CLAUDE_RUNWAY_TEST_SECRET"] = secret
        self.addCleanup(os.environ.pop, "CLAUDE_RUNWAY_TEST_SECRET", None)

        class _FakeChoice:
            def __init__(self, content):
                self.message = mock.Mock(content=content)

        class _FakeCompletion:
            def __init__(self, content):
                self.choices = [_FakeChoice(content)]

        class _FakeChatCompletions:
            def create(self, model, messages, temperature=0):
                # Echo the secret back, simulating a model that reproduces a
                # credential-shaped value verbatim from its input.
                return _FakeCompletion(f"Ran a command. Token seen: {secret}")

        class _FakeChat:
            def __init__(self):
                self.completions = _FakeChatCompletions()

        class _FakeClient:
            def __init__(self):
                self.chat = _FakeChat()

        with mock.patch.object(mod, "build_client", lambda base_url: _FakeClient()):
            result = _run(mod.compress_command_output(
                command='printf "%s" "$CLAUDE_RUNWAY_TEST_SECRET"',
                skip_if_under_chars=0, model="fake-model",
            ))
        self.assertNotIn(secret, result)
        self.assertIn("credential-shaped value(s) redacted", result)

    def test_redaction_still_runs_under_the_real_default_threshold(self):
        # PR #135 review (Copilot): the end-to-end test above used
        # skip_if_under_chars=0 specifically to route through the main
        # compression pipeline, which masked a real gap -- command output
        # shorter than the DEFAULT 2000-char threshold hit compress()'s
        # early return, which used to hand back the original text with no
        # redaction at all, regardless of preserve_identifiers. Fixed in
        # local_compress_lib.compress() (the early-return branch now
        # redacts when preserve_identifiers is set); this test exercises
        # compress_command_output's actual DEFAULT skip_if_under_chars
        # (2000) with a short secret-bearing command, which is the common
        # case for real command output. No LM Studio mock needed here --
        # the early-return path returns before resolve_model/client are
        # ever touched.
        mod = _load_compress_mcp_server()
        secret = "ghp_" + ("a" * 36)
        os.environ["CLAUDE_RUNWAY_TEST_SECRET"] = secret
        self.addCleanup(os.environ.pop, "CLAUDE_RUNWAY_TEST_SECRET", None)

        # Read the secret through the current interpreter rather than shell
        # expansion (`printf "%s" "$VAR"`): compress_command_output runs its
        # command with shell=True, which is cmd.exe on Windows, where "$VAR"
        # is never expanded -- the command would print the literal variable
        # name and this test would fail for a reason unrelated to redaction.
        result = _run(mod.compress_command_output(
            command=f'"{sys.executable}" -c "import os; print(os.environ[\'CLAUDE_RUNWAY_TEST_SECRET\'])"',
        ))
        self.assertNotIn(secret, result)
        self.assertIn("credential-shaped value(s) redacted", result)


class SavingsFooterRedactsSourceCredentials(unittest.TestCase):
    """Regression coverage for PR #135 review, round 3 (Copilot):
    _append_savings_footer's "source" field echoed the raw caller-supplied
    value (compress_command_output passes the actual shell command)
    verbatim and unredacted into the savings footer -- so a command like
    `curl -H 'Authorization: Bearer ghp_...'` still leaked its embedded
    credential via that field even though the compressed BODY the command
    produced was properly redacted. That footer gets persisted into the
    savings ledger by hooks/compress_output.py, turning a transient
    secret in a command line into a durable one in that database.
    """

    def test_source_credential_is_redacted_in_the_footer(self):
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        secret = "sk-ant-" + ("b" * 40)
        footer_json = mod._append_savings_footer(
            outer_text="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            inner_result="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            tool="compress_command_output", raw_text="raw", credited=True,
            source=f"curl -H 'Authorization: Bearer {secret}' https://example.com",
        )
        self.assertNotIn(secret, footer_json)
        self.assertIn("REDACTED-CREDENTIAL", footer_json)

    def test_clean_source_is_unaffected(self):
        # Confirms the fix doesn't mangle an ordinary command with no
        # credential in it -- redact_credentials() is a no-op when there's
        # nothing to redact.
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        result = mod._append_savings_footer(
            outer_text="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            inner_result="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            tool="compress_command_output", raw_text="raw", credited=True,
            source="pytest -q",
        )
        self.assertIn('"source": "pytest -q"', result)

    def test_credential_straddling_the_200_char_truncation_boundary_is_still_caught(self):
        # Redaction must run over the FULL source before truncating to 200
        # chars -- truncating first could split a credential in half,
        # leaving an unmatched (and un-redacted) fragment on one side of
        # the cut.
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        secret = "ghp_" + ("c" * 36)
        padding = "x" * 180
        source = f"{padding} {secret}"  # secret starts well past char 180
        self.assertGreater(len(source), 200)
        result = mod._append_savings_footer(
            outer_text="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            inner_result="[compressed 100 -> 10 chars across 1 chunk(s), ~90% smaller]\n\nsummary",
            tool="compress_command_output", raw_text="raw", credited=True,
            source=source,
        )
        self.assertNotIn(secret, result)


class SavingsFooterCarriesExactChars(unittest.TestCase):
    """Issues #437/#438: the footer carries exact character counts of raw_text and
    the returned outer_text, and no longer the chars / 3.5 token estimates."""

    def test_footer_has_exact_char_counts_and_no_token_estimates(self):
        mod = _load_compress_mcp_server()
        mod.TRACK_SAVINGS = True
        raw_text = "r" * 701
        outer_text = "[compressed 701 -> 10 chars across 1 chunk(s), ~99% smaller]\n\nsummary"
        result = mod._append_savings_footer(
            outer_text=outer_text, inner_result=outer_text,
            tool="compress_file", raw_text=raw_text, credited=True, source="f.log",
        )
        footer = json.loads(result.split("<!--CLAUDE_RUNWAY_SAVINGS:")[1].rsplit("-->", 1)[0])
        self.assertEqual(footer["raw_chars"], 701)
        self.assertEqual(footer["out_chars"], len(outer_text))
        self.assertNotIn("raw_tokens", footer)
        self.assertNotIn("out_tokens", footer)


class _FakeStreamedResponse:
    """Stand-in for requests.Response in stream=True mode -- exposes just
    the surface fetch_url actually touches (.headers, .raise_for_status(),
    .iter_content(), .close(), .encoding, .apparent_encoding, ._content, and
    the context-manager protocol fetch_url now uses via
    `with requests.get(...) as resp:` -- PR #124 review) so no real network
    call happens. `body_chunks` is a list of byte strings handed back one at
    a time from .iter_content(), simulating a streamed download regardless
    of what Content-Length (if any) claimed up front. `raise_in_iter`, if
    set, is raised partway through iteration to simulate a broken chunked
    response / read timeout -- the case that exposed the pre-fix connection
    leak. `apparent_encoding_result` fakes what real requests derives via
    chardet/charset_normalizer over `self._content` -- fetch_url populates
    `_content` itself (mirroring how it uses the real attribute) before
    reading this, so this fake doesn't need to run real detection logic.
    """

    def __init__(
        self, body_chunks, content_length=None, encoding="utf-8", status_ok=True,
        raise_in_iter=None, apparent_encoding_result="utf-8", content_type=None,
    ):
        self._body_chunks = body_chunks
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}
        # `content_type`, if set, is the raw Content-Type header -- fetch_url
        # inspects it to tell an EXPLICIT charset parameter apart from the
        # RFC 2616 ISO-8859-1 default requests reports for bare text/*.
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self.encoding = encoding
        self._status_ok = status_ok
        self._raise_in_iter = raise_in_iter
        self._apparent_encoding_result = apparent_encoding_result
        self._content = False
        self.closed = False

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("simulated HTTP error")

    def iter_content(self, chunk_size=65536):
        for chunk in self._body_chunks:
            yield chunk
        if self._raise_in_iter is not None:
            raise self._raise_in_iter

    @property
    def apparent_encoding(self):
        return self._apparent_encoding_result

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CompressFileDecodesWithoutSilentCorruption(unittest.TestCase):
    """Regression coverage for issue #267: compress_file read files with
    errors="ignore", silently dropping non-UTF-8 bytes (cp1252 accents) and
    letting UTF-16 arrive NUL-interleaved."""

    def _compress_file_text(self, data: bytes) -> str:
        """Run compress_file on `data` and return the exact text handed to
        _compress (stubbed, so no LM Studio is needed)."""
        mod = _load_compress_mcp_server()
        seen = {}

        async def fake_compress(text, *args, **kwargs):
            seen["text"] = text
            return "summary"

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.log"
            path.write_bytes(data)
            with mock.patch.object(mod, "_compress", fake_compress):
                _run(mod.compress_file(file_path=str(path)))
        return seen["text"]

    def test_cp1252_bytes_are_visible_replacements_not_dropped(self):
        text = self._compress_file_text("caf\u00e9 \u201cquoted\u201d".encode("cp1252"))
        self.assertIn("\ufffd", text)
        self.assertTrue(text.startswith("caf"))
        # The ASCII around each bad byte survives, and nothing was deleted:
        # 1 char per undecodable byte (3 bad bytes: e-acute and two quotes).
        self.assertEqual(text.count("\ufffd"), 3)
        self.assertIn("quoted", text)

    def test_utf16le_with_bom_decodes_cleanly(self):
        text = self._compress_file_text("hello wor\u00e9ld".encode("utf-16"))
        self.assertEqual(text, "hello wor\u00e9ld")
        self.assertNotIn("\x00", text)

    def test_utf16be_with_bom_decodes_cleanly(self):
        data = codecs.BOM_UTF16_BE + "abc\u00e9".encode("utf-16-be")
        self.assertEqual(self._compress_file_text(data), "abc\u00e9")

    def test_utf32_with_bom_is_not_mistaken_for_utf16(self):
        self.assertEqual(self._compress_file_text("abc".encode("utf-32")), "abc")

    def test_utf8_bom_is_stripped(self):
        self.assertEqual(self._compress_file_text(codecs.BOM_UTF8 + b"plain"), "plain")

    def test_plain_utf8_is_unchanged(self):
        self.assertEqual(self._compress_file_text("na\u00efve \u2713".encode("utf-8")), "na\u00efve \u2713")


class FetchUrlEnforcesSizeCap(unittest.TestCase):
    """Regression coverage for issue #39: fetch_url used to call
    requests.get() with no stream=True, no Content-Length check, and no
    size cap of any kind -- resp.text/resp.content fully materialized the
    body into memory unconditionally before max_total_chars (a
    post-extraction cap) ever got a chance to run.
    """

    def test_declared_content_length_over_cap_is_refused_without_reading_body(self):
        mod = _load_compress_mcp_server()
        fake_resp = _FakeStreamedResponse(
            body_chunks=[b"<html>should never be read</html>"],
            content_length=50 * 1024 * 1024,  # 50 MB, over the 10 MB default cap
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/huge-file"))

        self.assertIn("Error fetching", result)
        self.assertIn("Content-Length", result)
        self.assertTrue(fake_resp.closed)

    def test_missing_content_length_still_aborts_once_stream_exceeds_cap(self):
        """The case the issue explicitly calls out as the normal one: a
        server that never sends (or understates) Content-Length -- the
        incremental byte count during iter_content() is what has to catch
        this, since there's no header to check up front."""
        mod = _load_compress_mcp_server()
        oversized_chunk = b"x" * (11 * 1024 * 1024)  # 11 MB, over the 10 MB default cap
        fake_resp = _FakeStreamedResponse(
            body_chunks=[oversized_chunk],
            content_length=None,
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/no-content-length"))

        self.assertIn("Error fetching", result)
        self.assertIn("exceeded", result)
        self.assertTrue(fake_resp.closed)

    def test_response_under_cap_is_fetched_and_extracted_normally(self):
        mod = _load_compress_mcp_server()
        html = b"<html><body><p>Small page, well under any size cap.</p></body></html>"
        fake_resp = _FakeStreamedResponse(body_chunks=[html], content_length=len(html))
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/small-page"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("Small page", result)
        # PR #124 review fix: fetch_url now uses `with requests.get(...) as
        # resp:`, so the connection is released on every exit path,
        # including a normal successful fetch -- not just the early-return
        # error paths that used to call resp.close() explicitly.
        self.assertTrue(fake_resp.closed)

    def test_max_response_bytes_override_is_respected(self):
        """A caller-supplied max_response_bytes narrower than the default
        must be honored, not just the module-level default."""
        mod = _load_compress_mcp_server()
        body = b"<html>" + b"y" * 2000 + b"</html>"
        fake_resp = _FakeStreamedResponse(body_chunks=[body], content_length=len(body))
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/medium-page", max_response_bytes=1000))

        self.assertIn("Error fetching", result)
        self.assertIn("Content-Length", result)

    def test_connection_is_closed_when_iteration_raises_mid_stream(self):
        """PR #124 review (Copilot): a bare `resp = requests.get(...)` left
        the connection open on any exception from iter_content() (a normal
        read timeout or broken chunked response) -- those exceptions used to
        propagate straight past every resp.close() call in the function.
        Reproduced directly against the pre-fix code (a fake response whose
        iter_content() raised mid-stream left .closed False); this pins the
        fix, which switched to `with requests.get(...) as resp:` so the
        connection is released on any exit path, exception included.
        """
        mod = _load_compress_mcp_server()
        fake_resp = _FakeStreamedResponse(
            body_chunks=[b"partial data"],
            raise_in_iter=ConnectionError("simulated broken chunked response"),
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/flaky"))

        self.assertIn("Error fetching", result)
        self.assertTrue(fake_resp.closed)

    def test_invalid_server_declared_charset_falls_back_to_utf8_instead_of_crashing(self):
        """PR #124 review (Copilot): resp.encoding is derived from a
        server-controlled Content-Type header -- a value like
        `charset=invalid-bogus-charset` makes requests set resp.encoding to
        that literal (unvalidated) string. Decoding with an unknown codec
        name raises LookupError, which used to be unhandled here (reproduced
        directly: the pre-fix code raised LookupError instead of returning
        an error string). requests' own resp.text property specifically
        catches (LookupError, TypeError) around this same decode and falls
        back to UTF-8 -- this pins that same fallback.
        """
        mod = _load_compress_mcp_server()
        html = b"<html><body><p>Still readable as UTF-8 bytes.</p></body></html>"
        fake_resp = _FakeStreamedResponse(
            body_chunks=[html], content_length=len(html), encoding="invalid-bogus-charset",
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/bad-charset"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("Still readable", result)

    def test_headerless_response_uses_detected_encoding_not_a_blind_utf8_default(self):
        """PR #124 review round 2 (Copilot): when resp.encoding is None (no
        Content-Type charset declared at all), the old resp.text path fell
        back to apparent_encoding (chardet/charset_normalizer detection over
        the full body), not a blind UTF-8 assumption. Reproduced directly:
        a headerless Windows-1252-encoded page containing a curly
        single-quote (0x92) decoded to a bare U+FFFD replacement character
        under straight UTF-8 before this fix. Pins that fetch_url now
        populates resp._content with the already-downloaded bytes and reads
        resp.apparent_encoding instead of defaulting straight to UTF-8.
        """
        mod = _load_compress_mcp_server()
        # "It’s" (curly apostrophe, U+2019) encoded as cp1252 -- 0x92 is not
        # valid standalone UTF-8, so a straight UTF-8 decode with
        # errors="replace" loses it as U+FFFD.
        body = "It’s a legacy page.".encode("cp1252")
        fake_resp = _FakeStreamedResponse(
            body_chunks=[body],
            content_length=len(body),
            encoding=None,
            apparent_encoding_result="cp1252",
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/legacy-page"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("It’s a legacy page", result)
        # Confirms fetch_url populated the cache with OUR bytes (bounded by
        # max_response_bytes) rather than trying to re-read the network.
        self.assertEqual(fake_resp._content, body)

    def test_charsetless_text_html_is_not_decoded_as_latin1(self):
        """Issue #266: for a bare `Content-Type: text/html` (no charset
        parameter) requests sets resp.encoding to the RFC 2616 legacy default
        "ISO-8859-1" -- truthy, so the apparent_encoding detection (which
        only ran for a falsy encoding) was skipped and a UTF-8 page decoded
        as Latin-1 into mojibake without raising.
        """
        mod = _load_compress_mcp_server()
        body = "<html><body><p>Café — it’s “fine”, naïve résumé.</p></body></html>".encode("utf-8")
        fake_resp = _FakeStreamedResponse(
            body_chunks=[body],
            content_length=len(body),
            encoding="ISO-8859-1",
            content_type="text/html",
            apparent_encoding_result="utf-8",
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/utf8-no-charset"))

        self.assertNotIn("Error fetching", result)
        self.assertIn("Café", result)
        self.assertNotIn("Ã©", result)

    def test_explicit_iso_8859_1_charset_is_still_honored(self):
        """Issue #266: an EXPLICIT `charset=iso-8859-1` is the server's real
        declaration and must not be overridden by detection."""
        mod = _load_compress_mcp_server()
        body = "<html><body><p>Café au lait.</p></body></html>".encode("iso-8859-1")
        fake_resp = _FakeStreamedResponse(
            body_chunks=[body],
            content_length=len(body),
            encoding="ISO-8859-1",
            content_type="text/html; charset=iso-8859-1",
            # Deliberately wrong: if detection wrongly ran, this would win.
            apparent_encoding_result="utf-8",
        )
        with mock.patch.object(mod.requests, "get", return_value=fake_resp):
            result = _run(mod.fetch_url(url="https://example.com/latin1-explicit"))

        self.assertIn("Café au lait", result)


class RecordMetricAcceptsDictOrJsonStringMetadata(unittest.TestCase):
    """record_metric's `metadata` parameter was previously documented and
    typed as a JSON string only, but some MCP clients (confirmed live:
    Claude Code itself) silently coerce a string argument that happens to
    parse as a JSON object into a native dict before this tool ever sees
    it -- bypassing that declared `string` schema type entirely. The fix
    below widens the parameter to accept either shape directly, rather than
    erroring on the coerced-dict case with "metadata must be valid JSON" (a
    `TypeError` from `json.loads(dict)`, caught and misreported as a
    JSON-parse failure even though the value was never invalid to begin
    with)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_metadata_as_json_string_still_works(self):
        mod = _load_compress_mcp_server()
        result = mod.record_metric(
            "autowork", "ticket_merged", metadata='{"issue": 211, "pr": 260}'
        )
        self.assertEqual(result, "OK")
        rows = mod.metrics_lib.MetricsStore().detail("autowork")
        self.assertEqual(rows[0]["metadata"], {"issue": 211, "pr": 260})

    def test_metadata_as_already_parsed_dict_is_accepted(self):
        mod = _load_compress_mcp_server()
        result = mod.record_metric(
            "autowork", "ticket_merged", metadata={"issue": 211, "pr": 260}
        )
        self.assertEqual(result, "OK")
        rows = mod.metrics_lib.MetricsStore().detail("autowork")
        self.assertEqual(rows[0]["metadata"], {"issue": 211, "pr": 260})

    def test_metadata_as_list_is_still_rejected(self):
        mod = _load_compress_mcp_server()
        result = mod.record_metric("autowork", "ticket_merged", metadata='["not", "a", "dict"]')
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("must be a JSON object", result)

    def test_metadata_as_invalid_json_string_is_still_rejected(self):
        mod = _load_compress_mcp_server()
        result = mod.record_metric("autowork", "ticket_merged", metadata="{not valid json")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("must be valid JSON", result)

    def test_metadata_omitted_still_works(self):
        mod = _load_compress_mcp_server()
        result = mod.record_metric("autowork", "ticket_merged")
        self.assertEqual(result, "OK")


class GetMetricsSessionIdFilter(unittest.TestCase):
    """Issue #248: get_metrics gained an optional session_id passthrough for
    view="summary"/"by_event_type" -- covers the MCP tool surface on top of
    tests/test_metrics_lib.py's own MetricsStore-level coverage of the
    underlying filter."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_summary_view_narrows_to_session_id(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="summary", session_id="issue-248-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(data["event_count"], 1)
        self.assertEqual(data["total_value"], 1.0)

    def test_by_event_type_view_narrows_to_session_id(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_blocked", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="by_event_type", session_id="issue-248-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event_type"], "ticket_merged")

    def test_omitting_session_id_keeps_unfiltered_summary(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="summary", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(data["event_count"], 2)

    def test_trend_view_ignores_session_id_without_erroring(self):
        # session_id has no effect on trend() (out of scope per issue #248) --
        # passing it must not raise or otherwise break the trend view.
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-248-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="trend", bucket="day", session_id="issue-248-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        # Both rows count toward the trend regardless of session_id -- it's
        # simply not applied there.
        self.assertEqual(sum(r["event_count"] for r in data), 2)

    def test_summary_view_empty_filter_names_the_session_not_the_whole_metric(self):
        """Copilot review on PR #250 (round 4): a session_id filter that
        matches nothing must not claim the metric_id as a whole has no
        events -- it has plenty, just not under this session_id."""
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="summary", session_id="no-such-session", format="text")
        self.assertIn("no-such-session", result)
        self.assertNotIn("No events recorded yet for this metric_id.", result)

    def test_by_event_type_view_empty_filter_names_the_session_not_the_whole_metric(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="by_event_type", session_id="no-such-session", format="text")
        self.assertIn("no-such-session", result)
        self.assertNotIn("No events recorded yet for this metric_id.", result)


class GetMetricsDetailView(unittest.TestCase):
    """Issue #249: get_metrics gained a "detail" view -- raw per-event rows
    with deserialized metadata, most recent first. Covers the MCP tool
    surface on top of tests/test_metrics_lib.py's own MetricsStore-level
    coverage of the underlying detail()/format_detail_view()."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_detail_view_returns_rows_with_metadata(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, metadata={"model": "claude-sonnet-5", "rounds": 5})

        result = mod.get_metrics("autowork", view="detail", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["metadata"], {"model": "claude-sonnet-5", "rounds": 5})

    def test_n_is_reused_as_the_row_limit(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        for i in range(5):
            store.record("autowork", "x", value=1.0, event_timestamp=f"2026-09-0{i + 1}T00:00:00Z")

        result = mod.get_metrics("autowork", view="detail", n=2, format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["event_timestamp"], "2026-09-05T00:00:00Z")

    def test_detail_view_narrows_to_session_id(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-100000")
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-250-110000")

        result = mod.get_metrics("autowork", view="detail", session_id="issue-249-100000", format="json")
        import json as _json
        data = _json.loads(result)["data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "issue-249-100000")

    def test_detail_view_empty_filter_names_the_session_not_the_whole_metric(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0, session_id="issue-249-110000")

        result = mod.get_metrics("autowork", view="detail", session_id="no-such-session", format="text")
        self.assertIn("no-such-session", result)
        self.assertNotIn("No events recorded yet for this metric_id.", result)

    def test_unrecognised_view_error_lists_detail_as_valid(self):
        mod = _load_compress_mcp_server()
        result = mod.get_metrics("autowork", view="bogus")
        self.assertIn("detail", result)

    def test_direct_negative_limit_is_rejected_not_unbounded(self):
        # get_metrics's own n<1 check already prevents this via the MCP
        # surface -- this exercises detail()'s own independent guard (see
        # test_metrics_lib.py's Detail class for the direct-library-call
        # reproduction this backs).
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "x", value=1.0)
        with self.assertRaises(ValueError):
            store.detail("autowork", limit=-1)


class GetMetricsBucketAndNScopedToRelevantViews(unittest.TestCase):
    """Copilot review on PR #255: bucket/n are documented as "ignored" for
    views that don't consume them, but were validated unconditionally --
    an irrelevant bucket="month" or n=0 alongside view="summary" rejected
    an otherwise-valid call. Scoped so only the views that actually use
    each argument validate it."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "metrics.db")
        self._env_patch = mock.patch.dict(os.environ, {"CLAUDE_RUNWAY_METRICS_DB": self._db_path})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def test_summary_ignores_irrelevant_bad_bucket(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="summary", bucket="month", format="json")
        self.assertNotIn("Error:", result)

    def test_summary_ignores_irrelevant_bad_n(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="summary", n=0, format="json")
        self.assertNotIn("Error:", result)

    def test_by_event_type_ignores_irrelevant_bad_bucket(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="by_event_type", bucket="month", format="json")
        self.assertNotIn("Error:", result)

    def test_detail_ignores_irrelevant_bad_bucket(self):
        mod = _load_compress_mcp_server()
        store = mod.metrics_lib.MetricsStore()
        store.record("autowork", "ticket_merged", value=1.0)
        result = mod.get_metrics("autowork", view="detail", bucket="month", format="json")
        self.assertNotIn("Error:", result)

    def test_trend_still_validates_bucket(self):
        mod = _load_compress_mcp_server()
        result = mod.get_metrics("autowork", view="trend", bucket="month")
        self.assertIn("Error:", result)

    def test_trend_still_validates_n(self):
        mod = _load_compress_mcp_server()
        result = mod.get_metrics("autowork", view="trend", n=0)
        self.assertIn("Error:", result)

    def test_detail_still_validates_n_since_n_is_its_row_limit(self):
        mod = _load_compress_mcp_server()
        result = mod.get_metrics("autowork", view="detail", n=0)
        self.assertIn("Error:", result)


class ListLocalModelsReportsLoadedNotDownloaded(unittest.TestCase):
    """Issue #296: list_local_models used to report every model
    /v1/models returned -- every DOWNLOADED model, not just loaded ones
    (LM Studio's default is JIT loading) -- which is misleading when the
    output is used directly as a model= value that must name something
    actually loaded. It now goes through local_compress_lib.fetch_loaded_models,
    which prefers LM Studio's native /api/v0/models endpoint (true per-model
    load state) and falls back to /v1/models -- unchanged, same old
    limitation -- only when /api/v0 isn't available.

    Patches mod.fetch_loaded_models directly (imported into
    compress_mcp_server's own namespace from local_compress_lib) --
    fetch_loaded_models' own behavior against real HTTP responses is covered
    by test_local_compress_lib.py's FetchLoadedModelsTests; this file only
    needs to confirm list_local_models' own wording built on top of it.
    """

    def test_reports_loaded_chat_models_via_v0(self):
        mod = _load_compress_mcp_server()
        mod.fetch_loaded_models = lambda base_url: (["google/gemma-4-e4b"], "v0")
        result = mod.list_local_models()
        self.assertIn("Loaded chat models", result)
        self.assertIn("google/gemma-4-e4b", result)
        # Must not claim these are merely "available" -- v0 confirms load state.
        self.assertNotIn("load state unknown", result)

    def test_v1_fallback_does_not_claim_confirmed_load_state(self):
        mod = _load_compress_mcp_server()
        mod.fetch_loaded_models = lambda base_url: (["model-a", "model-b"], "v1")
        result = mod.list_local_models()
        self.assertIn("load state unknown", result)
        self.assertIn("model-a", result)
        self.assertNotIn("Loaded chat models", result)

    def test_no_loaded_models_reports_none_loaded(self):
        mod = _load_compress_mcp_server()
        mod.fetch_loaded_models = lambda base_url: ([], "v0")
        result = mod.list_local_models()
        self.assertIn("no model is loaded", result)

    def test_unreachable_reports_could_not_reach(self):
        mod = _load_compress_mcp_server()

        def _raise(base_url):
            raise ConnectionError("LM Studio is down")

        mod.fetch_loaded_models = _raise
        result = mod.list_local_models()
        self.assertIn("Could not reach LM Studio", result)


if __name__ == "__main__":
    unittest.main()
