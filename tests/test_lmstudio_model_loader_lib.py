#!/usr/bin/env python3
"""Tests for libs/lmstudio_model_loader_lib.py. The HTTP call is injected, so
there is no network and no LM Studio:

    .venv/bin/python -m unittest discover -s tests
"""

import io
import os
import sys
import unittest
import urllib.error
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(REPO_ROOT / "libs"))

sys.path.insert(0, str(REPO_ROOT / "tests"))

from fake_lmstudio import FakeLmStudio  # noqa: E402
from lmstudio_model_loader_lib import (  # noqa: E402
    ModelLoader,
    ModelLoadError,
    ModelNotLoadedError,
    rest_root,
)

BASE = "http://localhost:1234/v1"


class RestRoot(unittest.TestCase):
    def test_strips_a_trailing_v1_and_slash(self):
        self.assertEqual(rest_root("http://h:1234/v1"), "http://h:1234")
        self.assertEqual(rest_root("http://h:1234/v1/"), "http://h:1234")
        self.assertEqual(rest_root("http://h:1234"), "http://h:1234")


class LoadedModels(unittest.TestCase):
    def test_lists_only_models_in_the_loaded_state(self):
        loader = ModelLoader(BASE, http=FakeLmStudio(loaded=["a", "b"]))
        self.assertEqual(loader.loaded_model_ids(), {"a", "b"})

    def test_uses_the_native_endpoint_at_the_host_root(self):
        fake = FakeLmStudio()
        ModelLoader(BASE, http=fake).loaded_model_ids()
        self.assertEqual(fake.calls[0][:2], ("GET", "http://localhost:1234/api/v0/models"))

    def test_an_unreachable_server_is_a_load_error(self):
        fake = FakeLmStudio(fail_on={"/api/v0/models": ConnectionRefusedError("refused")})
        with self.assertRaises(ModelLoadError) as ctx:
            ModelLoader(BASE, http=fake).loaded_model_ids()
        self.assertIn("is LM Studio running", str(ctx.exception))

    def test_an_unexpected_shape_is_a_load_error(self):
        with self.assertRaises(ModelLoadError):
            ModelLoader(BASE, http=lambda *a: {"models": []}).loaded_model_ids()


class AcquireNotOnDemand(unittest.TestCase):
    def test_a_loaded_model_is_used_and_nothing_is_loaded(self):
        fake = FakeLmStudio(loaded=["a"])
        loader = ModelLoader(BASE, http=fake)
        self.assertFalse(loader.acquire("a", load_on_demand=False))
        self.assertEqual(fake.posts("/load"), [])
        self.assertEqual(loader.owned_models, [])

    def test_a_model_that_is_not_loaded_is_an_error_naming_it_and_loads_nothing(self):
        fake = FakeLmStudio(loaded=["other"])
        loader = ModelLoader(BASE, http=fake)
        with self.assertRaises(ModelNotLoadedError) as ctx:
            loader.acquire("missing-model", load_on_demand=False)
        self.assertIn("'missing-model'", str(ctx.exception))
        self.assertEqual(fake.posts("/load"), [])


class AcquireOnDemand(unittest.TestCase):
    def test_loads_with_the_context_length_and_records_ownership(self):
        fake = FakeLmStudio()
        loader = ModelLoader(BASE, http=fake)
        self.assertTrue(loader.acquire("m", load_on_demand=True, context_length=32768))
        self.assertEqual(
            fake.posts("/api/v1/models/load"),
            [("http://localhost:1234/api/v1/models/load", {"model": "m", "context_length": 32768})],
        )
        self.assertTrue(loader.owns("m"))
        self.assertEqual(loader.owned_models, ["m"])

    def test_no_context_length_is_sent_when_none_is_given(self):
        fake = FakeLmStudio()
        ModelLoader(BASE, http=fake).acquire("m", load_on_demand=True)
        self.assertEqual(fake.posts("/load")[0][1], {"model": "m"})

    def test_a_model_the_user_already_loaded_is_not_loaded_again_or_owned(self):
        fake = FakeLmStudio(loaded=["m"])
        loader = ModelLoader(BASE, http=fake)
        self.assertFalse(loader.acquire("m", load_on_demand=True, context_length=4096))
        self.assertEqual(fake.posts("/load"), [])
        self.assertFalse(loader.owns("m"))

    def test_acquiring_a_model_it_loaded_itself_does_not_load_twice(self):
        fake = FakeLmStudio()
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("m", load_on_demand=True)
        self.assertFalse(loader.acquire("m", load_on_demand=True))
        self.assertEqual(len(fake.posts("/load")), 1)

    def test_a_failed_load_names_the_model_and_the_servers_reason(self):
        body = io.BytesIO(b'{"error": "not enough memory"}')
        err = urllib.error.HTTPError("u", 400, "Bad Request", {}, body)  # type: ignore[arg-type]
        fake = FakeLmStudio(fail_on={"/models/load": err})
        loader = ModelLoader(BASE, http=fake)
        with self.assertRaises(ModelLoadError) as ctx:
            loader.acquire("big", load_on_demand=True)
        self.assertIn("'big'", str(ctx.exception))
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("not enough memory", str(ctx.exception))
        self.assertFalse(loader.owns("big"))

    def test_an_lm_studio_that_does_not_echo_an_instance_id_falls_back_to_the_model_id(self):
        calls: list = []

        def http(method, url, body, timeout):
            calls.append((method, url, body))
            if url.endswith("/api/v0/models"):
                return {"data": []}
            return {}

        loader = ModelLoader(BASE, http=http)
        loader.acquire("m", load_on_demand=True)
        loader.release("m")
        self.assertEqual(calls[-1][2], {"instance_id": "m"})


class Release(unittest.TestCase):
    def test_unloads_the_instance_it_loaded(self):
        fake = FakeLmStudio()
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("m", load_on_demand=True)
        self.assertTrue(loader.release("m"))
        self.assertEqual(fake.posts("/unload"), [("http://localhost:1234/api/v1/models/unload", {"instance_id": "m:1"})])
        self.assertFalse(loader.owns("m"))
        self.assertEqual(fake.loaded, {})

    def test_never_unloads_a_model_the_user_loaded(self):
        fake = FakeLmStudio(loaded=["user-model"])
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("user-model", load_on_demand=True)
        self.assertFalse(loader.release("user-model"))
        self.assertEqual(fake.posts("/unload"), [])
        self.assertIn("user-model", fake.loaded)

    def test_never_unloads_a_model_it_never_acquired(self):
        fake = FakeLmStudio(loaded=["x"])
        loader = ModelLoader(BASE, http=fake)
        self.assertFalse(loader.release("x"))
        self.assertEqual(fake.calls, [])

    def test_a_failed_unload_raises_and_keeps_ownership_so_it_can_be_retried(self):
        fake = FakeLmStudio(fail_on={"/models/unload": OSError("boom")})
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("m", load_on_demand=True)
        with self.assertRaises(ModelLoadError) as ctx:
            loader.release("m")
        self.assertIn("'m'", str(ctx.exception))
        self.assertTrue(loader.owns("m"))

    def test_release_all_unloads_only_what_it_loaded_and_tries_every_model(self):
        fake = FakeLmStudio(loaded=["user-model"])
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("user-model", load_on_demand=True)
        loader.acquire("a", load_on_demand=True)
        loader.acquire("b", load_on_demand=True)
        self.assertEqual(loader.release_all(), [])
        unloaded = [b["instance_id"] for _, b in fake.posts("/unload")]
        self.assertEqual(sorted(unloaded), ["a:1", "b:2"])
        self.assertEqual(set(fake.loaded), {"user-model"})

    def test_release_all_reports_failures_but_still_tries_the_rest(self):
        fake = FakeLmStudio(fail_on={"/models/unload": OSError("boom")})
        loader = ModelLoader(BASE, http=fake)
        loader.acquire("a", load_on_demand=True)
        loader.acquire("b", load_on_demand=True)
        errors = loader.release_all()
        self.assertEqual(len(errors), 2)
        self.assertEqual(len(fake.posts("/unload")), 2)


if __name__ == "__main__":
    unittest.main()
