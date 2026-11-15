"""
Loads and unloads LM Studio models for the `codebase-reviewer` server's
`loadOnDemand` reviewers, through LM Studio's REST API:

    POST {root}/api/v1/models/load     {"model": "<id>", "context_length": N}
    POST {root}/api/v1/models/unload   {"instance_id": "<id>"}
    GET  {root}/api/v0/models          (what is loaded right now)

`root` is the configured OpenAI-compatible base URL (`.../v1`) with that suffix
removed: the native REST API is served at the host root, not under `/v1`.

The rules this module enforces, and why:

  - It UNLOADS ONLY INSTANCES IT LOADED ITSELF. A model the user loaded (the
    compression model, say) is never unloaded, even when a reviewer names it
    with `loadOnDemand` true: if the model is already loaded when it is
    acquired, it is used as is, nothing is loaded, and it is not recorded as
    owned. Ownership is the instance id LM Studio returned for OUR load request.
  - `loadOnDemand` false means "must already be loaded": a model that is not is
    an error naming the model, and no load is attempted. The server never
    guesses that the user wanted it loaded.
  - A model that this loader loaded stays loaded across `acquire` calls until
    `release` or `release_all`, so a caller that holds it between two stages
    loads it once.

An optional `observer` (the run recorder of `reviewer_db_lib`) is told how long each load and unload
took and what LM Studio reports about the model. It is told on a best-effort basis: whatever it does
or raises, the load or unload goes on exactly as it would without one.

Standard library only; the HTTP call is injectable (`http`) so every rule above
is tested without a network or LM Studio.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

# Loading a large model takes minutes, so the load request gets a long budget;
# unloading and listing are quick. Constants, not settings: a load that needs
# longer than this is a machine that cannot run the reviewer anyway.
LOAD_TIMEOUT_SECONDS = 600.0
UNLOAD_TIMEOUT_SECONDS = 60.0
LIST_TIMEOUT_SECONDS = 15.0

# (method, url, json body or None, timeout seconds) -> parsed JSON response.
HttpFn = Callable[[str, str, Optional[dict], float], object]


class ModelLoadError(Exception):
    """A model could not be loaded, unloaded, or checked. The message names the model."""


class ModelNotLoadedError(ModelLoadError):
    """A reviewer's model is not loaded and the reviewer is not `loadOnDemand`."""


def _urllib_http(method: str, url: str, body: Optional[dict], timeout: float) -> object:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def rest_root(base_url: str) -> str:
    """The host root LM Studio's native REST API hangs off: `base_url` without
    a trailing slash or a trailing `/v1`."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def reviewer_api_url(base_url: str) -> str:
    """The URL the codebase-reviewer's completion client is built on: LM Studio's native REST API
    (`{root}/api/v0`) rather than the OpenAI-compatible `/v1`. The two answer chat completions in the same
    shape, but only the native one fills in the response's `stats` (time to first token, tokens per second,
    generation time, stop reason), which the run database records. The model loader above already needs this
    API, so a reviewer already requires it."""
    return rest_root(base_url) + "/api/v0"


def _http_error_detail(exc: BaseException) -> str:
    """The server's own message for an HTTP error (LM Studio puts the reason in
    the body), else the exception text."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        return f"HTTP {exc.code}" + (f": {body[:300]}" if body else "")
    return str(exc) or type(exc).__name__


class ModelLoader:
    def __init__(self, base_url: str, *, http: Optional[HttpFn] = None, observer: Optional[object] = None):
        self._root = rest_root(base_url)
        self._http: HttpFn = http or _urllib_http
        self.observer = observer
        # model id -> the instance id of the load THIS loader performed.
        self._owned: dict = {}

    # -- inspection ---------------------------------------------------------

    def loaded_model_ids(self) -> set:
        """The ids of the models LM Studio reports as loaded right now."""
        url = f"{self._root}/api/v0/models"
        try:
            payload = self._http("GET", url, None, LIST_TIMEOUT_SECONDS)
        except Exception as exc:
            raise ModelLoadError(
                f"cannot list LM Studio's loaded models at {url} ({_http_error_detail(exc)}); is LM Studio running?"
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ModelLoadError(f"LM Studio's model list at {url} has an unexpected shape")
        return {
            entry["id"]
            for entry in data
            if isinstance(entry, dict) and entry.get("state") == "loaded" and isinstance(entry.get("id"), str)
        }

    def model_details(self, model: str) -> Optional[dict]:
        """What LM Studio reports for `model` (id, publisher, arch, compatibility_type, quantization,
        max_context_length, loaded_context_length, state), or None when it cannot be asked or the model is
        not listed. Never raises: it only feeds a recorder."""
        try:
            payload = self._http("GET", f"{self._root}/api/v0/models", None, LIST_TIMEOUT_SECONDS)
            data = payload.get("data") if isinstance(payload, dict) else None
            for entry in data if isinstance(data, list) else []:
                if isinstance(entry, dict) and entry.get("id") == model:
                    return entry
        except Exception:
            pass
        return None

    def _tell(self, method: str, *args) -> None:
        """Calls `observer.<method>(*args)` if there is an observer; whatever it raises is swallowed."""
        observer = self.observer
        if observer is None:
            return
        try:
            getattr(observer, method)(*args)
        except Exception:
            pass

    @property
    def owned_models(self) -> list:
        """Models this loader loaded and has not yet unloaded, in load order."""
        return list(self._owned)

    def owns(self, model: str) -> bool:
        return model in self._owned

    # -- acquire / release --------------------------------------------------

    def acquire(self, model: str, *, load_on_demand: bool, context_length: Optional[int] = None) -> bool:
        """Makes `model` ready to serve. Returns True when this call loaded it.

        `load_on_demand` false: the model must already be loaded, else
        `ModelNotLoadedError` naming it (nothing is loaded). `load_on_demand`
        true: loaded here, with `context_length` when given, unless it is
        already loaded (by us earlier, or by the user), in which case it is
        used as is and, if the user loaded it, never unloaded by us."""
        if model in self._owned:
            return False
        loaded = self.loaded_model_ids()
        if model in loaded:
            if self.observer is not None:
                self._tell("model_seen", self.model_details(model))
            return False
        if not load_on_demand:
            raise ModelNotLoadedError(
                f"model {model!r} is not loaded in LM Studio, and its reviewer is not loadOnDemand: "
                "load it in LM Studio, or set loadOnDemand to true for the reviewers that use it"
            )
        body: dict = {"model": model}
        if context_length:
            body["context_length"] = context_length
        url = f"{self._root}/api/v1/models/load"
        started = time.time()
        try:
            response = self._http("POST", url, body, LOAD_TIMEOUT_SECONDS)
        except Exception as exc:
            # Read once: an HTTP error's body can be read only the first time.
            detail = _http_error_detail(exc)
            self._tell("model_loaded", model, started, time.time(), False, detail, None)
            raise ModelLoadError(f"could not load model {model!r} ({detail})") from exc
        ended = time.time()
        instance = response.get("instance_id") if isinstance(response, dict) else None
        # An LM Studio that does not echo the instance id: the model id is the
        # instance id of a model loaded under its default identifier.
        self._owned[model] = instance if isinstance(instance, str) and instance else model
        if self.observer is not None:
            self._tell("model_loaded", model, started, ended, True, None, self.model_details(model))
        return True

    def release(self, model: str) -> bool:
        """Unloads `model` if and only if this loader loaded it. Returns True
        when an unload request succeeded, False when there was nothing of ours
        to unload (never an error: not ours means leave it alone)."""
        instance = self._owned.get(model)
        if instance is None:
            return False
        url = f"{self._root}/api/v1/models/unload"
        started = time.time()
        try:
            self._http("POST", url, {"instance_id": instance}, UNLOAD_TIMEOUT_SECONDS)
        except Exception as exc:
            detail = _http_error_detail(exc)
            self._tell("model_unloaded", model, started, time.time(), False, detail)
            # Still ours, so a later release_all can retry it.
            raise ModelLoadError(f"could not unload model {model!r} ({detail})") from exc
        self._tell("model_unloaded", model, started, time.time(), True, None)
        del self._owned[model]
        return True

    def release_all(self) -> list:
        """Unloads everything this loader loaded. Tries every model even if one
        fails; returns the error messages (empty when all went well)."""
        errors = []
        for model in list(self._owned):
            try:
                self.release(model)
            except ModelLoadError as exc:
                errors.append(str(exc))
        return errors
