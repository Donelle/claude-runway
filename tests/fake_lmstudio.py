"""A fake of LM Studio's REST API for tests: the `http` callable that
`libs/lmstudio_model_loader_lib.ModelLoader` takes. Keeps a set of loaded
models and records every call, so a test can assert exactly which loads and
unloads happened. No network.
"""


class FakeLmStudio:
    """Records every call and keeps a set of loaded models, like LM Studio."""

    def __init__(self, loaded=(), fail_on=None, capacity=None):
        self.loaded = {m: m for m in loaded}  # model id -> instance id
        # Most models LM Studio will hold at once (None: no limit); a load past
        # it fails the way a machine that is out of memory does.
        self.capacity = capacity
        self.calls: list = []
        self.fail_on = fail_on or {}
        self._counter = 0

    def __call__(self, method, url, body, timeout):
        self.calls.append((method, url, body))
        for fragment, exc in self.fail_on.items():
            if fragment in url:
                raise exc
        if url.endswith("/api/v0/models"):
            return {
                "data": [{"id": i, "state": "loaded", "type": "llm"} for i in self.loaded]
                + [{"id": "downloaded-only", "state": "not-loaded", "type": "llm"}]
            }
        if url.endswith("/api/v1/models/load"):
            if self.capacity is not None and len(self.loaded) >= self.capacity:
                raise OSError(f"cannot load {body['model']}: out of memory with {sorted(self.loaded)} loaded")
            self._counter += 1
            instance = f"{body['model']}:{self._counter}"
            self.loaded[body["model"]] = instance
            return {"type": "llm", "instance_id": instance, "status": "loaded"}
        if url.endswith("/api/v1/models/unload"):
            for model, instance in list(self.loaded.items()):
                if instance == body["instance_id"]:
                    del self.loaded[model]
            return {"instance_id": body["instance_id"]}
        raise AssertionError(f"unexpected call {method} {url}")

    def posts(self, suffix):
        return [(u, b) for m, u, b in self.calls if m == "POST" and u.endswith(suffix)]
