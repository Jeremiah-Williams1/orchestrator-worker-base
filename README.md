# orchestrator-worker-base

Shared `run_worker()` loop for the inference orchestrator's Python
workers. Every worker repo (classification, regression, whatever's
next) installs this instead of duplicating Redis/BRPOP/Pushgateway
plumbing.

## What lives here vs. what lives in each worker repo

**Here (`base.py`):** Redis connection, the `BRPOP` polling loop, job
status writes (`processing` → `completed`/`failed`), Prometheus counter
+ histogram definitions, the Pushgateway push, and error handling for
all of the above -- malformed JSON, missing fields, predict_fn
exceptions, Redis hiccups, Pushgateway outages. None of this is
model-specific.

**In each worker repo:** `predict.py` (model loading + the actual
inference logic) and a 5-line `worker.py` that imports `run_worker` and
calls it with a `job_type` and a `predict_fn`. That's the entire
interface — see "Contract" below.

## Install

In a worker repo's `requirements.txt`:

```
git+https://github.com/<you>/orchestrator-worker-base.git
```

**Pin to a tag or commit, not a moving branch**, once this stabilizes —
`@main` means every `docker build` could silently pull in a change to
shared code across *all* worker repos at once, which is a bad thing to
discover during a cloud migration. e.g.:

```
git+https://github.com/<you>/orchestrator-worker-base.git@v0.1.0
```

## Contract

```python
def run_worker(job_type: str, predict_fn: Callable[[dict], dict]) -> None:
```

- `job_type` — a string like `"classification"` or `"regression"`.
  Determines the Redis queue key (`queue:<job_type>`), the Pushgateway
  job name (`<job_type>-worker`), and the `type` label on both metrics.
- `predict_fn` — takes the job's `input` dict, returns a `dict` (whatever
  shape makes sense for that model — `{"prediction": ...}`,
  `{"predicted_mpg": ...}`, etc.). **Any exception it raises is caught
  by `run_worker`** and turned into `{"status": "failed", "error": str(e)}`
  on the job — `predict_fn` does not need its own try/except for that.
  It should raise on bad input rather than silently returning a
  placeholder value; `base.py` is what decides what happens to the job
  if it does.

A worker repo's `worker.py` is then just:

```python
from base import run_worker
from predict import make_prediction

def predict_fn(input_data: dict) -> dict:
    return {"prediction": make_prediction(input_data)}

if __name__ == "__main__":
    run_worker(job_type="regression", predict_fn=predict_fn)
```

## What the loop tolerates without crashing

Worth knowing since `load-test.sh` is specifically designed to send this
stuff at the API:

- **Malformed JSON on the queue** — logged and skipped.
- **A job missing `input`** — marked `failed` with a clear error, `predict_fn` is never called.
- **A job with no `id`** — logged and skipped (nothing to key a status write off of).
- **`predict_fn` raising anything** — job marked `failed`, the *worker process keeps running* (one bad job doesn't restart the pod).
- **Redis briefly unreachable during `BRPOP`** — logged, 5s backoff, retries — doesn't crash.
- **Pushgateway unreachable** — logged as a warning; the job's own success/failure is unaffected, only that one metrics push is lost.

What it does **not** currently tolerate gracefully: `REDIS_URL` being
unset entirely — that's intentional, it raises a `KeyError` immediately
at startup rather than silently defaulting to `localhost`, so a missing
env var in a Deployment/ConfigMap fails fast and visibly instead of the
pod looking "up" while talking to nothing.

## Testing changes to base.py

```bash
pip install -e . pytest
pytest
```

Tests mock both Redis and Pushgateway — nothing real needs to be
running. They cover the happy path plus every "tolerates without
crashing" case above. If you change `base.py`'s error handling, run
these before bumping the tag worker repos point at — a regression here
breaks every worker at once, not just one.

## Versioning

Bump the version in `pyproject.toml` and tag releases (`git tag v0.1.1`)
rather than relying on worker repos tracking `@main`. Each worker repo's
`requirements.txt` then pins to a specific tag, and upgrading a worker to
newer shared-loop behavior is a deliberate one-line change, not
something that happens automatically on the next `docker build`.
