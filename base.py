import json
import logging
import os
import time
from typing import Callable

import redis
from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

JOB_TTL = 3600


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe_default(obj):
    """
    Fallback for json.dumps() when predict_fn's return value contains
    something that isn't a plain JSON type. predict_fn authors should
    already convert numpy arrays/scalars themselves (.tolist() / float()
    / int()) -- this is a safety net for when that's forgotten, not a
    substitute for doing it properly. Without this, a forgotten cast in
    any worker's predict_fn would raise TypeError here and crash the
    whole worker loop, not just fail that one job.
    """
    if hasattr(obj, "tolist"):  # numpy arrays AND numpy scalars both have this
        return obj.tolist()
    if hasattr(obj, "item"):  # other numpy/array-like scalar fallback
        return obj.item()
    return str(obj)  # last resort -- never let serialization itself crash the loop


def _safe_set(r: "redis.Redis", job_key: str, job: dict) -> None:
    """
    Write job state back to Redis, but don't let a Redis hiccup -- or a
    non-JSON-serializable value sneaking into job["result"] -- crash the
    whole loop. The job has already been (or is about to be) processed;
    losing/degrading the status write is recoverable, crashing the
    worker process is not.
    """
    try:
        payload = json.dumps(job, default=_json_safe_default)
    except Exception as e:
        # _json_safe_default's str() fallback means this should be very
        # rare, but if job itself is somehow unserializable, log and give
        # up on this write rather than crash the loop.
        log.error(f"failed to serialize job {job_key} for status write: {e}")
        return

    try:
        r.set(job_key, payload, ex=JOB_TTL)
    except redis.exceptions.RedisError as e:
        log.error(f"failed to write status for {job_key}: {e}")


def run_worker(job_type: str, predict_fn: Callable[[dict], dict]) -> None:
    """
    job_type: 'classification', 'regression', etc -- determines the queue
              key (queue:<job_type>), metric labels, and Pushgateway job name.
    predict_fn: takes the job's input dict, returns a result dict.
                This is the ONLY thing each worker repo provides. Any
                exception it raises is caught here and turned into a
                'failed' job status -- predict_fn does not need its own
                try/except for that.
    """
    queue_key = f"queue:{job_type}"
    pushgateway_url = os.getenv("PUSHGATEWAY_URL", "http://pushgateway-svc:9091")
    redis_url = os.environ["REDIS_URL"]  # intentionally no default -- misconfiguring
    # this should fail loudly at startup, not silently connect to localhost.

    r = redis.from_url(redis_url, decode_responses=True)

    registry = CollectorRegistry()
    jobs_completed = Counter(
        "inference_jobs_completed_total", "Total jobs completed",
        ["type", "status"], registry=registry,
    )
    inference_duration = Histogram(
        "inference_duration_seconds", "Time spent running the ML model",
        ["type"], registry=registry,
    )

    log.info(f"{job_type} worker started, polling {queue_key}...")

    while True:
        try:
            result = r.brpop(queue_key, timeout=5)
        except redis.exceptions.RedisError as e:
            # Redis being briefly unreachable shouldn't take the whole
            # worker process down -- back off and retry rather than crash
            # (and get rescheduled by k8s into a possible crash loop).
            log.error(f"redis error during BRPOP: {e}; retrying in 5s")
            time.sleep(5)
            continue

        if result is None:
            continue  # timeout, no job -- normal, just poll again

        _, raw = result

        # --- Malformed input handling -------------------------------
        # The load-test script intentionally sends malformed JSON and
        # jobs with missing/bad fields. None of that should be able to
        # crash the loop -- log it and move on to the next job.
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as e:
            log.error(f"skipping malformed JSON on {queue_key}: {e} (raw={raw!r})")
            continue

        if not isinstance(job, dict):
            log.error(f"skipping non-object job on {queue_key}: {raw!r}")
            continue

        job_id = job.get("id")
        if not job_id:
            log.error(f"skipping job with no 'id' field: {raw!r}")
            continue
        job_key = f"job:{job_id}"

        if "input" not in job:
            job["status"], job["error"] = "failed", "job missing 'input' field"
            job["updated_at"] = _now()
            _safe_set(r, job_key, job)
            jobs_completed.labels(type=job_type, status="failed").inc()
            log.error(f"job {job_id} missing 'input', marked failed")
            continue

        # --- Normal path ----------------------------------------------
        job["status"] = "processing"
        job["updated_at"] = _now()
        _safe_set(r, job_key, job)
        log.info(f"processing job {job_id}")

        start = time.time()
        try:
            job["result"] = predict_fn(job["input"])
            job["status"], job["error"] = "completed", ""
        except Exception as e:
            # Catches anything predict_fn raises -- bad input shape,
            # model errors, whatever. The job is marked failed rather
            # than the worker process dying, so one bad job doesn't take
            # down the pod (and trigger an unnecessary restart/reschedule).
            log.error(f"inference failed for {job_id}: {e}")
            job["status"], job["error"] = "failed", str(e)
        finally:
            duration = time.time() - start
            inference_duration.labels(type=job_type).observe(duration)
            jobs_completed.labels(type=job_type, status=job["status"]).inc()
            try:
                push_to_gateway(pushgateway_url, job=f"{job_type}-worker", registry=registry)
            except Exception as e:
                # A Pushgateway outage shouldn't fail the job itself --
                # the prediction already happened. Just lose this metric push.
                log.warning(f"failed to push metrics: {e}")

        job["updated_at"] = _now()
        _safe_set(r, job_key, job)
        log.info(f"job {job_id} -> {job['status']}")
