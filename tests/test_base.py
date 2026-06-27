"""
Tests run_worker() against a mocked Redis client and a mocked
push_to_gateway -- no real Redis, no real Pushgateway needed.

The loop in base.py is `while True`, so each test breaks out of it by
making the mocked brpop raise _StopTest on its last configured call, and
asserts on what happened before that point.

Run with:
    pip install -e . pytest
    pytest
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("REDIS_URL", "redis://fake:6379")

import base  # noqa: E402  (import after env var is set, base.py reads it inside run_worker)


class _StopTest(Exception):
    """Raised by the mocked brpop to break run_worker()'s infinite loop on cue."""


def _make_mock_redis(brpop_side_effects):
    mock_redis = MagicMock()
    mock_redis.brpop.side_effect = brpop_side_effects
    mock_redis.set.return_value = True
    return mock_redis


@patch("base.push_to_gateway")
@patch("base.redis.from_url")
def test_happy_path_marks_job_completed(mock_from_url, mock_push):
    job = {"id": "job-1", "input": {"x": 1}}
    mock_redis = _make_mock_redis([
        ("queue:regression", json.dumps(job)),
        _StopTest(),
    ])
    mock_from_url.return_value = mock_redis

    predict_fn = MagicMock(return_value={"prediction": 42})

    with pytest.raises(_StopTest):
        base.run_worker(job_type="regression", predict_fn=predict_fn)

    predict_fn.assert_called_once_with({"x": 1})

    # Two .set() calls expected: status -> "processing", then -> "completed".
    assert mock_redis.set.call_count == 2
    final_call_job = json.loads(mock_redis.set.call_args_list[-1].args[1])
    assert final_call_job["status"] == "completed"
    assert final_call_job["result"] == {"prediction": 42}
    assert final_call_job["error"] == ""


@patch("base.push_to_gateway")
@patch("base.redis.from_url")
def test_predict_fn_exception_marks_job_failed(mock_from_url, mock_push):
    job = {"id": "job-2", "input": {"x": 1}}
    mock_redis = _make_mock_redis([
        ("queue:regression", json.dumps(job)),
        _StopTest(),
    ])
    mock_from_url.return_value = mock_redis

    def predict_fn(_):
        raise ValueError("model blew up")

    with pytest.raises(_StopTest):
        base.run_worker(job_type="regression", predict_fn=predict_fn)

    final_call_job = json.loads(mock_redis.set.call_args_list[-1].args[1])
    assert final_call_job["status"] == "failed"
    assert "model blew up" in final_call_job["error"]


@patch("base.push_to_gateway")
@patch("base.redis.from_url")
def test_malformed_json_is_skipped_not_fatal(mock_from_url, mock_push):
    mock_redis = _make_mock_redis([
        ("queue:regression", "{not valid json"),
        _StopTest(),
    ])
    mock_from_url.return_value = mock_redis

    predict_fn = MagicMock()

    with pytest.raises(_StopTest):
        base.run_worker(job_type="regression", predict_fn=predict_fn)

    predict_fn.assert_not_called()
    mock_redis.set.assert_not_called()  # no job id to key off of -- nothing to write


@patch("base.push_to_gateway")
@patch("base.redis.from_url")
def test_job_missing_input_marked_failed_without_calling_predict_fn(mock_from_url, mock_push):
    job = {"id": "job-3"}  # no "input"
    mock_redis = _make_mock_redis([
        ("queue:regression", json.dumps(job)),
        _StopTest(),
    ])
    mock_from_url.return_value = mock_redis

    predict_fn = MagicMock()

    with pytest.raises(_StopTest):
        base.run_worker(job_type="regression", predict_fn=predict_fn)

    predict_fn.assert_not_called()
    final_call_job = json.loads(mock_redis.set.call_args_list[-1].args[1])
    assert final_call_job["status"] == "failed"
    assert "input" in final_call_job["error"]


@patch("base.push_to_gateway")
@patch("base.redis.from_url")
def test_pushgateway_failure_does_not_crash_or_skip_status_write(mock_from_url, mock_push):
    mock_push.side_effect = ConnectionError("pushgateway unreachable")
    job = {"id": "job-4", "input": {"x": 1}}
    mock_redis = _make_mock_redis([
        ("queue:regression", json.dumps(job)),
        _StopTest(),
    ])
    mock_from_url.return_value = mock_redis

    predict_fn = MagicMock(return_value={"prediction": 1})

    with pytest.raises(_StopTest):
        base.run_worker(job_type="regression", predict_fn=predict_fn)

    final_call_job = json.loads(mock_redis.set.call_args_list[-1].args[1])
    assert final_call_job["status"] == "completed"  # job still succeeds despite metrics push failing
