"""Tests for JSON logging configuration."""

import json
import logging
import sys
from datetime import UTC, datetime

import pytest

from helix.logging_config import JsonFormatter, configure_logging


def make_record(msg="hi %s", args=("there",), exc_info=None, level=logging.INFO):
    return logging.LogRecord("helix", level, "x.py", 1, msg, args, exc_info)


def test_formatter_renders_one_line_json():
    payload = json.loads(JsonFormatter().format(make_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "helix"
    assert payload["message"] == "hi there"
    assert "exception" not in payload


def test_formatter_uses_record_creation_time():
    record = make_record()
    record.created = 0.0
    payload = json.loads(JsonFormatter().format(record))
    assert payload["timestamp"] == datetime.fromtimestamp(0, UTC).isoformat()


def test_formatter_includes_exception():
    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError:
        record = make_record("boom", None, sys.exc_info(), logging.ERROR)
    payload = json.loads(JsonFormatter().format(record))
    assert "ZeroDivisionError: division by zero" in payload["exception"]


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)


def test_configure_logging_sets_single_json_handler(restore_root_logger):
    configure_logging("DEBUG")
    configure_logging("WARNING")
    root = restore_root_logger
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.WARNING
