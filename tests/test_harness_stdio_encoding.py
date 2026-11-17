"""JSONL wire encoding is independent of the Windows code page and injected text streams."""

import io
import json
import sys
from types import SimpleNamespace

from autotem.harness import cli


def request(value):
    return (
        json.dumps(
            {
                "tool": "validate_plan",
                "request_id": value,
                "arguments": {"plan": {"input_path": value}},
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def serve_pipe(monkeypatch, raw):
    incoming = io.TextIOWrapper(io.BytesIO(raw), encoding="gbk", errors="surrogateescape")
    buffer = io.BytesIO()
    outgoing = io.TextIOWrapper(buffer, encoding="utf-16")
    harness = SimpleNamespace(validate_plan=lambda plan: plan)
    with monkeypatch.context() as isolated:
        isolated.setattr(sys, "stdin", incoming)
        isolated.setattr(sys, "stdout", outgoing)
        assert cli.serve_stdio(harness) == 0
    assert incoming.encoding == "gbk" and outgoing.encoding == "utf-16"
    return [json.loads(line) for line in buffer.getvalue().decode("utf-8").splitlines()]


def test_default_pipe_preserves_literal_unicode_and_uses_utf8_for_responses(monkeypatch):
    value = "样本 项目 [A] & (B)/亮度 β.npy"
    (reply,) = serve_pipe(monkeypatch, request(value).encode("utf-8"))
    assert reply["ok"] and reply["request_id"] == reply["result"]["input_path"] == value


def test_invalid_utf8_and_oversized_multibyte_frame_preserve_next_request(monkeypatch):
    good = request("样本").encode("utf-8")
    monkeypatch.setattr(cli, "MAX_REQUEST_BYTES", len(good))
    oversized = request("样本" * 120).encode("utf-8")
    invalid, too_large, next_reply = serve_pipe(
        monkeypatch, b'{"tool":"\xff"}\n' + oversized + good
    )
    assert invalid["error"]["code"] == too_large["error"]["code"] == "REQUEST_INVALID"
    assert "size limit" in too_large["error"]["message"]
    assert next_reply["ok"] and next_reply["request_id"] == "样本"


def test_injected_text_streams_still_preserve_unicode():
    src = io.StringIO(request("测量"))
    sink = io.StringIO()
    assert cli.serve_stdio(SimpleNamespace(validate_plan=lambda plan: plan), src, sink) == 0
    reply = json.loads(sink.getvalue())
    assert reply["ok"] and reply["result"]["input_path"] == "测量"
