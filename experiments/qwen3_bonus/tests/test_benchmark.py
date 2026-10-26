import pytest
from experiments.qwen3_bonus.benchmark import content_from_sse


def test_sse_counts_only_real_text_and_surfaces_stream_errors():
    assert content_from_sse(': ping')==''
    assert content_from_sse('data: [DONE]')==''
    assert content_from_sse('data: {"choices":[{"delta":{"role":"assistant"}}]}')==''
    assert content_from_sse('data: {"choices":[{"delta":{"content":"你好"}}]}')=='你好'
    with pytest.raises(RuntimeError):content_from_sse('data: {"error":{"code":"local_unavailable"}}')
