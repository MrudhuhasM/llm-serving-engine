from datetime import datetime

from engine.request import Request


def test_request_required_fields():
    request = Request(
        request_id="req-1",
        prompt="Hello",
        input_ids=[1, 2, 3],
        max_new_tokens=10,
    )

    assert request.request_id == "req-1"
    assert request.prompt == "Hello"
    assert request.input_ids == [1, 2, 3]
    assert request.max_new_tokens == 10


def test_request_default_values():
    before = datetime.now()
    request = Request(
        request_id="req-2",
        prompt="Hi",
        input_ids=[4, 5],
        max_new_tokens=5,
    )
    after = datetime.now()

    assert request.generated_token_ids == []
    assert request.finished is False
    assert request.finished_reason is None
    assert before <= request.created_at <= after


def test_request_default_generated_token_ids_are_independent():
    request_a = Request(
        request_id="req-a",
        prompt="A",
        input_ids=[1],
        max_new_tokens=1,
    )
    request_b = Request(
        request_id="req-b",
        prompt="B",
        input_ids=[2],
        max_new_tokens=1,
    )

    request_a.generated_token_ids.append(42)

    assert request_a.generated_token_ids == [42]
    assert request_b.generated_token_ids == []


def test_request_optional_fields_can_be_set():
    created_at = datetime(2026, 1, 1, 12, 0, 0)
    request = Request(
        request_id="req-3",
        prompt="Done",
        input_ids=[9],
        max_new_tokens=20,
        generated_token_ids=[100, 101],
        finished=True,
        finished_reason="length",
        created_at=created_at,
    )

    assert request.generated_token_ids == [100, 101]
    assert request.finished is True
    assert request.finished_reason == "length"
    assert request.created_at == created_at


def test_request_fields_are_mutable():
    request = Request(
        request_id="req-4",
        prompt="Mutable",
        input_ids=[7],
        max_new_tokens=3,
    )

    request.generated_token_ids.append(8)
    request.finished = True
    request.finished_reason = "stop"

    assert request.generated_token_ids == [8]
    assert request.finished is True
    assert request.finished_reason == "stop"
