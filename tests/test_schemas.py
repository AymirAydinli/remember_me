import pytest
from pydantic import ValidationError

from remember_me.schemas import ConversationSummaryOutput


def test_validates_conversation_summary() -> None:
    result = ConversationSummaryOutput(
        summary="  Anna discussed her upcoming trip.  ",
        topics=[" travel ", "family"],
        follow_up="  Ask how the trip went.  ",
    )

    assert result.summary == "Anna discussed her upcoming trip."
    assert result.topics == ["travel", "family"]
    assert result.follow_up == "Ask how the trip went."


def test_rejects_empty_summary() -> None:
    with pytest.raises(ValidationError):
        ConversationSummaryOutput(
            summary="   ",
            topics=[],
        )


def test_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        ConversationSummaryOutput.model_validate(
            {
                "summary": "A valid summary.",
                "topics": [],
                "invented_field": "Not allowed",
            }
        )
