from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SummaryText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]

TopicText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
    ),
]

FollowUpText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]


class ConversationSummaryOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    summary: SummaryText
    topics: list[TopicText] = Field(
        default_factory=list,
        max_length=8,
    )
    follow_up: FollowUpText | None = None
