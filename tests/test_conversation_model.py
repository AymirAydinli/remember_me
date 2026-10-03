from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from remember_me.database import Base
from remember_me.models import Conversation, Person


def test_person_has_multiple_conversations() -> None:
    test_engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=test_engine)

    try:
        with Session(test_engine) as db:
            person = Person(
                name="Anna",
                relationship="Daughter",
            )
            person.conversations.extend(
                [
                    Conversation(
                        summary="Anna discussed an upcoming trip.",
                        topics=["travel"],
                        follow_up="Ask how the trip went.",
                        occurred_at=datetime(2026, 10, 3, 15, 30),
                    ),
                    Conversation(
                        summary="Anna plans to visit on Sunday.",
                        topics=["family", "Sunday visit"],
                        follow_up=None,
                        occurred_at=datetime(2026, 10, 4, 12, 0),
                    ),
                ]
            )

            db.add(person)
            db.commit()

            saved_person = db.scalar(select(Person))

            assert saved_person is not None
            assert len(saved_person.conversations) == 2
            assert saved_person.conversations[0].topics == ["travel"]
            assert saved_person.conversations[1].follow_up is None

            db.delete(saved_person)
            db.commit()

            assert db.scalar(select(Conversation)) is None
    finally:
        test_engine.dispose()
