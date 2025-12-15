import pytest

from app.agents.classifier_agent import ClassifierAgent


@pytest.fixture
def agent() -> ClassifierAgent:
    return ClassifierAgent()


def test_positive_feedback(agent: ClassifierAgent) -> None:
    msg = "Thank you for helping me with my account!"
    result = agent.classify(msg)
    assert result["category"] == "positive_feedback"


def test_negative_feedback(agent: ClassifierAgent) -> None:
    msg = "I am not happy with your service, this is a terrible issue."
    result = agent.classify(msg)
    assert result["category"] == "negative_feedback"


def test_query(agent: ClassifierAgent) -> None:
    msg = "Can you check ticket 123456 status?"
    result = agent.classify(msg)
    assert result["category"] == "query"
    assert result["ticket_number"] == "123456"
