from typing import Optional, Dict, Any

from app.agents import ClassifierAgent, FeedbackAgent, QueryAgent, KnowledgeAgent
from app.db.dao import log_event
from app.logs.logger import logger


class Orchestrator:
    """
    Entry point for the application. Orchestrates:
    - Classification
    - Routing to feedback, ticket-status, or knowledge (RAG) agents
    - Logging
    """

    def __init__(self) -> None:
        self.classifier = ClassifierAgent()
        self.feedback_agent = FeedbackAgent()
        self.query_agent = QueryAgent()
        self.knowledge_agent = KnowledgeAgent()

    def handle_message(
        self,
        message: str,
        session_id: str,
        customer_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("Orchestrator.handle_message called")
        classifier_result: Dict[str, Any] = {}
        routed_agent = None
        response_text = ""
        ticket_number: Optional[str] = None
        success = True
        error_message: Optional[str] = None

        try:
            classifier_result = self.classifier.classify(message)
            category = classifier_result.get("category", "query")
            sentiment = classifier_result.get("sentiment", "neutral")
            ticket_number = classifier_result.get("ticket_number")

            if category == "positive_feedback":
                routed_agent = "feedback_handler_positive"
                response_text = self.feedback_agent.handle_positive(
                    message, customer_name
                )

            elif category == "negative_feedback":
                routed_agent = "feedback_handler_negative"
                response_text, ticket_number = self.feedback_agent.handle_negative(
                    message, customer_name
                )

            else:  # "query"
                if ticket_number:
                    routed_agent = "query_handler"
                    response_text = self.query_agent.handle_query(
                        message, ticket_number=ticket_number
                    )
                else:
                    routed_agent = "knowledge_handler"
                    response_text = self.knowledge_agent.handle_knowledge_query(message)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in Orchestrator.handle_message")
            success = False
            error_message = str(exc)
            response_text = (
                "We’re experiencing issues right now. Please try again later "
                "or contact support."
            )

        log_event(
            session_id=session_id,
            user_message=message,
            classifier=classifier_result.get("category") if classifier_result else None,
            routed_agent=routed_agent,
            response=response_text,
            ticket_number=ticket_number,
            success=success,
            error_message=error_message,
        )

        return {
            "response": response_text,
            "category": classifier_result.get("category") if classifier_result else None,
            "sentiment": classifier_result.get("sentiment") if classifier_result else None,
            "ticket_number": ticket_number,
            "routed_agent": routed_agent,
            "success": success,
            "error_message": error_message,
        }
