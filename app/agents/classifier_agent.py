from typing import Literal, Dict, Any
import json
import re

import openai

from config.settings import settings
from app.logs.logger import logger

Category = Literal["positive_feedback", "negative_feedback", "query"]


class ClassifierAgent:
    """
    Agent responsible for classifying incoming messages into:
    - positive_feedback
    - negative_feedback
    - query
    and optionally extracting a ticket_number.

    Uses OpenAI's chat completion with JSON-style output for robust classification.
    """

    def __init__(self) -> None:
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model

    def classify(self, message: str) -> Dict[str, Any]:
        """
        Return a dict like:
        {
          "category": "positive_feedback" | "negative_feedback" | "query",
          "sentiment": "positive" | "neutral" | "negative",
          "ticket_number": "650932" | None
        }
        """
        logger.info("ClassifierAgent.classify called")

        result: Dict[str, Any] = {
            "category": "query",
            "sentiment": "neutral",
            "ticket_number": None,
        }

        system_prompt = (
            "You are a classifier for banking customer support messages.\n"
            "You MUST respond with a strict JSON object with keys:\n"
            '  - \"category\": one of [\"positive_feedback\", \"negative_feedback\", \"query\"]\n'
            '  - \"sentiment\": one of [\"positive\", \"neutral\", \"negative\"]\n'
            '  - \"ticket_number\": a 6-digit string if present in the message, '
            "otherwise null.\n\n"
            "Rules:\n"
            "- If the user is mainly THANKING, praising or appreciating service, "
            'category = \"positive_feedback\".\n'
            "- If the user is mainly COMPLAINING, unhappy, or describing a problem, "
            'category = \"negative_feedback\".\n'
            "- If the user is ASKING about the status of a ticket, requesting help, "
            'or asking a question, category = \"query\".\n'
            "- If there is a ticket number like 'ticket 650932' or '#650932', "
            'extract it as a 6-digit string in \"ticket_number\".\n'
            "- Do NOT include any other fields.\n"
        )

        user_prompt = (
            "Classify the following customer message.\n\n"
            f"Message:\n{message}\n\n"
            "Return ONLY the JSON object."
        )

        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

            content = completion.choices[0].message["content"]
            logger.debug(f"Raw classifier LLM content: {content}")

            parsed = json.loads(content)

            category = parsed.get("category", "query")
            sentiment = parsed.get("sentiment", "neutral")
            ticket_number = parsed.get("ticket_number", None)

            if ticket_number is not None:
                m = re.fullmatch(r"\d{6}", str(ticket_number))
                ticket_number = m.group(0) if m else None

            result["category"] = category
            result["sentiment"] = sentiment
            result["ticket_number"] = ticket_number

        except Exception:
            logger.exception("Error calling OpenAI in ClassifierAgent; using fallback.")

            lower_msg = message.lower()
            if any(k in lower_msg for k in ["thank", "great", "good job", "appreciate"]):
                result["category"] = "positive_feedback"
                result["sentiment"] = "positive"
            elif any(
                k in lower_msg
                for k in ["not happy", "bad", "terrible", "issue", "problem", "complain"]
            ):
                result["category"] = "negative_feedback"
                result["sentiment"] = "negative"
            else:
                result["category"] = "query"
                result["sentiment"] = "neutral"

            m = re.search(r"(?:ticket\s*#?\s*|#)(\d{6})", message)
            if m:
                result["ticket_number"] = m.group(1)

        logger.debug(f"ClassifierAgent result: {result}")
        return result
