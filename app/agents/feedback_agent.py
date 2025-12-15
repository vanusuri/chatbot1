from typing import Optional, Tuple
import random

import openai

from config.settings import settings
from app.db.dao import create_ticket
from app.logs.logger import logger


class FeedbackAgent:
    """
    Handles positive and negative feedback flows:
    - Positive: generate thank-you message via LLM.
    - Negative: create ticket and generate empathetic apology message via LLM.
    """

    def __init__(self) -> None:
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model

    # --------- Positive Flow --------- #

    def handle_positive(self, message: str, customer_name: Optional[str] = None) -> str:
        logger.info("FeedbackAgent.handle_positive called")

        system_prompt = (
            "You are a polite and concise banking customer support agent.\n"
            "Your goal is to acknowledge and thank the customer for their positive feedback.\n"
            "Respond in ONE or TWO sentences, friendly and professional.\n"
            "If a customer name is provided, address them by name.\n"
        )

        name_part = f"Customer name: {customer_name}" if customer_name else "Customer name: (not provided)"
        user_prompt = (
            f"{name_part}\n\n"
            "The customer left this positive feedback:\n"
            f"\"{message}\"\n\n"
            "Write a brief response."
        )

        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            content = completion.choices[0].message["content"].strip()
            logger.debug(f"Positive feedback LLM response: {content}")
            if content:
                return content

        except Exception:
            logger.exception("Error calling OpenAI in FeedbackAgent.handle_positive")

        if customer_name:
            return (
                f"Thank you for your kind words, {customer_name}! "
                "We’re delighted to assist you."
            )
        return "Thank you for your kind words! We’re delighted to assist you."

    # --------- Negative Flow --------- #

    def handle_negative(
        self, message: str, customer_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Returns (response_text, ticket_number).

        This method:
        - Generates a ticket_number.
        - Stores a new SupportTicket row.
        - Uses LLM to craft an empathetic apology message with that ticket number.
        """
        logger.info("FeedbackAgent.handle_negative called")

        ticket_number = self._generate_ticket_number()
        create_ticket(
            ticket_number=ticket_number,
            customer_name=customer_name,
            message=message,
            status="Open",
        )

        system_prompt = (
            "You are a banking customer support agent.\n"
            "The customer is unhappy or reporting an issue.\n"
            "You must respond with empathy, apologize for the inconvenience, "
            "and reassure them that their issue is being investigated.\n"
            "Include the ticket number in the reply as '#<ticket_number>'.\n"
            "Respond in about 2–3 sentences, professional but warm.\n"
        )

        name_part = f"Customer name: {customer_name}" if customer_name else "Customer name: (not provided)"
        user_prompt = (
            f"{name_part}\n"
            f"New ticket number: {ticket_number}\n\n"
            "Customer message:\n"
            f"\"{message}\"\n\n"
            "Write a brief, empathetic response that mentions the ticket number."
        )

        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
            )
            content = completion.choices[0].message["content"].strip()
            logger.debug(f"Negative feedback LLM response: {content}")
            if content:
                if str(ticket_number) not in content:
                    content += f" Your ticket #{ticket_number} has been created."
                return content, ticket_number

        except Exception:
            logger.exception("Error calling OpenAI in FeedbackAgent.handle_negative")

        if customer_name:
            response = (
                f"We apologize for the inconvenience, {customer_name}. "
                f"A new ticket #{ticket_number} has been generated, and our team will "
                f"follow up shortly."
            )
        else:
            response = (
                f"We apologize for the inconvenience. A new ticket #{ticket_number} "
                f"has been generated, and our team will follow up shortly."
            )

        return response, ticket_number

    # --------- Helpers --------- #

    def _generate_ticket_number(self) -> str:
        """Generate a pseudo-unique 6-digit ticket number.

        NOTE: For production, ensure collision checks in DB or use PK-based scheme.
        """
        ticket_number = f"{random.randint(0, 999999):06d}"
        logger.debug(f"Generated ticket_number: {ticket_number}")
        return ticket_number
