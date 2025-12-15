from typing import Optional

import openai

from config.settings import settings
from app.db.dao import get_ticket_by_number
from app.logs.logger import logger


class QueryAgent:
    """
    Handles queries, especially those asking for ticket status.

    - Extracts ticket number from message (if not already provided).
    - Looks up status in the support_tickets table.
    - Uses OpenAI to generate a natural, user-friendly response
      that references the ticket number and current status.
    """

    def __init__(self) -> None:
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model

    def handle_query(
        self, message: str, ticket_number: Optional[str] = None
    ) -> str:
        logger.info("QueryAgent.handle_query called")

        if not ticket_number:
            ticket_number = self._extract_ticket_number(message)

        if not ticket_number:
            try:
                system_prompt = (
                    "You are a banking customer support assistant.\n"
                    "The user is asking a question but has not provided a ticket number.\n"
                    "Politely ask them to provide their 6-digit ticket number.\n"
                    "Respond in one short, friendly sentence."
                )
                user_prompt = (
                    "The customer is asking about their ticket, but no ticket number "
                    "could be detected from the message:\n\n"
                    f"\"{message}\"\n\n"
                    "Write a brief reply."
                )

                completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                )
                content = completion.choices[0].message["content"].strip()
                if content:
                    return content
            except Exception:
                logger.exception(
                    "Error calling OpenAI in QueryAgent.handle_query (no ticket)."
                )

            return (
                "Could you please provide your 6-digit ticket number so I can "
                "check its status?"
            )

        ticket = get_ticket_by_number(ticket_number)

        if ticket is None:
            try:
                system_prompt = (
                    "You are a helpful banking customer support assistant.\n"
                    "You could not find the ticket in the system.\n"
                    "Explain this to the customer politely, ask them to "
                    "double-check the number, and offer alternative help.\n"
                    "Respond in 1–2 sentences, friendly and professional.\n"
                    "Include the ticket number as '#<ticket_number>'."
                )
                user_prompt = (
                    f"The user asked about ticket number {ticket_number}, "
                    "but it does not exist in our records.\n\n"
                    "Write a brief response for the customer."
                )

                completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                )
                content = completion.choices[0].message["content"].strip()
                if content:
                    return content
            except Exception:
                logger.exception(
                    "Error calling OpenAI in QueryAgent.handle_query (ticket not found)."
                )

            return (
                f"I’m unable to find ticket #{ticket_number} in our records. "
                "Please double-check the number or contact support."
            )

        status_text = ticket.status or "Open"
        created = ticket.created_at
        customer_name = ticket.customer_name
        message_snippet = (
            ticket.message[:200] + "..."
            if ticket.message and len(ticket.message) > 200
            else ticket.message
        )

        status_category = "open"
        if status_text.lower() in {"resolved", "closed"}:
            status_category = "resolved"
        elif status_text.lower() in {"in progress", "pending"}:
            status_category = "in_progress"

        try:
            system_prompt = (
                "You are a professional but friendly banking customer support agent.\n"
                "You are given:\n"
                "- a ticket number\n"
                "- current ticket status\n"
                "- optionally the customer's name\n"
                "- optionally a short summary of the issue\n\n"
                "Your job is to describe the ticket status clearly and reassure the customer.\n"
                "Guidance:\n"
                "- If status is 'Open' or an equivalent, let them know it's been logged "
                "and will be reviewed.\n"
                "- If status is 'In Progress' / 'Pending', explain that the team is actively "
                "working on it.\n"
                "- If status is 'Resolved' / 'Closed', inform them it is marked resolved "
                "and invite them to reach out again if needed.\n"
                "Respond in 1–3 short sentences.\n"
                "Always mention the ticket as '#<ticket_number>'."
            )

            user_prompt_parts = [
                f"Ticket number: {ticket_number}",
                f"Ticket status: {status_text}",
                f"Status category (for your reasoning): {status_category}",
            ]
            if customer_name:
                user_prompt_parts.append(f"Customer name: {customer_name}")
            if message_snippet:
                user_prompt_parts.append(
                    f"Original issue summary/snippet: \"{message_snippet}\""
                )

            user_prompt = "\n".join(user_prompt_parts)

            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
            )
            content = completion.choices[0].message["content"].strip()
            logger.debug(f"Ticket status LLM response: {content}")
            if content:
                if str(ticket_number) not in content:
                    content += f" (This refers to ticket #{ticket_number}.)"
                return content

        except Exception:
            logger.exception(
                "Error calling OpenAI in QueryAgent.handle_query (ticket found)."
            )

        return f"Your ticket #{ticket.ticket_number} is currently marked as: {ticket.status}."

    # --------- Helpers --------- #

    def _extract_ticket_number(self, message: str) -> Optional[str]:
        import re

        m = re.search(r"(?:ticket\s*#?\s*|#)(\d{6})", message)
        if m:
            return m.group(1)
        return None
