from typing import List

import openai

from config.settings import settings
from app.logs.logger import logger
from app.rag.retriever import retrieve_relevant_chunks


class KnowledgeAgent:
    """
    Knowledge/RAG agent.

    - Uses the support document index to retrieve relevant chunks.
    - Calls the LLM with those chunks as context.
    - Answers user questions based ONLY on the support documents; if
      the answer is not present, clearly indicates that.
    """

    def __init__(self) -> None:
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model

    def handle_knowledge_query(self, message: str) -> str:
        logger.info("KnowledgeAgent.handle_knowledge_query called")

        chunks = retrieve_relevant_chunks(message, top_k=4)
        if not chunks:
            return (
                "I’m not able to find information about that in our current support "
                "documents. Please contact customer support for further assistance."
            )

        context_blocks: List[str] = []
        for score, title, content in chunks:
            context_blocks.append(f"[{title}] (score={score:.3f})\n{content}")

        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a banking customer support assistant powered by a knowledge base.\n"
            "You will be given several excerpts from official support documents, "
            "followed by a customer question.\n"
            "Your job is to answer the question using ONLY this provided context.\n"
            "If the answer is not in the context, say that you do not know and "
            "suggest contacting customer support.\n"
            "Be concise, accurate, and professional.\n"
        )

        user_prompt = (
            "Here are the most relevant support document excerpts:\n\n"
            f"{context_str}\n\n"
            "Customer question:\n"
            f"{message}\n\n"
            "Based ONLY on the support document excerpts above, answer the question. "
            "If the answer is not present, say you do not know."
        )

        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = completion.choices[0].message["content"].strip()
            logger.debug(f"KnowledgeAgent LLM response: {content}")
            return content or (
                "I’m not able to find information about that in our current support "
                "documents. Please contact customer support for further assistance."
            )
        except Exception:
            logger.exception("Error calling OpenAI in KnowledgeAgent.handle_knowledge_query")
            return (
                "I’m not able to retrieve information from our support documents at the moment. "
                "Please try again later or contact customer support."
            )
