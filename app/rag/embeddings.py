from typing import List

import openai

from config.settings import settings
from app.logs.logger import logger

#This function calls OpenAI’s embedding model to convert text into a numerical vector that captures its semantic meaning. These embeddings are used in our RAG pipeline to perform similarity search over support documents, allowing the system to retrieve relevant information based on meaning rather than keyword matching.
def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI for a given text."""
    openai.api_key = settings.openai_api_key
    logger.debug("Requesting embedding from OpenAI")
    response = openai.Embedding.create(
        model=settings.embedding_model,
        input=[text],
    )
    return response["data"][0]["embedding"]
