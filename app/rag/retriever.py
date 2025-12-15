from typing import List, Tuple
import json

import numpy as np

from app.db.dao import get_all_support_doc_chunks
from app.logs.logger import logger
from .embeddings import get_embedding


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def retrieve_relevant_chunks(
    query: str, top_k: int = 5
) -> List[Tuple[float, str, str]]:
    """
    Retrieve top_k most similar support doc chunks for the query.

    Returns a list of tuples:
    [(similarity, title, content), ...]
    """
    logger.info("Retrieving relevant chunks for query via RAG")
    query_emb = np.array(get_embedding(query), dtype=float)

    chunks = get_all_support_doc_chunks()
    scored = []

    for ch in chunks:
        emb_list = json.loads(ch.embedding)
        emb_vec = np.array(emb_list, dtype=float)
        score = _cosine_sim(query_emb, emb_vec)
        title = ch.title or ch.doc_id
        scored.append((score, title, ch.content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
