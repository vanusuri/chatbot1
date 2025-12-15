from typing import Optional, List

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings
from .models import Base, SupportTicket, AgentLog, SupportDocChunk

_engine = create_engine(settings.db_url, echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=_engine)


def get_db_session() -> Session:
    """Create a new session. (Use context manager in real code.)"""
    return SessionLocal()


# --------- SupportTicket operations --------- #

def create_ticket(
    ticket_number: str,
    message: str,
    customer_name: Optional[str] = None,
    channel: str = "Streamlit_UI",
    status: str = "Open",
) -> SupportTicket:
    session = get_db_session()
    try:
        ticket = SupportTicket(
            ticket_number=ticket_number,
            customer_name=customer_name,
            message=message,
            status=status,
            channel=channel,
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket
    finally:
        session.close()


def get_ticket_by_number(ticket_number: str) -> Optional[SupportTicket]:
    session = get_db_session()
    try:
        stmt = select(SupportTicket).where(SupportTicket.ticket_number == ticket_number)
        result = session.execute(stmt).scalar_one_or_none()
        return result
    finally:
        session.close()


def get_all_tickets(limit: int = 100) -> List[SupportTicket]:
    session = get_db_session()
    try:
        stmt = select(SupportTicket).limit(limit)
        result = session.execute(stmt).scalars().all()
        return result
    finally:
        session.close()


# --------- AgentLog operations --------- #

def log_event(
    session_id: str,
    user_message: str,
    classifier: Optional[str] = None,
    routed_agent: Optional[str] = None,
    response: Optional[str] = None,
    ticket_number: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> None:
    session = get_db_session()
    try:
        log = AgentLog(
            session_id=session_id,
            user_message=user_message,
            classifier=classifier,
            routed_agent=routed_agent,
            response=response,
            ticket_number=ticket_number,
            success=success,
            error_message=error_message,
        )
        session.add(log)
        session.commit()
    finally:
        session.close()


def get_recent_logs(limit: int = 50) -> List[AgentLog]:
    session = get_db_session()
    try:
        stmt = select(AgentLog).order_by(AgentLog.timestamp.desc()).limit(limit)
        return session.execute(stmt).scalars().all()
    finally:
        session.close()


# --------- SupportDocChunk operations (RAG) --------- #

def clear_support_docs() -> None:
    """Delete all existing support document chunks."""
    session = get_db_session()
    try:
        session.execute(delete(SupportDocChunk))
        session.commit()
    finally:
        session.close()


def add_support_doc_chunk(
    doc_id: str,
    chunk_index: int,
    title: Optional[str],
    content: str,
    embedding_json: str,
) -> SupportDocChunk:
    session = get_db_session()
    try:
        chunk = SupportDocChunk(
            doc_id=doc_id,
            chunk_index=chunk_index,
            title=title,
            content=content,
            embedding=embedding_json,
        )
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        return chunk
    finally:
        session.close()


def get_all_support_doc_chunks() -> List[SupportDocChunk]:
    session = get_db_session()
    try:
        stmt = select(SupportDocChunk)
        return session.execute(stmt).scalars().all()
    finally:
        session.close()
