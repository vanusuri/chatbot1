from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_number = Column(String(32), unique=True, nullable=False)
    customer_name = Column(String(255), nullable=True)
    message = Column(Text, nullable=False)
    status = Column(String(64), nullable=False, default="Open")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    channel = Column(String(64), default="Streamlit_UI")

    def __repr__(self) -> str:
        return f"<SupportTicket(ticket_number={self.ticket_number}, status={self.status})>"


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(128), nullable=False)
    user_message = Column(Text, nullable=False)
    classifier = Column(String(64), nullable=True)
    routed_agent = Column(String(64), nullable=True)
    response = Column(Text, nullable=True)
    ticket_number = Column(String(32), nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AgentLog(session_id={self.session_id}, classifier={self.classifier})>"


class SupportDocChunk(Base):
    __tablename__ = "support_doc_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(255), nullable=False)  # e.g., filename
    chunk_index = Column(Integer, nullable=False)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON-encoded list[float]
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<SupportDocChunk(doc_id={self.doc_id}, chunk_index={self.chunk_index})>"
