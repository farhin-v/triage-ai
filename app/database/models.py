from sqlalchemy import Column, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(String, primary_key=True)
    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    urgency = Column(String(50), nullable=True)
    sentiment = Column(String(50), nullable=True)
    escalated = Column(String(10), nullable=True)
    escalation_reason = Column(Text, nullable=True)
    generated_response = Column(Text, nullable=True)
    faithfulness_score = Column(String(10), nullable=True)
    status = Column(String(50), default="pending")
    created_at = Column(String, nullable=True)
    updated_at = Column(String, nullable=True)
