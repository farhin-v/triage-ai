from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connections import get_db
from app.database.models import Ticket
from app.agent.graph import agent
from pydantic import BaseModel
import uuid

class TicketRequest(BaseModel):
    subject: str
    body: str

router = APIRouter(prefix="/tickets", tags=["Tickets"])

@router.post("/")
def submit_ticket(ticket: TicketRequest, db: Session = Depends(get_db)):
    ticket_id = str(uuid.uuid4())
    
    new_ticket = Ticket(
        id=ticket_id,
        subject=ticket.subject,
        body=ticket.body,
        status="open"
    )
    
    db.add(new_ticket)
    db.commit()
    
    initial_state = {
        "ticket_id": ticket_id,
        "subject": ticket.subject,
        "body": ticket.body,
        "category": None,
        "urgency": None,
        "sentiment": None,
        "retrieved_docs": None,
        "escalate": None,
        "escalation_reason": None,
        "generated_response": None,
        "error": None
    }
    
    try:
        result = agent.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "ticket_id": ticket_id,
        "category": result["category"],
        "urgency": result["urgency"],
        "sentiment": result["sentiment"],
        "escalated": result["escalate"],
        "escalation_reason": result["escalation_reason"],
        "generated_response": result["generated_response"],
        "status": "escalated" if result["escalate"] else "resolved"
    }