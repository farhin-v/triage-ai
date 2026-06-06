from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.connections import get_db
from app.database.models import Ticket
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
        status="pending"
    )

    db.add(new_ticket)
    db.commit()

    return {"ticket_id": ticket_id, "status": "pending", "message": "Ticket received successfully"}

