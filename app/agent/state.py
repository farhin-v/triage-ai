from typing import TypedDict, Optional, List

class AgentState(TypedDict, total=False):
    ticket_id: str
    subject: str
    body: str
    category: Optional[str]
    urgency: Optional[str]
    sentiment: Optional[str]
    retrieved_docs: Optional[List[str]]
    escalate: Optional[bool]
    escalation_reason: Optional[str]
    generated_response: Optional[str]
    error: Optional[str]