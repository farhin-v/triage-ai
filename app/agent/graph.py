from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.state import AgentState
from app.agent.tools import search_knowledge_base, classify_ticket, decide_escalation
from app.database.connections import SessionLocal
from app.database.models import Ticket
import os

def classification_node(state: AgentState) -> AgentState:
    """Classify the ticket category, urgency and sentiment."""
    result = classify_ticket.invoke({
        "subject": state["subject"],
        "body": state["body"]
    })
    state["category"] = result.get("category")
    state["urgency"] = result.get("urgency")
    state["sentiment"] = result.get("sentiment")
    return state

def retrieval_node(state: AgentState) -> AgentState:
    """Search the knowledge base for relevant documents."""
    category = state.get("category", "")
    query = f"{category} {state['subject']} {state['body']}"
    docs = search_knowledge_base.invoke({"query": query})
    state["retrieved_docs"] = docs
    return state

def escalation_node(state: AgentState) -> AgentState:
    """Decide whether the ticket should be escalated to a human agent."""
    result = decide_escalation.invoke({
        "category": state["category"],
        "urgency": state["urgency"],
        "sentiment": state["sentiment"]
    })
    state["escalate"] = result.get("escalate")
    state["escalation_reason"] = result.get("escalation_reason")
    return state

def response_generation_node(state: AgentState) -> AgentState:
    """Generate a response to the customer using Gemini."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    docs_text = "\n\n".join(state["retrieved_docs"]) if state["retrieved_docs"] else "No relevant documents found."

    prompt = f"""You are a professional customer support agent for a SaaS subscription company.

Your job is to generate accurate, grounded, and helpful responses to customer tickets using ONLY:
- The customer ticket (subject + body)
- The retrieved policy documents

You must NOT use any external knowledge or assumptions.

--------------------------------------------------

TRUTH AND GROUNDING RULE:

- If information is not explicitly present in the ticket or policy documents, treat it as UNKNOWN.
- Do NOT assume system behavior, internal actions, or hidden processes.
- Do NOT invent solutions outside the provided policies.

--------------------------------------------------

HOW TO USE POLICY DOCUMENTS:

- Extract only relevant sections from the provided policy text.
- Apply policies directly to the customer’s situation.
- Do NOT quote full policies unless necessary.
- Do NOT mention section numbers unless they are explicitly visible in the text.

--------------------------------------------------

DECISION FRAMEWORK BEFORE RESPONDING:

1. Understand the issue
   - What is the customer problem?
   - What is the customer trying to achieve?

2. Check policy support
   - Is there an explicit policy that resolves this?
   - If yes → RESOLVE
   - If partial → REQUEST_MORE_INFORMATION
   - If unclear → ESCALATE

3. Decide response type:
   - RESOLVE_FROM_POLICY
   - REQUEST_MORE_INFORMATION
   - ESCALATE

--------------------------------------------------

STRICT PROHIBITIONS:

- Do NOT claim actions were performed (no refunds, cancellations, updates)
- Do NOT say “I have processed” or “I have completed”
- Do NOT mention internal systems or backend access
- Do NOT invent troubleshooting steps
- Do NOT add generic advice unless present in policy
- Do NOT promise emails, refunds, or timelines unless explicitly stated in policy
- Do NOT hallucinate missing policy details
- Answer only from retrieved documents.
- Do NOT invent policies or timelines.

--------------------------------------------------

CUSTOMER ACKNOWLEDGEMENT RULE:

- Acknowledge ONLY what the customer explicitly said
- Do NOT reinterpret or exaggerate their issue

Example:
Correct: "I understand you're unable to access your account."
Incorrect: "Your account has failed due to system issues."

--------------------------------------------------

ESCALATION RULE:

If escalated:
- Clearly acknowledge the issue
- Apologize for inconvenience
- State it has been escalated to the appropriate team
- Do NOT add fake timelines or internal workflows

--------------------------------------------------

MISSING INFORMATION RULE:

If information is missing:
- Ask only the minimum required question(s)
- Do NOT overload the customer with multiple questions

--------------------------------------------------
You DO NOT know whether:

- a refund has been approved
- a refund has been issued
- the issue has been escalated
- a team is reviewing the case
- an investigation has started
- a ticket has been assigned
- the problem has been reproduced

Unless explicitly provided in the input.

Never claim:

- "I escalated your ticket"
- "Your refund is being processed"
- "Our team is investigating"
- "The issue has been forwarded"
- "The issue has been fixed"
- "The refund will be issued"

These statements are prohibited unless explicitly provided.
--------------------------------------------------
ALLOWED GOALS

Your response may only:

1. Acknowledge the customer's concern.
2. Summarize what the customer reported.
3. Request missing information when necessary.
4. Explain why the information is needed.
5. Maintain a professional and empathetic tone.

--------------------------------------------------
PROHIBITED BEHAVIOR

Do NOT:

- promise outcomes
- promise timelines
- promise refunds
- claim actions occurred
- mention internal workflows
- mention internal teams
- invent troubleshooting results
- invent investigation status

---------------------------------------------------------

RESPONSE STRUCTURE (STRICT):

1. Greeting (optional, short)
2. Acknowledgement of issue
3. Policy-based explanation OR clarification question OR escalation note
4. Clear next step for customer
5. Sign off

--------------------------------------------------

STYLE GUIDELINES:

- Professional, calm, and human-like
- No repetition
- No internal jargon
- No long paragraphs
- Prefer short clear sentences

--------------------------------------------------

INPUT DATA:

Subject:
{state["subject"]}

Body:
{state["body"]}

Category:
{state["category"]}

Urgency:
{state["urgency"]}

Sentiment:
{state["sentiment"]}

Escalated:
{state["escalate"]}

Relevant Policy Documents:
{docs_text}

--------------------------------------------------

FINAL INSTRUCTION:

Generate a response that is strictly grounded in the above policies and fully aligned with the decision framework.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        content = response.content
    elif isinstance(response.content, list):
        content = response.content[0]["text"]
    else:
        content = str(response.content)

    state["generated_response"] = content.strip()
    return state

def save_node(state: AgentState) -> AgentState:
    """Save the final results back to PostgreSQL."""
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).filter(Ticket.id == state["ticket_id"]).first()
        if ticket:
            ticket.category = state["category"]
            ticket.urgency = state["urgency"]
            ticket.sentiment = state["sentiment"]
            ticket.escalated = state["escalate"]
            ticket.escalation_reason = state["escalation_reason"]
            ticket.generated_response = state["generated_response"]
            ticket.status = "escalated" if state["escalate"] else "resolved"
            db.commit()
    except Exception as e:
        state["error"] = str(e)
        db.rollback()
    finally:
        db.close()
    return state

def create_graph():
    """Create and return the LangGraph agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", classification_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("generate", response_generation_node)
    graph.add_node("save", save_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "escalation")
    graph.add_edge("escalation", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)
    return graph.compile()

agent = create_graph()