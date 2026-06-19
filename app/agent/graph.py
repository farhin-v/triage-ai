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
    """Decide whether the ticket should be escalated to a human agent.

    NOTE: backend routing signal only. It sets the ticket status in the
    database / dashboard. It is NOT passed into the customer response prompt -
    the customer should never be told about internal escalation or routing.
    """
    result = decide_escalation.invoke({
        "category": state["category"],
        "urgency": state["urgency"],
        "sentiment": state["sentiment"]
    })
    state["escalate"] = result.get("escalate")
    state["escalation_reason"] = result.get("escalation_reason")
    return state

def response_generation_node(state: AgentState) -> AgentState:
    """Generate a grounded, customer-facing response using Gemini.

    The reply must actually help: answer from policy or ask for the specific
    information needed. It never punts to 'a team will contact you' by default,
    and never references escalation, internal teams, or internal procedures.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    docs_text = "\n\n".join(state["retrieved_docs"]) if state.get("retrieved_docs") else "No relevant policy documents found."

    prompt = f"""You are a customer support agent for a SaaS subscription company. Write the reply the CUSTOMER will read.

Your goal is to actually help the customer. Every reply must give the customer a concrete answer or a concrete next step they can act on - never a vague "someone will get back to you".

USE THE POLICY, BUT DON'T LEAK INTERNAL PROCESS:
The POLICY DOCUMENTS below may mix customer-facing facts with internal routing steps.
- DO use the customer-facing facts: timeframes, eligibility, what the customer should do, what generally happens, and what information they must provide.
- Do NOT repeat internal routing to the customer: never use the words "escalate" or "escalated", never name an internal team (billing team, technical team, shipping team, etc.), and never describe internal handling steps.

GROUNDING RULE:
Every factual statement about policies, refunds, fees, timeframes, or eligibility MUST come directly from the POLICY DOCUMENTS below. Never invent a policy, amount, timeframe, or threshold. Never claim an action has already happened (no "I've issued your refund", "I've processed this").

HOW TO REPLY (one short greeting, then the substance, then one short sign-off):
1. One short greeting.
2. One sentence acknowledging the specific issue the customer raised.
3. The substance - ALWAYS help the customer, using whichever applies:
   - If the policy answers the question, give the customer the relevant answer in plain language.
   - If you need information to move forward (order number, transaction ID, registered email, etc.), state the relevant policy AND ask for the specific minimum information you need.
   - For a status question about something specific (an order, a charge), state the relevant standard policy (such as the expected timeframe) and ask for the identifier you need to look it up (e.g. the order number).
   Do NOT default to telling the customer that a team will contact them. Give them an actual answer or a clear action they can take. Only mention a person following up if the policy genuinely requires human review, and even then give the relevant policy and ask for what is needed first.
4. One short sign-off.

Be concise and professional. Plain customer language only.

TICKET
Subject: {state["subject"]}
Body: {state["body"]}

POLICY DOCUMENTS
{docs_text}

Write only the customer reply, nothing else.
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