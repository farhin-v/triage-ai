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
    
    query = f"{state['subject']} {state['body']}"   
    
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
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model_kwargs={"thinking_level": "low"}
    )    
    docs_text = "\n\n".join(state["retrieved_docs"]) if state["retrieved_docs"] else "No relevant documents found."
    
    prompt = f"""You are a helpful customer support agent. Write a professional and empathetic response to the customer ticket below.

Use the relevant policy documents provided to make your response accurate.

Ticket Subject: {state["subject"]}
Ticket Body: {state["body"]}
Category: {state["category"]}
Urgency: {state["urgency"]}
Customer Sentiment: {state["sentiment"]}

Relevant Policy Documents:
{docs_text}

Write a clear, helpful and empathetic response to this customer:"""

    response = llm.invoke(prompt)
    
    state["generated_response"] = response.content.strip()
    
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
    graph.add_node("escalate", escalation_node)
    graph.add_node("generate", response_generation_node)
    graph.add_node("save", save_node)
    
    graph.set_entry_point("classify")
    
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "escalate")
    graph.add_edge("escalate", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)
    
    return graph.compile()


agent = create_graph()