from langchain_core.tools import tool
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI 
import os

@tool
def search_knowledge_base(query: str) -> List[str]:
    """Search the Qdrant knowledge base for documents relevant to the query."""
    
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_vector = model.encode(query).tolist()
    
    results = client.query_points(
        collection_name="knowledge_base",
        query=query_vector,
        limit=5
    )
    
    docs = [result.payload["text"] for result in results.points]
    
    return docs

@tool
def classify_ticket(subject: str, body: str) -> dict:
    """Classify the support ticket and detect urgency and sentiment using Gemini."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    prompt = f"""You are a support ticket classifier. Analyze the ticket below and return ONLY the following format with no extra text:

CATEGORY: <one of: billing, technical, account, shipping, general>
URGENCY: <one of: low, medium, high>
SENTIMENT: <one of: neutral, frustrated, angry>

Ticket Subject: {subject}
Ticket Body: {body}"""

    response = llm.invoke(prompt)
    
    lines = response.content.strip().split("\n")
    
    result = {}
    for line in lines:
        if "CATEGORY:" in line:
            result["category"] = line.split(":")[1].strip().lower()
        elif "URGENCY:" in line:
            result["urgency"] = line.split(":")[1].strip().lower()
        elif "SENTIMENT:" in line:
            result["sentiment"] = line.split(":")[1].strip().lower()
    
    return result

@tool
def decide_escalation(category: str, urgency: str, sentiment: str) -> dict:
    """Decide whether the ticket should be escalated to a human agent."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    prompt = f"""You are a support ticket escalation system. Based on the ticket details below decide if this ticket needs a human agent or can be handled automatically.

Return ONLY the following format with no extra text:

ESCALATE: <yes or no>
REASON: <one sentence explanation>

Ticket Category: {category}
Urgency: {urgency}
Customer Sentiment: {sentiment}"""

    response = llm.invoke(prompt)
    
    lines = response.content.strip().split("\n")
    
    result = {}
    for line in lines:
        if "ESCALATE:" in line:
            value = line.split(":")[1].strip().lower()
            result["escalate"] = value == "yes"
        elif "REASON:" in line:
            result["escalation_reason"] = line.split(":", 1)[1].strip()
    
    return result