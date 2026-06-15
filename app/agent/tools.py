from langchain_core.tools import tool
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from typing import List
import os

@tool
def search_knowledge_base(query: str) -> List[str]:
    """Search the Qdrant knowledge base for documents relevant to the query."""
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", None)
    )

    genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_vector = result.embeddings[0].values

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
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = f"""You are an expert customer support triage agent for a SaaS subscription company.

Your job is to analyze customer support tickets and extract structured classification for routing and escalation.

You MUST rely ONLY on the customer ticket content provided below.
Do NOT use external knowledge, assumptions, or inferred system behavior.

--------------------------------------------------

STRICT GROUNDING RULE:
- If something is not explicitly stated in the ticket, mark it as UNKNOWN in reasoning.
- Do NOT guess causes, solutions, or system behavior.

--------------------------------------------------

STEP 1 — UNDERSTAND THE TICKET
- Identify the PRIMARY ISSUE (what is going wrong).
- Identify the CUSTOMER GOAL (what the customer wants resolved).

STEP 2 — IMPACT ANALYSIS (ONLY from ticket text)
Evaluate only what is explicitly mentioned:
- Service disruption (yes/no/unknown)
- Financial impact (yes/no/unknown)
- Account access issue (yes/no/unknown)
- Security concern (yes/no/unknown)

STEP 3 — COMPLETENESS CHECK
- Identify missing required information.
- Do NOT assume missing details.

--------------------------------------------------

CLASSIFICATION RULES:

CATEGORY (choose one only):
- billing
- technical
- account
- shipping
- general

URGENCY:
- high: service blocked, security risk, financial loss, or explicit urgent request
- medium: issue affects usage but partial access exists
- low: informational request, how-to question, or minor issue

SENTIMENT:
- angry (explicit frustration, blame, urgency, threats)
- frustrated (confused, dissatisfied, repeated issue)
- neutral (calm, informational)

--------------------------------------------------

CRITICAL RULES:
- Do NOT suggest solutions
- Do NOT add troubleshooting steps
- Do NOT infer missing system behavior
- Do NOT assume intent beyond what is written
- Treat all customer claims as unverified facts

--------------------------------------------------

OUTPUT FORMAT (STRICT):

PRIMARY_ISSUE: <short issue>
CUSTOMER_GOAL: <what customer wants>
CATEGORY: <billing/technical/account/shipping/general>
URGENCY: <high/medium/low>
SENTIMENT: <angry/frustrated/neutral>

Ticket Subject:
{subject}

Ticket Body:
{body}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        content = response.content
    elif isinstance(response.content, list):
        content = response.content[0]["text"]
    else:
        content = str(response.content)

    lines = content.strip().split("\n")
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
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = f"""You are an expert escalation decision engine for a SaaS customer support system.

Your job is to decide whether a ticket MUST be escalated to a human agent or can be handled automatically.

You must strictly follow company escalation policies and customer impact rules.

--------------------------------------------------

ANALYSIS PROCESS:

STEP 1 — Understand the issue clearly from inputs
STEP 2 — Identify customer impact (financial, security, access, frustration)
STEP 3 — Identify automation eligibility
STEP 4 — Identify risk level (security, financial, legal, reputational)
STEP 5 — Check if information is missing and could block resolution

--------------------------------------------------

AUTOMATION CAN HANDLE:
- General policy or informational queries
- Standard account operations (upgrade, downgrade, cancellation)
- Password/login troubleshooting (basic cases)
- Refund requests WITH transaction ID and clear eligibility
- Shipping/order status (standard cases)

--------------------------------------------------

HUMAN AGENT REQUIRED IF ANY APPLY:
- Suspected unauthorized access or account compromise
- Billing disputes where customer denies purchase
- Payment deducted but service not activated (uncertain cases)
- Refund requests involving high-value or unclear eligibility
- Legal threats or compliance-related complaints
- Repeated unresolved issues
- Explicit request to speak to a human agent
- Any case involving security or financial risk

--------------------------------------------------

DECISION LOGIC (STRICT PRIORITY):

1. If security risk = YES → ALWAYS escalate
2. If unauthorized access suspected → ALWAYS escalate
3. If billing dispute with denial → escalate
4. If urgency = high AND sentiment = angry → escalate
5. If urgency = high AND involves billing/security → escalate
6. If urgency = medium AND involves security → escalate
7. Otherwise → do NOT escalate

--------------------------------------------------

IMPORTANT RULES:
- Do NOT rely only on category/urgency labels
- Always prioritize customer risk and system risk
- If uncertain → escalate
- Better to escalate than miss a critical issue

--------------------------------------------------

OUTPUT FORMAT (STRICT — NO EXTRA TEXT):

ESCALATE: <yes or no>
REASON: <one clear sentence explaining why>

Ticket Category: {category}
Urgency: {urgency}
Customer Sentiment: {sentiment}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        content = response.content
    elif isinstance(response.content, list):
        content = response.content[0]["text"]
    else:
        content = str(response.content)

    lines = content.strip().split("\n")
    result = {}
    for line in lines:
        if "ESCALATE:" in line:
            value = line.split(":")[1].strip().lower()
            result["escalate"] = value == "yes"
        elif "REASON:" in line:
            result["escalation_reason"] = line.split(":", 1)[1].strip()
    return result