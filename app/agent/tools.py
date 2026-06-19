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
        limit=5  # was 2 — too few; the relevant policy section was often ranked 3rd-5th and never retrieved
    )

    docs = [point.payload["text"] for point in results.points]
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
- Identify missing required information to resolve the issue.
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
- Do NOT suggest solutions if not mentioned in documents
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
    """Decide whether the ticket should be escalated to a human agent.

    Deterministic rules over the classified labels. This is intentionally
    simple and conservative. The previous LLM version received only the three
    labels (category/urgency/sentiment), so it could never actually apply its
    detailed policy (it never saw transaction IDs, dispute signals, security
    details, etc.), and it escalated inconsistently — including routine,
    policy-resolvable tickets. Routine high-urgency requests are NOT escalated;
    we escalate only when a human clearly adds value.

    Tune the rules below to your product. To make escalation smarter, pass the
    ticket subject/body in and detect real signals (dispute, security, "speak
    to a human") rather than relying on coarse labels alone.
    """
    sentiment = (sentiment or "").lower().strip()

    if sentiment == "angry":
        return {
            "escalate": True,
            "escalation_reason": "Customer expressed strong frustration; routing to a human agent."
        }

    return {
        "escalate": False,
        "escalation_reason": "Issue can be handled from policy without human escalation."
    }