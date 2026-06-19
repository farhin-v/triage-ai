import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from ragas import evaluate, RunConfig
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from app.agent.tools import search_knowledge_base
from typing import List

# Set up Gemini LLM wrapped for RAGAS
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
))

# Reuse the same embedding approach already proven to work in load_knowledge_base.py
# (the langchain_google_genai embedding wrapper hit model-name format errors earlier,
# so we call the google-genai SDK directly here too instead of reintroducing that wrapper)
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiRagasEmbeddings(BaseRagasEmbeddings):
    """Minimal RAGAS-compatible embeddings wrapper around the google-genai SDK,
    so ResponseRelevancy (which needs embeddings, not just an LLM) can run
    without depending on langchain_google_genai's embedding class."""

    def embed_query(self, text: str) -> List[float]:
        result = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

evaluator_embeddings = GeminiRagasEmbeddings()

# Test tickets to send to the real agent.
# NOTE: 3 tickets is too few for a stable average — one verbose reply swings the
# whole score. Add more tickets here (with accurate reference answers drawn from
# your actual policy docs) before trusting / reporting the numbers.
test_tickets = [
    {
        "subject": "I was charged twice for my subscription and need a refund",
        "body": "I noticed two charges of $29.99 from your company on the same day. I need the duplicate charge refunded immediately.",
        "reference": "Duplicate charges for the same transaction are refunded in full. The refund is processed within 3 to 5 business days after verification. The customer must provide at least one of: transaction ID, payment receipt, or bank statement showing the duplicate charge."
    },
    {
        "subject": "I cannot log into my account",
        "body": "I have been trying to log into my account for two days but keep getting an error. I need urgent help.",
        "reference": "Customers should use the forgot password link on the login page; a reset email is sent within 5 minutes. If it does not arrive, check spam, and if it still does not arrive, support must manually trigger a password reset from the admin panel."
    },
    {
        "subject": "Where is my order? It has been 2 weeks",
        "body": "I placed an order three weeks ago and it still has not arrived. The tracking page shows no updates.",
        "reference": "Standard delivery takes 5-7 business days. Orders delayed beyond this timeframe will be investigated and updated within 24 hours."
    }
]

print("\nCalling real agent for each test ticket...")
print("Make sure uvicorn is running on port 8000\n")

dataset_list = []

for ticket in test_tickets:
    print(f"Processing: {ticket['subject']}")

    try:
        response = requests.post(
            "http://localhost:8000/tickets/",
            json={
                "subject": ticket["subject"],
                "body": ticket["body"]
            },
            timeout=60
        )
        result = response.json()

        if "detail" in result:
            print(f"  Error: {result['detail']}")
            continue

        # Get the actual agent response
        generated_response = result.get("generated_response", "")
        category = result.get("category", "")

        # Match the agent's retrieval query EXACTLY — the agent prepends the
        # category (see retrieval_node in graph.py). Vector retrieval is
        # deterministic, so the same query returns the same contexts the agent
        # actually generated from. This is the core faithfulness fix: we now
        # grade the response against the evidence the agent really saw, not a
        # different re-retrieval.
        query = f"{category} {ticket['subject']} {ticket['body']}"
        retrieved_contexts = search_knowledge_base.invoke({"query": query})

        # Eyeball each reply: how much is policy content vs greeting/sign-off?
        print(f"  Category: {category}")
        print(f"  Response:\n{generated_response}\n")

        dataset_list.append({
            "user_input": ticket["subject"] + " " + ticket["body"],
            "response": generated_response,
            "retrieved_contexts": retrieved_contexts,
            "reference": ticket["reference"]
        })

        print(f"  Done — response length: {len(generated_response)} chars")

    except Exception as e:
        print(f"  Failed: {str(e)}")

if not dataset_list:
    print("\nNo results to evaluate. Make sure the API is running.")
    sys.exit(1)

print(f"\nEvaluating {len(dataset_list)} real agent responses...\n")

evaluation_dataset = EvaluationDataset.from_list(dataset_list)

metrics = [
    Faithfulness(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
]

run_config = RunConfig(
    max_workers=1,
    max_wait=60,
    timeout=120
)

results = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    run_config=run_config
)

df = results.to_pandas()

print("\n" + "=" * 70)
print("RAGAS EVALUATION RESULTS — REAL AGENT RESPONSES")
print("=" * 70)

metric_columns = {
    "Faithfulness": "faithfulness",
    "Factual Correctness": "factual_correctness(mode=f1)",
    "Context Precision (retrieval)": "llm_context_precision_without_reference",
    "Response Relevancy": "answer_relevancy",
}

scores = {}
for label, col in metric_columns.items():
    if col in df.columns:
        scores[label] = df[col].mean()
        print(f"{label:32}: {scores[label]:.2f}")
    else:
        print(f"{label:32}: could not compute (column '{col}' not in results)")

print("=" * 70)
print("\nWhat each metric tells you:")
print("  Faithfulness          -> generation: is the response grounded in retrieved docs?")
print("  Factual Correctness   -> generation: do the facts match the reference answer?")
print("  Context Precision     -> retrieval: was what we retrieved actually relevant?")
print("  Response Relevancy    -> generation: does the response address what was asked?")
print("\nDiagnostic pattern:")
print("  Low Context Precision + low Faithfulness -> fix retrieval/chunking first")
print("  High Context Precision + low Faithfulness -> fix the generation prompt")
print("  High Faithfulness + low Response Relevancy -> grounded, but answering the wrong thing")

print("\nScore Guide:")
print("0.8 - 1.0 : Excellent")
print("0.6 - 0.8 : Good")
print("0.4 - 0.6 : Needs improvement")
print("0.0 - 0.4 : Poor")
print("=" * 70 + "\n")