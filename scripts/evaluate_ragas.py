import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from ragas import evaluate, RunConfig
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.tools import search_knowledge_base

# Set up Gemini LLM wrapped for RAGAS
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
))

# Test tickets to send to the real agent
test_tickets = [
    {
        "subject": "I was charged twice for my subscription and need a refund",
        "body": "I noticed two charges of $29.99 from your company on the same day. I need the duplicate charge refunded immediately.",
        "reference": "Duplicate charges are eligible for a full refund. Customers should provide their transaction ID for processing within 3-5 business days."
    },
    {
        "subject": "I cannot log into my account",
        "body": "I have been trying to log into my account for two days but keep getting an error. I need urgent help.",
        "reference": "Customers unable to log in should try clearing cache and cookies or use a different browser. Support is available within 24 hours."
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
        
        # Get the actual agent response and retrieved docs
        generated_response = result.get("generated_response", "")
        
        # Also get retrieved contexts from Qdrant for this ticket
        query = f"{ticket['subject']} {ticket['body']}"
        retrieved_contexts = search_knowledge_base.invoke({"query": query})
        
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
    FactualCorrectness(llm=evaluator_llm)
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

print("\n" + "="*60)
print("RAGAS EVALUATION RESULTS — REAL AGENT RESPONSES")
print("="*60)

faithfulness = df['faithfulness'].mean() if 'faithfulness' in df.columns else None
factual = df['factual_correctness(mode=f1)'].mean() if 'factual_correctness(mode=f1)' in df.columns else None

print(f"Faithfulness       : {faithfulness:.2f}" if isinstance(faithfulness, float) else "Faithfulness       : could not compute")
print(f"Factual Correctness: {factual:.2f}" if isinstance(factual, float) else "Factual Correctness: could not compute")

print("="*60)
print("\nScore Guide:")
print("0.8 - 1.0 : Excellent")
print("0.6 - 0.8 : Good")
print("0.4 - 0.6 : Needs improvement")
print("0.0 - 0.4 : Poor")
print("="*60 + "\n")