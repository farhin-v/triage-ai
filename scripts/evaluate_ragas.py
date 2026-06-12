import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate, RunConfig
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.tools import search_knowledge_base

# Set up Gemini LLM wrapped for RAGAS
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model_kwargs={"thinking_level": "low"}
))

# Build dataset using correct RAGAS 0.2.x EvaluationDataset format
# Faithfulness needs: user_input, response, retrieved_contexts
# FactualCorrectness needs: response, reference
dataset_list = [
    {
        "user_input": "I was charged twice for my subscription and need a refund",
        "response": "We apologize for the duplicate charge. According to our billing refund policy, duplicate charges are eligible for a full refund within 3-5 business days. Please provide your transaction ID and we will process the refund immediately.",
        "retrieved_contexts": search_knowledge_base.invoke({"query": "duplicate charge refund subscription"}),
        "reference": "Duplicate charges are eligible for a full refund. Customers should provide their transaction ID for processing within 3-5 business days."
    },
    {
        "user_input": "I cannot log into my account",
        "response": "We are sorry to hear you are having trouble logging in. Please try resetting your password using the forgot password link. If the issue persists our technical support team will assist you within 24 hours.",
        "retrieved_contexts": search_knowledge_base.invoke({"query": "cannot login account access"}),
        "reference": "Customers unable to log in should use the password reset option. Technical support is available within 24 hours."
    },
    {
        "user_input": "Where is my order? It has been 2 weeks",
        "response": "We apologize for the delay. According to our shipping policy standard delivery takes 5-7 business days. Since it has been 2 weeks we will investigate immediately and provide you with an update within 24 hours.",
        "retrieved_contexts": search_knowledge_base.invoke({"query": "order delay shipping policy"}),
        "reference": "Standard delivery takes 5-7 business days. Orders delayed beyond this timeframe will be investigated and updated within 24 hours."
    }
]

# Use correct EvaluationDataset class from RAGAS 0.2.x docs
evaluation_dataset = EvaluationDataset.from_list(dataset_list)

# Both metrics only need LLM — no embeddings needed at all
metrics = [
    Faithfulness(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm)
]

# Rate limit config — max_workers=1 prevents parallel calls
run_config = RunConfig(
    max_workers=1,
    max_wait=60,
    timeout=120
)

# Run evaluation
print("\nRunning RAGAS evaluation...")
print("This may take a few minutes due to rate limiting...\n")

results = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    run_config=run_config
)

# Access scores using _repr_dict as shown in official docs
scores = results._repr_dict

print("\n" + "="*60)
print("RAGAS EVALUATION RESULTS")
print("="*60)

faithfulness = scores.get("faithfulness")
factual = scores.get("factual_correctness(mode=f1)")

print(f"Faithfulness        : {faithfulness:.2f}" if isinstance(faithfulness, float) else "Faithfulness        : could not compute")
print(f"Factual Correctness : {factual:.2f}" if isinstance(factual, float) else "Factual Correctness : could not compute")

print("="*60)
print("\nScore Guide:")
print("0.8 - 1.0 : Excellent")
print("0.6 - 0.8 : Good")
print("0.4 - 0.6 : Needs improvement")
print("0.0 - 0.4 : Poor")
print("="*60 + "\n")