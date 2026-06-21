# Triage AI — Agentic Support Ticket Automation

An AI agent that handles customer support tickets end to end — classifying, retrieving relevant policy context, deciding escalation, and generating grounded responses — without human intervention for standard cases.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/Farhinv/triage-ai)  
**Live API:** [triage-ai-cylb.onrender.com/docs](https://triage-ai-cylb.onrender.com/docs)

---

## What It Does

Submit a support ticket (subject + body). The agent:

1. **Classifies** — category (billing / technical / account / shipping / general), urgency (low / medium / high), sentiment (neutral / frustrated / angry)
2. **Retrieves** — fetches relevant policy chunks from a 121-chunk Qdrant vector store using semantic search
3. **Decides** — determines whether the ticket needs human escalation or can be resolved automatically
4. **Generates** — writes a grounded, policy-backed customer response using Gemini
5. **Saves** — stores ticket, classification, and response to PostgreSQL; displays on Gradio dashboard

---

## Architecture

```
Incoming Ticket (subject + body)
        │
        ▼
┌─────────────────────────────────────────┐
│           LangGraph Agent (5 nodes)      │
│                                         │
│  classify → retrieve → escalation →     │
│  generate → save                        │
└─────────────────────────────────────────┘
        │
        ├── Qdrant Cloud (vector store, 121 chunks)
        ├── Google Gemini (embeddings + generation)
        ├── PostgreSQL (ticket storage)
        └── LangSmith (tracing + observability)

Frontend: Gradio (HuggingFace Spaces)
Backend:  FastAPI (Render)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Google Gemini (gemini-3.1-flash-lite) |
| Embeddings | Google Gemini Embeddings (gemini-embedding-001, 3072-dim) |
| Vector Store | Qdrant Cloud |
| Relational DB | PostgreSQL (Render) |
| Backend | FastAPI |
| Frontend | Gradio |
| Evaluation | RAGAS |
| Observability | LangSmith |
| Deployment | Docker, Render, HuggingFace Spaces |

---

## RAG Evaluation (RAGAS)

| Metric | Score | Notes |
|---|---|---|
| Context Precision | 0.80 | Retrieval quality — excellent |
| Response Relevancy | 0.85 | Response addresses the question — excellent |
| Faithfulness | 0.41 | Penalized by conversational structure (greetings, problem restatement) — not a retrieval failure |
| Factual Correctness | 0.31 | Same limitation as Faithfulness — generation metrics not designed for dialogue agents |

**Note on generation metrics:** RAGAS Faithfulness and Factual Correctness assume tightly grounded, extractive answers. A support agent that restates the customer's problem, opens with a greeting, and closes with a sign-off will score low on these metrics even when responses are accurate. Retrieval quality (Context Precision, Response Relevancy) is the more reliable signal for this architecture.

---

## Key Engineering Decisions

**Why Google Gemini Embeddings instead of sentence-transformers?**  
Render free tier has a 512MB RAM limit. sentence-transformers exceeded this and crashed the service. Switching to the Gemini Embeddings API kept the system live at zero additional cost with no loss in retrieval quality.

**Why top_k=5 for retrieval?**  
Knowledge base documents are chunked by numbered section — approximately 5 chunks per file. Testing top_k values from 2 to 14 showed that 2 chunks often missed full policy context while values above 5 added noise without improving response quality. 5 chunks consistently covered one complete policy document.

**Why chunk by section headings instead of fixed token count?**  
Fixed-length chunking splits policy points mid-sentence and loses the document title and heading context. Section-based chunking preserves each chunk's semantic unit — title, heading, content — which directly improved retrieval precision.

**Why Render PostgreSQL instead of Supabase?**  
Render free tier cannot reach Supabase's IPv6-only addresses. Using Render's own PostgreSQL (same region as the web service) eliminated the connectivity issue entirely.

---

## Project Structure

```
triage-ai/
├── app/
│   ├── agent/          # LangGraph graph, state, tools
│   ├── api/            # FastAPI routes
│   ├── database/       # models, connections
│   ├── dashboard/      # Gradio UI
│   └── main.py
├── data/
│   └── knowledge_base/ # 20 policy .txt files across 5 categories
├── scripts/
│   ├── load_knowledge_base.py  # chunks and ingests docs into Qdrant
│   ├── evaluate_ragas.py       # RAGAS evaluation pipeline
│   ├── debug_retrieval.py      # retrieval inspection tool
│   ├── debug_response.py       # response quality inspection
│   ├── debug_tools.py          # LangGraph tool debugging
│   ├── test_agent.py           # end-to-end agent test
│   └── test_db.py              # database connection test
├── app.py              # Gradio dashboard entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running Locally

**Prerequisites:** Python 3.11+, Docker (optional)

```bash
git clone https://github.com/farhin-v/triage-ai
cd triage-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up your `.env`:
```
GOOGLE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
DATABASE_URL=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
```

Load the knowledge base (first time only):
```bash
python scripts/load_knowledge_base.py
```

Start the backend:
```bash
uvicorn app.main:app --reload
```

Start the dashboard:
```bash
python app.py
```

---

## Known Constraints

- **Cold starts:** Hosted on Render free tier — first request after inactivity takes 30–60 seconds. Dashboard has a 90-second timeout with a user-friendly message.
- **RAGAS evaluation cost:** Evaluation runs were limited to 3 tickets due to free-tier API budget. Retrieval metrics (Context Precision, Response Relevancy) are the primary quality signal.