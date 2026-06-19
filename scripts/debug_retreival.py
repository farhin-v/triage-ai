import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.agent.tools import search_knowledge_base

# Same 3 test tickets used in evaluate_ragas.py
test_tickets = [
    {
        "subject": "I was charged twice for my subscription and need a refund",
        "body": "I noticed two charges of $29.99 from your company on the same day. I need the duplicate charge refunded immediately.",
    },
    {
        "subject": "I cannot log into my account",
        "body": "I have been trying to log into my account for two days but keep getting an error. I need urgent help.",
    },
    {
        "subject": "Where is my order? It has been 2 weeks",
        "body": "I placed an order three weeks ago and it still has not arrived. The tracking page shows no updates.",
    }
]

for ticket in test_tickets:
    print("\n" + "=" * 70)
    print(f"TICKET: {ticket['subject']}")
    print("=" * 70)

    query = f"{ticket['subject']} {ticket['body']}"
    retrieved_contexts = search_knowledge_base.invoke({"query": query})

    print(f"Retrieved {len(retrieved_contexts)} chunk(s):\n")

    for i, chunk in enumerate(retrieved_contexts, 1):
        # Each chunk is "TITLE\nHEADING\ncontent..." per the section-based
        # chunking in load_knowledge_base.py — print just title + heading
        # so it's easy to scan, then a short preview of the content.
        lines = chunk.strip().split("\n")
        title = lines[0] if len(lines) > 0 else "(no title)"
        heading = lines[1] if len(lines) > 1 else "(no heading)"
        preview = " ".join(lines[2:])[:120] if len(lines) > 2 else ""

        print(f"  [{i}] {title} — {heading}")
        print(f"      \"{preview}...\"")

print("\n" + "=" * 70)
print("Look for: is the section that actually answers each ticket present?")
print("If a clearly relevant section is missing, that's a recall problem —")
print("the limit is too low, or the embedding similarity isn't ranking it")
print("highly enough to make the cut.")
print("=" * 70 + "\n")