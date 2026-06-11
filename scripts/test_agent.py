import requests
import json

ticket = {
    "subject": "I was charged twice for my subscription",
    "body": "Hi, I noticed that I was charged twice for my monthly subscription this month. I need a refund for the duplicate charge. My account email is john@example.com"
}

response = requests.post("http://localhost:8000/tickets/", json=ticket)
result = response.json()

print("\n" + "="*60)
print("TRIAGE AI — TICKET RESULT")
print("="*60)
print(f"Ticket ID  : {result['ticket_id']}")
print(f"Category   : {result['category']}")
print(f"Urgency    : {result['urgency']}")
print(f"Sentiment  : {result['sentiment']}")
print(f"Status     : {result['status']}")
print(f"Escalated  : {result['escalated']}")
print(f"Reason     : {result['escalation_reason']}")
print("\n--- GENERATED RESPONSE ---\n")
print(result['generated_response'])
print("="*60 + "\n")
