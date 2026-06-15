import requests
import json

ticket = {
    "subject": "I was charged twice for my subscription",
    "body": "Hi, I noticed that I was charged twice for my monthly subscription this month. I need a refund for the duplicate charge."
}

response = requests.post("http://localhost:8000/tickets/", json=ticket)
print(json.dumps(response.json(), indent=2))