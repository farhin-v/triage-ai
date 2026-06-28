"""
Triage AI — Ticket Simulation Script
Sends 30 realistic support tickets to the local backend.
Run your FastAPI backend first: uvicorn app.main:app --reload
Then run: python simulate_tickets.py
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000/tickets/"

tickets = [
    # --- BILLING (7 tickets) ---
    {
        "subject": "Charged twice for my subscription this month",
        "body": "Hi, I noticed two identical charges on my credit card for this month's subscription. Both are for the same amount and same date. I need one of them refunded. My account email is sarah.jones@email.com."
    },
    {
        "subject": "Payment failed but I was still charged",
        "body": "I tried to renew my subscription yesterday and the payment page showed an error saying the transaction failed. But I can see the charge on my bank statement. My account is still showing as inactive. Please help."
    },
    {
        "subject": "Incorrect charge after plan downgrade",
        "body": "I downgraded my plan from Pro to Basic last week. I was told no partial refund would be given for the current period which I understood. But this month I was still charged the Pro rate instead of the Basic rate. This needs to be corrected immediately."
    },
    {
        "subject": "Prorated charge not matching what I expected",
        "body": "I upgraded my plan midway through the billing cycle and the prorated charge on my invoice does not match what I calculated. I upgraded on the 15th of the month but the charge seems to be for the full month. Can you explain how the proration was calculated?"
    },
    {
        "subject": "Subscription renewed even though I cancelled",
        "body": "I cancelled my subscription three weeks ago and received a confirmation email. But I have just been charged for another month. I want an immediate refund and confirmation that my account is cancelled."
    },
    {
        "subject": "Cannot update my payment method",
        "body": "I am trying to update my credit card details in the billing section but every time I click save I get an error message saying payment method could not be saved. My current card is expiring next week and I do not want my service to be interrupted."
    },
    {
        "subject": "Invoice shows wrong plan name",
        "body": "My invoice for this month shows I am on the Enterprise Plan but I am on the Pro Plan. I am worried I might be charged Enterprise rates. Please check my account and send me a corrected invoice."
    },

    # --- TECHNICAL (8 tickets) ---
    {
        "subject": "Cannot access my account despite correct password",
        "body": "I have been unable to log in to my account since this morning. I am entering the correct password and have not changed it recently. I tried resetting the password but the reset email is not arriving. I have checked my spam folder. This is urgent as I need access for work."
    },
    {
        "subject": "Complete loss of service for the past 2 hours",
        "body": "Our entire team has been unable to access the platform since 9am this morning. None of us can log in. We are getting a 503 error on the login page. This is affecting our entire operation. We need this resolved immediately."
    },
    {
        "subject": "App crashing on iOS every time I open it",
        "body": "The mobile app on my iPhone has been crashing immediately on launch since the update yesterday. I have tried reinstalling the app twice and restarting my phone. I am on iOS 17.4 and iPhone 14. The app crashes within 5 seconds of opening."
    },
    {
        "subject": "Data appears to be missing from my account",
        "body": "I logged in this morning and several months of records that I had saved in the platform are no longer visible. I have not deleted anything. This is critical data that I need for a client presentation tomorrow. Please investigate urgently."
    },
    {
        "subject": "Feature not working after account upgrade",
        "body": "I upgraded to the Pro plan this morning specifically to access the advanced reporting feature. The upgrade went through and I was charged but the advanced reporting tab is still not showing in my account. It has been 4 hours."
    },
    {
        "subject": "Getting error 500 when trying to export data",
        "body": "Every time I try to export my data to CSV I get an internal server error. This has been happening for the past 3 days. I need to export this data for our monthly reporting. I have tried different browsers and clearing cache. Nothing works."
    },
    {
        "subject": "Two factor authentication not sending codes",
        "body": "I enabled two factor authentication last week and it was working fine. Since yesterday I am not receiving the SMS codes on my phone. I have tried requesting the code multiple times. My phone number is correct in the settings. I am completely locked out."
    },
    {
        "subject": "Page loads but shows blank white screen after login",
        "body": "After I log in the page loads but shows a completely blank white screen. No error message. This started happening this afternoon. I have tried Chrome and Firefox and incognito mode. Same result on all. My colleague on the same account can log in fine."
    },

    # --- SHIPPING (6 tickets) ---
    {
        "subject": "Order marked as delivered but never received",
        "body": "My order shows as delivered on the tracking page since two days ago but I have not received anything. I have checked with my neighbours and building reception. Nobody received a package on my behalf. I need this resolved as it was a time sensitive order."
    },
    {
        "subject": "Package arrived damaged",
        "body": "My order arrived today but the box was clearly crushed and the item inside is broken. I have taken photos of the damaged packaging and the damaged product. I want either a replacement or a full refund. How do I send you the photos?"
    },
    {
        "subject": "Wrong item delivered",
        "body": "I ordered the Standard Kit but received what appears to be the Basic Kit instead. The packaging has a different product code to what I ordered. I have my original order confirmation showing what I ordered. I need the correct item sent and instructions for returning the wrong one."
    },
    {
        "subject": "Order significantly delayed with no update",
        "body": "My order was supposed to arrive within 5 business days and it has now been 9 business days. The tracking has not updated in 4 days and just shows in transit. I have an important event next week that I needed this for. What is happening with my shipment?"
    },
    {
        "subject": "Tracking number not working",
        "body": "I received my shipping confirmation email with a tracking number but when I enter it on the courier website it says the tracking number is invalid or not found. My order was placed 3 days ago. Can you check if the shipment has actually been dispatched?"
    },
    {
        "subject": "Order shipped to wrong address",
        "body": "I updated my delivery address in my account settings before placing the order but my confirmation email shows it is being shipped to my old address. I moved 2 months ago and nobody lives there anymore. Can this be intercepted and redirected?"
    },

    # --- ACCOUNT (5 tickets) ---
    {
        "subject": "Want to downgrade my plan before next billing cycle",
        "body": "I want to downgrade from the Pro plan to the Basic plan. I understand I will keep my current features until the end of this billing period. I just want to confirm the downgrade will take effect from next month and I will not lose my current data immediately."
    },
    {
        "subject": "How do I cancel my account completely",
        "body": "I would like to cancel my account entirely and have all my data deleted. I have read that data deletion happens within 30 days of cancellation. Can you confirm the process, what happens to my data, and whether I will receive confirmation once everything is deleted?"
    },
    {
        "subject": "Enterprise plan upgrade request",
        "body": "Our company would like to upgrade to the Enterprise plan. I understand this cannot be done through the self-service portal. Can you connect us with our account manager? Our company name is Brightfield Solutions and our current account ID is BF-20291."
    },
    {
        "subject": "Account features disappeared after plan change",
        "body": "I downgraded my account last billing cycle and I understood some features would be removed. However I seem to have lost access to data that should still be accessible for 30 days according to your policy. It has only been 10 days since my downgrade. Please restore my access."
    },
    {
        "subject": "Cannot create a new account, email already in use",
        "body": "I am trying to create a new account but keep getting an error saying my email address is already registered. I do not remember ever creating an account with this email. I have tried the forgot password option but the reset email is not arriving. Can you check what is on file for my email?"
    },

    # --- GENERAL (4 tickets) ---
    {
        "subject": "Request for a copy of all my personal data",
        "body": "Under your privacy policy I understand I have the right to request a copy of all data you hold about me. I would like to exercise this right. Please send me a complete export of all personal data associated with my account including support history, usage data, and billing information."
    },
    {
        "subject": "Request to delete all my data",
        "body": "I have cancelled my account and I am now formally requesting deletion of all my personal data as per your privacy policy. I understand this will be completed within 30 days. Please confirm receipt of this request and send me a confirmation once the deletion is complete."
    },
    {
        "subject": "Concerned about how my data is being used",
        "body": "I recently read some news about data privacy and I want to understand exactly what data you collect about me and how it is used. Specifically I want to know if my data is ever shared with or sold to third parties, and how my payment information is stored."
    },
    {
        "subject": "Feedback about the support experience",
        "body": "I wanted to share some feedback about my recent support interaction. The agent I spoke with was helpful but the wait time was very long. I also think it would be useful to have a live chat option rather than only email support. I hope this feedback is useful for improving the service."
    },
]


def send_ticket(ticket, index, total):
    try:
        response = requests.post(BASE_URL, json=ticket, timeout=60)
        result = response.json()
        print(f"\n[{index}/{total}] ✓ Ticket sent")
        print(f"  Subject   : {ticket['subject'][:60]}")
        print(f"  Category  : {result.get('category', 'N/A')}")
        print(f"  Urgency   : {result.get('urgency', 'N/A')}")
        print(f"  Escalated : {result.get('escalated', 'N/A')}")
        print(f"  Status    : {result.get('status', 'N/A')}")
    except requests.exceptions.Timeout:
        print(f"\n[{index}/{total}] ✗ Timeout — ticket may still be processing")
    except Exception as e:
        print(f"\n[{index}/{total}] ✗ Error: {e}")


def main():
    print("=" * 60)
    print("TRIAGE AI — TICKET SIMULATION")
    print(f"Sending {len(tickets)} tickets to {BASE_URL}")
    print("Make sure your FastAPI backend is running first.")
    print("=" * 60)

    for i, ticket in enumerate(tickets, 1):
        send_ticket(ticket, i, len(tickets))
        if i < len(tickets):
            time.sleep(10)  # wait between calls to avoid rate limits

    print("\n" + "=" * 60)
    print(f"DONE — {len(tickets)} tickets sent")
    print("Check LangSmith for traces and your dashboard for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()