import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import requests

def process_ticket(subject, body):
    if not subject.strip() or not body.strip():
        return "Please enter both subject and body.", "", "", "", "", ""
    
    try:
        response = requests.post(
            "http://localhost:8000/tickets/",
            json={"subject": subject, "body": body}
        )
        result = response.json()
        
        if "detail" in result:
            return f"Error: {result['detail']}", "", "", "", "", ""
        
        category = result.get("category", "unknown").upper()
        urgency = result.get("urgency", "unknown").upper()
        sentiment = result.get("sentiment", "unknown").upper()
        escalated = "YES — Requires Human Agent" if result.get("escalated") else "NO — Handled Automatically"
        escalation_reason = result.get("escalation_reason", "N/A")
        generated_response = result.get("generated_response", "No response generated")
        
        return category, urgency, sentiment, escalated, escalation_reason, generated_response
    
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", ""

with gr.Blocks(title="Triage AI", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎯 Triage AI")
    gr.Markdown("### AI Powered Support Ticket Automation")
    gr.Markdown("Enter a support ticket below and let the AI agent classify, retrieve relevant knowledge, decide escalation, and generate a response.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Submit Ticket")
            subject_input = gr.Textbox(
                label="Ticket Subject",
                placeholder="e.g. I was charged twice for my subscription",
                lines=1
            )
            body_input = gr.Textbox(
                label="Ticket Body",
                placeholder="e.g. Hi, I noticed a duplicate charge on my account this month...",
                lines=6
            )
            submit_btn = gr.Button("Process Ticket", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("## Agent Analysis")
            category_output = gr.Textbox(label="Category", interactive=False)
            urgency_output = gr.Textbox(label="Urgency", interactive=False)
            sentiment_output = gr.Textbox(label="Customer Sentiment", interactive=False)
            escalated_output = gr.Textbox(label="Escalation Decision", interactive=False)
            escalation_reason_output = gr.Textbox(label="Escalation Reason", interactive=False)
    
    gr.Markdown("## Generated Response")
    response_output = gr.Textbox(
        label="AI Generated Customer Response",
        lines=10,
        interactive=False
    )
    
    gr.Markdown("### Example Tickets")
    gr.Examples(
        examples=[
            ["Charged twice for subscription", "Hi, I was charged twice for my monthly subscription this month. I need a refund for the duplicate charge. My account email is john@example.com"],
            ["Cannot log into my account", "I have been trying to log into my account for the past 2 days but keep getting an error. I need urgent help as I have an important deadline."],
            ["Where is my order?", "I placed an order 3 weeks ago and still have not received it. The tracking number shows no updates. I am very frustrated and need this resolved immediately."],
            ["How do I update billing info?", "I recently got a new credit card and would like to update my billing information. Can you help me with the steps?"]
        ],
        inputs=[subject_input, body_input]
    )
    
    submit_btn.click(
        fn=process_ticket,
        inputs=[subject_input, body_input],
        outputs=[category_output, urgency_output, sentiment_output, escalated_output, escalation_reason_output, response_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)