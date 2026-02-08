# ----------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (OpenAI API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

# Initialize the LLM (AI model)
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# STEP 1: DEFINE SAMPLE CUSTOMER TICKETS
# ----------------------------------------------------------
# These are example customer requests
tickets = [
    "My internet has been down all morning, and I can't work!",
    "I forgot my password, please help me reset it.",
    "I need a report for last month's sales figures.",
    "System crashed and I can't access any files!",
    "How can I change my billing information?"
]

# ----------------------------------------------------------
# STEP 2: FUNCTION TO CLASSIFY URGENCY
# ----------------------------------------------------------
def classify_ticket_urgency(ticket_text):
    """
    This function asks the AI to classify a ticket
    into urgency levels: High, Medium, Low.
    """
    # Instruction for AI
    prompt = f"""
    You are a customer support assistant.
    Classify the following customer request into urgency levels:
    - High: Needs immediate attention
    - Medium: Important but not urgent
    - Low: Can wait or is informational

    Ticket:
    {ticket_text}

    Respond with only the urgency level: High, Medium, or Low.
    """

    # Call the AI model
    response = llm.invoke(prompt)

    # Return AI's answer (urgency level)
    return response.content.strip()


# ----------------------------------------------------------
# STEP 3: CLASSIFY ALL TICKETS
# ----------------------------------------------------------
print("=== Customer Ticket Urgency Classification ===")
for i, ticket in enumerate(tickets, 1):
    urgency = classify_ticket_urgency(ticket)
    print(f"Ticket {i}: {ticket}")
    print(f"→ Urgency: {urgency}\n")
