# Import the necessary tools from LangChain and LangGraph
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os


# We define a simple Python dictionary to store workflow data (state)
# This state will be passed from one step (node) to another.
state = {"text": ""}

# Create an LLM object using OpenAI model
# This will be used in different workflow steps
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
print("OPENAI_API_KEY found.")

# Create an LLM object using OpenAI model
# This will be used in different workflow steps
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

# ---- Step 1: Topic detection ----
def detect_topic(state):
    """Detects the topic of the input text."""
    text = state["text"]  # extract input text
    response = llm.invoke(
        f"What is the main topic of this text? Keep it short.\n\n{text}"
    )
    state["topic"] = response.content  # save topic into state
    return state


# ---- Step 2: Summarization ----
def summarize_text(state):
    """Generates a short summary of the text."""
    text = state["text"]  
    response = llm.invoke(
        f"Give me a simple and short summary of this text:\n\n{text}"
    )
    state["summary"] = response.content  # save summary
    return state


# Build the LangGraph workflow
workflow = StateGraph(dict)

# Register workflow steps
workflow.add_node("topic_detection", detect_topic)
workflow.add_node("summarization", summarize_text)

# Define workflow order
workflow.set_entry_point("topic_detection")
workflow.add_edge("topic_detection", "summarization")
workflow.add_edge("summarization", END)

# Compile graph so we can run it
app = workflow.compile()

# ---- Run the workflow ----
input_text = """
Methane and carbon dioxide are two major greenhouse gases. 
Modern gas analyzers provide real-time emissions monitoring 
to support environmental research and regulatory compliance.
"""

# Add input text to the workflow state
state["text"] = input_text

# Execute the workflow
result = app.invoke(state)

# Print results
print("TOPIC DETECTED:")
print(result["topic"])
print("\nSUMMARY:")
print(result["summary"])
