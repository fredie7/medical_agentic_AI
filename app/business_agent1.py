import sys
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.tools import tool
from typing import Literal, Dict,TypedDict, Sequence, Annotated
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,ToolMessage,BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
sys.path.append(str(Path(__file__).resolve().parent.parent / "processed_data"))
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="Business Consulting Agent API")

# Allow frontend access (Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response Models
# -------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str



# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
print("OPENAI_API_KEY found.")


# Import data from warehouse
root_dir = Path(__file__).resolve().parent.parent.parent
categories = root_dir / "data_warehouse" / "processed_data" / "dim_categories.csv"
customers = root_dir / "data_warehouse" / "processed_data" / "dim_customers.csv"
dates = root_dir / "data_warehouse" / "processed_data" / "dim_dates.csv"
currencies = root_dir / "data_warehouse" / "processed_data" / "dim_currencies.csv"
transactions = root_dir / "data_warehouse" / "processed_data" / "fact_transactions.csv"

# print("Resolved path:", output_path)
# print("File exists:", output_path.exists())

dim_categories = pd.read_csv(categories)
dim_customers = pd.read_csv(customers)
dim_dates = pd.read_csv(dates)
dim_currencies = pd.read_csv(currencies)
fact_transactions = pd.read_csv(transactions)

# Join the Static dimensions with the fact table
fact_enriched = (
    fact_transactions
    .merge(dim_categories, on="category_key", how="left", validate="many_to_one")
    .merge(dim_currencies, on="currency_key", how="left", validate="many_to_one")
    .merge(dim_dates, on="date_key", how="left", validate="many_to_one")
)

# Enforce date types
fact_enriched["date"] = pd.to_datetime(fact_enriched["date"])
dim_customers["effective_from"] = pd.to_datetime(dim_customers["effective_from"])
dim_customers["effective_to"] = pd.to_datetime(dim_customers["effective_to"])

# Join the SCD Type-2 custmomer table
fact_customer_joined = fact_enriched.merge(
    dim_customers,
    on="customer_id",
    how="left",
    validate="many_to_many"
)

# Handle the open-ended SCD customer rows
fact_customer_joined["effective_to"] = (
    fact_customer_joined["effective_to"]
    .fillna(pd.Timestamp("2099-12-31"))
)

# Filter valid SCD2 customer record ----
business_data = fact_customer_joined[
    (fact_customer_joined["date"] >= fact_customer_joined["effective_from"]) &
    (fact_customer_joined["date"] <= fact_customer_joined["effective_to"])
]

# Quality check
# assert business_data.groupby("transaction_id").size().max() == 1

# Drop the columns that are not needed
business_data = (
    business_data
        .drop(columns=[
            "transaction_timestamp",
            "transaction_timestamp_y",
            "customer_key_x"
        ])
        .rename(columns={
            "transaction_timestamp_x": "transaction_timestamp",
            "customer_key_y": "customer_key"
        })
)

# -------------------------
# Policy documents (UNSTRUCTURED DATA)
# -------------------------
policy_docs = [
    Document(
        page_content="Refunds usually take 5-10 business days.",
        metadata={"source": "policy"},
    ),
    Document(
        page_content="Food items are non-refundable.",
        metadata={"source": "policy"},
    ),
    Document(
        page_content="Transactions above 500 EUR are flagged for review.",
        metadata={"source": "policy"},
    ),
    Document(
        page_content=(
            "Potential fraud indicators include high transaction frequency, "
            "unusual transaction amounts, and cross-border transactions."
        ),
        metadata={"source": "policy"},
    ),
]

# -------------------------
# Build The RAG Pipeline
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

policy_chunks = text_splitter.split_documents(policy_docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

vectorstore = FAISS.from_documents(policy_chunks, embeddings)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Build tools


@tool
def get_transaction_field(transaction_id: int, field: str) -> str:
    """
    Get a specific field for a transaction.
    Valid fields:
    transaction_id, customer_id, transaction_amount_eur,
    base_currency, transaction_currency, transaction_timestamp,
    category, country, signup_date, is_high_value_transaction
    """
    if field not in business_data.columns:
        return f"Invalid field. Available fields are: {list(business_data.columns)}"

    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    return str(row.iloc[0][field])

@tool
def get_customer_transactions(customer_id: int) -> str:
    """
    Get a summary of transactions for a given customer.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]

    if rows.empty:
        return "No transactions found for this customer."

    summary = rows[[
    "transaction_id",
    "transaction_amount_eur",
    "base_currency",
    "country",
    "category"
    ]]


    return summary.to_string(index=False)

@tool
def policy_lookup(question: str) -> str:
    """
    Answer questions about company policies such as refunds,
    returns, fraud rules, and transaction reviews.
    """
    result = retriever.invoke({"query": question})
    return result["result"]

@tool
def average_transaction_amount() -> str:
    """
    Get the average transaction amount in EUR.
    """
    avg = business_data["transaction_amount_eur"].mean()
    return f"The average transaction amount is {avg:.2f} EUR."

@tool
def get_transaction_summary(transaction_id: int) -> str:
    """
    Get a human-readable summary of a transaction.
    """
    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    r = row.iloc[0]

    return (
        f"Transaction {r.transaction_id}:\n"
        f"- Customer ID: {r.customer_id}\n"
        f"- Amount: {r.transaction_amount_eur} {r.base_currency}\n"
        f"- Category: {r.category}\n"
        f"- Country: {r.country}\n"
        f"- Date: {r.transaction_timestamp}\n"
        f"- Is current customer: {r.is_current}"
    )

@tool
def list_transaction_categories() -> str:
    """
    List all transaction categories.
    """
    categories = sorted(business_data["category"].dropna().unique())
    return "Transaction categories: " + ", ".join(categories)

@tool
def get_customer_spending_by_category(customer_id: int) -> str:
    """
    Get spending breakdown by category for a customer.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]

    if rows.empty:
        return "No transactions found."

    summary = (
        rows.groupby("category")["transaction_amount_eur"]
        .sum()
        .sort_values(ascending=False)
    )

    return summary.to_string()


@tool
def list_supported_countries() -> str:
    """
    List all countries where transactions have occurred.
    """
    countries = sorted(business_data["country"].dropna().unique())
    return "Supported countries: " + ", ".join(countries)

@tool
def check_high_value_transaction(transaction_id: int, threshold_eur: float = 500) -> str:
    """
    Check if transaction exceeds a EUR threshold.
    """
    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    transaction_amount_eur = row.iloc[0]["transaction_amount_eur"]

    if transaction_amount_eur > threshold_eur:
        return (
            f"Transaction {transaction_id} is high-value "
            f"({transaction_amount_eur} EUR) and may be reviewed."
        )
    
@tool
def check_cross_border(transaction_id: int) -> str:
    """
    Check if a transaction is cross-border.
    """
    row = business_data.loc[business_data["transaction_id"] == transaction_id]
    if row.empty:
        return "Transaction not found."

    country = row.iloc[0]["country"]

    return (
        f"Transaction {transaction_id} occurred in {country}. "
        "Cross-border transactions may require additional review."
    )

@tool
def list_supported_currencies() -> str:
    """
    List all supported currencies in the system.
    """
    currencies = sorted(business_data["transaction_currency"].dropna().unique())
    return "Supported currencies: " + ", ".join(currencies)

@tool
def get_recent_transactions(customer_id: int, limit: int = 5) -> str:
    """
    Get recent transactions for a customer.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]

    if rows.empty:
        return "No transactions found."

    rows = rows.sort_values("transaction_timestamp", ascending=False).head(limit)

    return rows[[
        "transaction_id",
        "transaction_amount_eur",
        "base_currency",
        "country",
        "category",
        "transaction_timestamp",
    ]].to_string(index=False)

def get_customer_profile(customer_id: int) -> str:
    """
    Get customer profile and activity summary.
    """
    rows = business_data.loc[business_data["customer_id"] == customer_id]
    if rows.empty:
        return "Customer not found."

    first = rows.iloc[0]

    return (
        f"Customer {customer_id}:\n"
        f"- Signup date: {first.signup_date}\n"
        f"- Total transactions: {len(rows)}\n"
        f"- Countries used: {rows.country.nunique()}"
    )

@tool
def high_value_by_spend() -> Dict:
    """
    Returns the top 5 customers with the highest total spend.
    """
    stats = business_data.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("transaction_amount_eur", "sum")
    ).reset_index()

    # Top 5 customers by spend
    top_customers = stats.sort_values("total_spend", ascending=False).head(5)
    
    return {
        "high_value_by_spend": top_customers[["customer_id", "total_spend"]].to_dict(orient="records")
    }

@tool
def platform_statistics() -> str:
    """
    Get high-level platform statistics.
    """
    return (
        f"Platform statistics:\n"
        f"- Total transactions: {len(business_data)}\n"
        f"- Total customers: {business_data['customer_id'].nunique()}\n"
        f"- Average amount (EUR): {business_data['transaction_amount_eur'].mean():.2f}\n"
        f"- Countries served: {business_data['country'].nunique()}"
    )

@tool
def high_value_by_frequency() -> Dict:
    """
    Returns the top 5 customers with the highest number of transactions.
    """
    stats = business_data.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("transaction_amount_eur", "sum")
    ).reset_index()

    # Top 5 customers by transaction count
    top_customers = stats.sort_values("transaction_count", ascending=False).head(5)
    
    return {
        "high_value_by_frequency": top_customers[["customer_id", "transaction_count"]].to_dict(orient="records")
    }


# List of tools to bind to the LLM for enhanced contextual interaction with the supervisory agent
tools = [
    get_transaction_field,
    get_transaction_summary,
    get_customer_profile,
    get_recent_transactions,
    get_customer_spending_by_category,
    check_cross_border,
    policy_lookup,
    list_supported_currencies,
    list_supported_countries,
    list_transaction_categories,
    average_transaction_amount,
    platform_statistics,
    high_value_by_frequency,
    high_value_by_spend,

]

# Bind the tools to the LLM to aid contextual interaction with the supervisory agent
llm = llm.bind_tools(tools) 

class BusinessConsultingAgentState(TypedDict):
    """Create a list of messages that the business consulting agent will use to interact with the supervisory agent."""
    messages: Annotated[Sequence[BaseMessage],add_messages]

def business_consulting_agent(state: BusinessConsultingAgentState) -> str:
    """
    Business consulting agent that interacts with the supervisory agent and uses tools to answer customer queries.
    """
    system_prompt = SystemMessage(
        content=(
            "You are a helpful and knowledgeable business consulting agent. "
            "You assist the supervisory agent by answering customer queries, "
            "providing transaction details, and explaining company policies. "
            "Use the available tools to retrieve information as needed."
        )
    )
        # Track the tool calls made by the supervisory agent to other agents
    print("[Agent] checking tools for calls...")
    
    # Invoke the LLM with the system prompt and the current messages in the state
    response = llm.invoke([system_prompt] + state['messages'])

    # Track Tools
    print("Checking for tool calls...")
    for i, tool_call in enumerate(response.tool_calls):
        # Print the tool name
        print(f" Tool #{i + 1}: {tool_call.get('name', 'UnknownTool')}")
        # Print the arguments passed to the tool
        print(f" Args: {tool_call.get('args', {})}")
   
    # Add the response to the messages in the state
    return {"messages": [response]}

def should_continue(state: BusinessConsultingAgentState) -> Literal["continue", "end"]:
    # Get the list of messages in the conversation
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"   
    
# Create a state graph to manage the conversation flow
graph = StateGraph(BusinessConsultingAgentState)

# Add the supervisory agent node to the graph
graph.add_node("business_consulting_agent",business_consulting_agent)

# Create a tool node to handle tool invocations
tool_node = ToolNode(tools=tools)

# Include the tool node in the graph
graph.add_node("tools",tool_node)

# Set the entry point of the graph to the supervisory agent
# This is where the conversation starts
graph.set_entry_point("business_consulting_agent")

# Add edges to the graph to control the flow of conversation
graph.add_conditional_edges(
    "business_consulting_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# Add an edge from the tool node back to the supervisory agent to continue the conversation where necessary
graph.add_edge("tools", "business_consulting_agent")

# Compile the graph to create the agent application
agent_app = graph.compile()
# print(agent_app)

# print("\nCustomer Care Agent is running.")
# print("Type 'exit' or 'quit' to stop.\n")

# while True:
#     user_input = input("USER: ").strip()

#     if user_input.lower() in {"exit", "quit"}:
#         print("Agent stopped. Goodbye")
#         break

#     try:
#         response = agent_app.invoke(
#             {"messages": [("user", user_input)]}
#         )

#         print("AGENT:", response["messages"][-1].content)

#     except Exception as e:
#         print("Error:", str(e))

# -------------------------
# Chat Endpoint
# -------------------------

# Create memory store for conversation history
conversation_store = {
    "session_id_1": [BaseMessage, BaseMessage, ...],
    "session_id_2": [...]
}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Create or reuse session
        session_id = request.session_id or str(uuid4())

        if session_id not in conversation_store:
            conversation_store[session_id] = []

        # Append user message
        conversation_store[session_id].append(
            HumanMessage(content=request.message)
        )

        # Invoke agent with full conversation
        result = agent_app.invoke(
            {"messages": conversation_store[session_id]}
        )

        ai_message = result["messages"][-1]

        # Store AI response
        conversation_store[session_id].append(ai_message)

        return ChatResponse(
            response=ai_message.content,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
