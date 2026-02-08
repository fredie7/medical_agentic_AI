import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, TypedDict, Sequence, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

DATAFRAME_STORE: Dict[str, pd.DataFrame ] = {}

@tool
def load_financial_data(file_path: str):
    """Load financial data from csv file into Dataframe store"""
    df = pd.read_csv(file_path)
    DATAFRAME_STORE["df"] = df
    return f"""
    Data loaded successfully
    Columns: {df.columns.tolist()}
    Rows: {len(df)}"""

@tool
def analyze_financial_data(revenue_column: str, expense_column: str) -> str:
    """Analyze financial data and return summary statistics"""
    if not DATAFRAME_STORE:
        return "No data loaded. Please load financial data"
    df = DATAFRAME_STORE.get("df") if DATAFRAME_STORE else None
    revenue = df[revenue_column]
    expenses = df[expense_column]
    profit = revenue - expenses
    summary = (
        f"Total Revenue: {revenue.sum():.2f}",
        f"Total Expenses: {expenses.sum():.2f}",
        f"Total Profit: {profit.sum():.2f}",
        f"Average revenue: {revenue.mean():.2f}"
    )
    return "\n".join(summary)

@tool
def plot_financials(time_column: str, revenue_column: str, expense_column: str):
    """Plot revenue and expense over time"""
    df = DATAFRAME_STORE.get("df")
    if not df:
        return "No data loaded. Please load data"

    plt.figure(figsize=(8,4))
    plt.plot(df[time_column], df[revenue_column], label="Revenue")
    plt.plot(df[time_column], df[expense_column], label="Expenses")
    plt.xlabel("Time")
    plt.ylabel("Amount")
    plt.title("Financial Overview")
    plt.legend()
    plt.show()
    return plt

tools = [load_financial_data, analyze_financial_data, plot_financials]

class FinancialPipelineState(TypedDict):
    messages = Annotated[Sequence[BaseMessage], add_messages]

def financial_agent(state: FinancialPipelineState) -> FinancialPipelineState:
    """Financial agent that orchestrates tool calls and executions"""
    system_prompt = """
    You are a financial analyst

    1. Load financial data from csv file.
    2. Identify revenue and expenses.
        Use actual column names from the loaded dataset.
        If unsure, choose the closest semantic match.
    3. Analyze financial data and provide summary statistics.
    4. Plot revenue and expenses over time.
    """
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ])
    state["messages"] += [AIMessage(content=response.content)]
    if response.tool_calls:
        for i, call in enumerate(response.tool_calls):
            print(f"{i+1}. {call["name"]} - {call["args"]}")
    return {"message": response}

def should_continue(state: FinancialPipelineState):
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return "end"
    return "continue"

graph=StateGraph(FinancialPipelineState)
graph.add_node("agent", financial_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")
financial_app = graph.compile()
