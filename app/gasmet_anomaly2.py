import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import statistics
import random

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

# LLM model
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)


# ----------------------------------------------------------
# STEP 0: FAKE DATA GENERATION (SIMULATING GAS ANALYZER)
# ----------------------------------------------------------

def generate_fake_sensor_data():
    """
    Generate fake methane + CO2 readings.
    Spikes occur 20% of the time.
    """
    data = []

    for _ in range(10):
        reading = {
            "methane_ppm": random.uniform(1.8, 2.5),
            "co2_ppm": random.uniform(390, 420)
        }

        # Add occasional spike
        if random.random() < 0.2:
            reading["methane_ppm"] += random.uniform(1, 3)
            reading["co2_ppm"] += random.uniform(50, 120)

        data.append(reading)

    return data


# ----------------------------------------------------------
# STEP 1: PREPROCESS DATA
# ----------------------------------------------------------

def preprocess_data(state):
    """
    Compute mean, std, and normalize.
    Also detect rule-based anomalies before AI.
    """

    data = state["sensor_data"]

    # Extract separate lists
    methane_list = [d["methane_ppm"] for d in data]
    co2_list = [d["co2_ppm"] for d in data]

    # Basic statistics
    methane_mean = statistics.mean(methane_list)
    methane_std = statistics.stdev(methane_list)

    co2_mean = statistics.mean(co2_list)
    co2_std = statistics.stdev(co2_list)

    # Normalize + rule anomaly
    anomalies = 0  
    for d in data:
        d["methane_norm"] = (d["methane_ppm"] - methane_mean) / methane_std
        d["co2_norm"] = (d["co2_ppm"] - co2_mean) / co2_std

        # Simple rule: values > 2 std dev are anomalies
        d["rule_anomaly"] = (
            abs(d["methane_norm"]) > 2 or abs(d["co2_norm"]) > 2
        )
        if d["rule_anomaly"]:
            anomalies += 1

    # Save results + count
    state["preprocessed"] = {
        "data": data,
        "anomaly_count": anomalies,
        "methane_mean": methane_mean,
        "methane_std": methane_std,
        "co2_mean": co2_mean,
        "co2_std": co2_std
    }

    return state


# ----------------------------------------------------------
# TOOL: RISK LEVEL CALCULATOR (BEGINNER FRIENDLY)
# ----------------------------------------------------------

def calculate_risk_level(anomaly_count: int):
    """
    Simple 'tool' to classify risk levels.
    In real projects, this could call an API, DB, etc.
    """
    if anomaly_count == 0:
        return "No risk"
    elif anomaly_count <= 2:
        return "Low risk"
    elif anomaly_count <= 4:
        return "Medium risk"
    else:
        return "High risk"


# ----------------------------------------------------------
# STEP 2: AI DETECTS ANOMALIES
# ----------------------------------------------------------

def detect_anomalies(state):
    """
    AI reviews data and explains findings.
    """
    pre = state["preprocessed"]
    data = pre["data"]

    rows = ""
    for i, r in enumerate(data):
        rows += (
            f"Reading {i+1}: CH4={r['methane_ppm']:.2f} (norm {r['methane_norm']:.2f}), "
            f"CO2={r['co2_ppm']:.2f} (norm {r['co2_norm']:.2f})\n"
        )

    prompt = f"""
    Here is greenhouse gas data from a gas analyzer.
    Identify anomalies using normalized values.
    Data:
    {rows}
    Provide a simple explanation for a beginner.
    """

    result = llm.invoke(prompt)
    state["ai_analysis"] = result.content

    return state


# ----------------------------------------------------------
# STEP 3A: NORMAL REPORT (IF ANOMALIES FOUND)
# ----------------------------------------------------------

def generate_report(state):
    analysis = state["ai_analysis"]

    risk = calculate_risk_level(state["preprocessed"]["anomaly_count"])

    prompt = f"""
    Create a final report for greenhouse gas monitoring.

    Include:
    - Summary of anomalies
    - Risk level: {risk}
    - Beginner-friendly explanation

    Data analysis:
    {analysis}
    """

    result = llm.invoke(prompt)
    state["final_report"] = result.content
    return state


# ----------------------------------------------------------
# STEP 3B: CLEAN REPORT (IF NO ANOMALIES)
# ----------------------------------------------------------

def generate_clean_report(state):
    """
    Called when no anomalies exist.
    """
    risk = calculate_risk_level(0)

    prompt = f"""
    Generate a clean, beginner-friendly report.

    Explain:
    - No anomalies detected
    - Risk level: {risk}
    - Why stable methane/CO2 is important
    """

    result = llm.invoke(prompt)
    state["final_report"] = result.content
    return state


# ----------------------------------------------------------
# CONDITIONAL BRANCHING FUNCTION
# ----------------------------------------------------------

def anomaly_condition(state):
    """
    Returns True if anomalies exist, False otherwise.
    LangGraph uses this for branching.
    """
    return state["preprocessed"]["anomaly_count"] > 0


# ----------------------------------------------------------
# BUILD MULTI-BRANCH WORKFLOW
# ----------------------------------------------------------

wf = StateGraph(dict)

# Add nodes
wf.add_node("preprocess_data", preprocess_data)
wf.add_node("detect_anomalies", detect_anomalies)
wf.add_node("generate_report", generate_report)
wf.add_node("generate_clean_report", generate_clean_report)

# Entry point
wf.set_entry_point("preprocess_data")

# After preprocessing, always detect anomalies
wf.add_edge("preprocess_data", "detect_anomalies")

# Branching based on condition:
# IF anomalies → generate full report
# ELSE → generate clean report
wf.add_conditional_edges(
    "detect_anomalies",
    anomaly_condition,
    {True: "generate_report", False: "generate_clean_report"}
)

wf.add_edge("generate_report", END)
wf.add_edge("generate_clean_report", END)

# Compile graph
app = wf.compile()


# ----------------------------------------------------------
# RUN WORKFLOW
# ----------------------------------------------------------

fake_data = generate_fake_sensor_data()
initial_state = {"sensor_data": fake_data}
result = app.invoke(initial_state)


# ----------------------------------------------------------
# PRINT OUTPUT
# ----------------------------------------------------------

print("\n=== FINAL REPORT ===")
print(result["final_report"])
