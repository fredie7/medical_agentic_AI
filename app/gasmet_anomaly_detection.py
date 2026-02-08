import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import statistics
import random
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

# ----------------------------------------------------------
# STEP 0: FAKE DATA GENERATION (SIMULATING GAS ANALYZER)
# ----------------------------------------------------------

def generate_fake_sensor_data():
    """
    Generate fake methane and CO2 readings, similar to what a
    Gasmet analyzer might output. Includes occasional spikes.
    """
    data = []

    for _ in range(10):
        reading = {
            "methane_ppm": random.uniform(1.8, 2.5),
            "co2_ppm": random.uniform(390, 420)
        }

        # Add a spike 20% of the time
        if random.random() < 0.2:
            reading["methane_ppm"] += random.uniform(1, 3)
            reading["co2_ppm"] += random.uniform(50, 120)

        data.append(reading)

    return data


# ----------------------------------------------------------
# SETUP AI MODEL
# ----------------------------------------------------------

llm = ChatOpenAI(model="gpt-4.1-mini")


# ----------------------------------------------------------
# STEP 1: PREPROCESSING
# ----------------------------------------------------------

def preprocess_data(state):
    """
    Basic preprocessing:
    - Extract methane + CO2 arrays
    - Calculate mean + standard deviation
    - Add normalized values for easier anomaly detection
    """

    data = state["sensor_data"]

    # Extract lists of methane and CO2 values only
    methane_values = [d["methane_ppm"] for d in data]
    co2_values = [d["co2_ppm"] for d in data]

    # Calculate basic statistics
    methane_mean = statistics.mean(methane_values)
    co2_mean = statistics.mean(co2_values)

    methane_std = statistics.stdev(methane_values)
    co2_std = statistics.stdev(co2_values)

    # Add normalized values (z-score) for each reading
    for d in data:
        d["methane_norm"] = (d["methane_ppm"] - methane_mean) / methane_std
        d["co2_norm"] = (d["co2_ppm"] - co2_mean) / co2_std

    # Store preprocessing results
    state["preprocessed"] = {
        "methane_mean": methane_mean,
        "methane_std": methane_std,
        "co2_mean": co2_mean,
        "co2_std": co2_std,
        "data": data
    }

    return state


# ----------------------------------------------------------
# STEP 2: AI-BASED ANOMALY DETECTION
# ----------------------------------------------------------

def detect_anomalies(state):
    """
    The AI looks at:
    - Raw values
    - Normalized values
    - Means and standard deviations
    And makes a judgment about anomalies.
    """

    pre = state["preprocessed"]
    data = pre["data"]

    # Build a readable text block for the AI
    rows = ""
    for i, r in enumerate(data):
        rows += (
            f"Reading {i+1}: "
            f"CH4={r['methane_ppm']:.2f} ppm (norm {r['methane_norm']:.2f}), "
            f"CO2={r['co2_ppm']:.2f} ppm (norm {r['co2_norm']:.2f})\n"
        )

    prompt = f"""
    Here is greenhouse gas data from a gas analyzer.

    The normalized values represent how many standard deviations
    a reading is above or below the mean.

    Data:
    {rows}

    Based on normalized values, identify:
    - Which readings appear anomalous (e.g., norm > 2 or < -2)
    - Whether methane or CO2 spikes exist
    - A beginner-friendly explanation (no technical jargon)
    """

    result = llm.invoke(prompt)

    state["anomaly_analysis"] = result.content
    return state


# ----------------------------------------------------------
# STEP 3: AI-GENERATED REPORTING
# ----------------------------------------------------------

def generate_report(state):
    """
    The AI writes a short, simple final report summarizing:
    - What happened
    - Which spikes were found
    - Why it matters for greenhouse gas monitoring
    """

    analysis = state["anomaly_analysis"]

    prompt = f"""
    You are creating a final report for a Gasmet-style analyzer test.

    Summarize:
    - What anomalies were detected
    - Why the spikes might matter in greenhouse gas research
    - Keep the explanation simple for a beginner researcher

    Use short paragraphs.
    Here is the analysis from the previous step:

    {analysis}
    """

    result = llm.invoke(prompt)

    state["final_report"] = result.content
    return state


# ----------------------------------------------------------
# STEP 4: EXPORT TO CSV
# ----------------------------------------------------------

def export_to_csv(state):
    """
    Writes the dataset (raw + normalized + rule anomaly tag)
    into a CSV file.
    """
    data = state["preprocessed"]["data"]

    filename = "gas_readings.csv"

    # Write CSV
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "methane_ppm",
            "co2_ppm",
            "methane_norm",
            "co2_norm",
            "rule_anomaly"
        ])

        # Each data row
        for d in data:
            writer.writerow([
                d["methane_ppm"],
                d["co2_ppm"],
                d["methane_norm"],
                d["co2_norm"],
                d["rule_anomaly"]
            ])

    state["csv_file"] = filename
    return state


# ----------------------------------------------------------
# STEP 5: PLOT VISUALIZATION
# ----------------------------------------------------------

def create_plot(state):
    """
    Creates a simple plot with methane and CO2 trends.
    Uses matplotlib defaults and avoids specifying colors.
    """
    data = state["preprocessed"]["data"]

    methane = [d["methane_ppm"] for d in data]
    co2 = [d["co2_ppm"] for d in data]
    x = list(range(1, len(data) + 1))

    plt.figure(figsize=(10, 5))

    # Default line colors, per instructions
    plt.plot(x, methane, label="Methane (ppm)")
    plt.plot(x, co2, label="CO2 (ppm)")

    plt.title("Gas Analyzer Readings (Fake Data)")
    plt.xlabel("Reading Number")
    plt.ylabel("Gas Concentration (ppm)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("gas_plot.png")  # Save plot to file

    state["plot_file"] = "gas_plot.png"
    return state

# ----------------------------------------------------------
# BUILD THE MULTI-STEP WORKFLOW
# ----------------------------------------------------------

# Create a workflow with dictionary state
wf = StateGraph(dict)

# Add the three steps (nodes)
wf.add_node("preprocess_data", preprocess_data)
wf.add_node("detect_anomalies", detect_anomalies)
wf.add_node("generate_report", generate_report)

# Set flow: preprocessing → detection → reporting
wf.set_entry_point("preprocess_data")
wf.add_edge("preprocess_data", "detect_anomalies")
wf.add_edge("detect_anomalies", "generate_report")

# End after report generation
wf.add_edge("generate_report", END)

# Compile workflow
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

print("=== FINAL REPORT ===")
print(result["final_report"])
