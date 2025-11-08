import pm4py
import pandas as pd
import matplotlib.pyplot as plt

# Load the log
log = pm4py.read_xes("bpi-chall.xes")

if isinstance(log, pd.DataFrame):
    df = log
else:
    df = pm4py.convert_to_dataframe(log)

# (a) Number of cases
num_cases = len(log)
print(f"Number of cases: {num_cases}")

# (b) Number of process variants
from pm4py.algo.filtering.log.variants import variants_filter
variants = pm4py.statistics.variants.log.get.get_variants(log)
print(f"Number of process variants: {len(variants)}")

# (c) Number of events / activities / resources
num_events = len(df)
unique_activities = df["concept:name"].nunique()
unique_resources = df["org:resource"].nunique() if "org:resource" in df.columns else 0

print(f"Number of events: {num_events}")
print(f"Number of unique activities: {unique_activities}")
print(f"Number of unique resources: {unique_resources}")

print(f"Number of events: {num_events}")
print(f"Number of unique activities: {unique_activities}")
print(f"Number of unique resources: {unique_resources}")

# --- (d) Case durations ---
# Convert to pandas dataframe to easily compute durations
df = pm4py.convert_to_dataframe(log)

# Ensure timestamps are parsed correctly
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

# Compute start and end times per case
case_durations = (
    df.groupby("case:concept:name")["time:timestamp"]
    .agg(["min", "max"])
    .assign(duration=lambda x: (x["max"] - x["min"]).dt.total_seconds())
)

avg_duration = case_durations["duration"].mean()
std_duration = case_durations["duration"].std()

print(f"Average case duration: {avg_duration:.2f} seconds")
print(f"Std dev of case duration: {std_duration:.2f} seconds")

# --- Plot distribution to see long vs short cases ---
plt.hist(case_durations["duration"], bins=50)
plt.title("Distribution of Case Durations (seconds)")
plt.xlabel("Duration (s)")
plt.ylabel("Frequency")
plt.show()