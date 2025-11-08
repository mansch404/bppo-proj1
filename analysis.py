import pm4py
import pandas as pd
import matplotlib.pyplot as plt

# --- Load the log ---
log = pm4py.read_xes("bpi-chall.xes")

if not isinstance(log, pd.DataFrame):
    df = pm4py.convert_to_dataframe(log)
else:
    df = log.copy()

df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

# --- (1) Number of cases ---
num_cases = df["case:concept:name"].nunique()

# --- (2) Number of events ---
num_events = len(df)

# --- (3) Number of process variants ---
variants = pm4py.statistics.variants.log.get.get_variants(log)
num_variants = len(variants)

# --- (4) Number of case and event labels ---
num_case_labels = df["case:concept:name"].nunique()
num_event_labels = df["concept:name"].nunique()

# --- (5) Case length (number of events per case) ---
case_lengths = df.groupby("case:concept:name").size()
mean_case_length = case_lengths.mean()
std_case_length = case_lengths.std()

# --- (6) Case duration statistics ---
case_durations = (
    df.groupby("case:concept:name")["time:timestamp"]
    .agg(["min", "max"])
    .assign(duration=lambda x: (x["max"] - x["min"]).dt.total_seconds())
)
mean_case_duration = case_durations["duration"].mean()
std_case_duration = case_durations["duration"].std()

# --- (7) Number of categorical event attributes ---
categorical_event_attrs = [
    col for col in df.columns
    if not pd.api.types.is_numeric_dtype(df[col]) and col not in ["time:timestamp"]
]
num_categorical_event_attrs = len(categorical_event_attrs)

# --- (8) Additional simple metrics ---
num_unique_resources = df["org:resource"].nunique() if "org:resource" in df.columns else 0
num_total_event_attrs = len(df.columns)

print("=== Process Log Metrics ===")
print(f"Number of cases: {num_cases}")
print(f"Number of events: {num_events}")
print(f"Number of process variants: {num_variants}")
print(f"Number of case labels: {num_case_labels}")
print(f"Number of event labels: {num_event_labels}")
print(f"Mean case length: {mean_case_length:.2f}")
print(f"Std dev of case length: {std_case_length:.2f}")
print(f"Mean case duration (s): {mean_case_duration:.2f}")
print(f"Std dev of case duration (s): {std_case_duration:.2f}")
print(f"Number of categorical event attributes: {num_categorical_event_attrs}")
print(f"Number of unique resources: {num_unique_resources}")
print(f"Total number of event attributes: {num_total_event_attrs}")

# --- (Optional) Plot case duration distribution ---
plt.hist(case_durations["duration"], bins=50)
plt.title("Distribution of Case Durations (seconds)")
plt.xlabel("Duration (s)")
plt.ylabel("Frequency")
plt.show()
