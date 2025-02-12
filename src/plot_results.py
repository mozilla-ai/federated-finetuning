import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Path where results are stored
RESULTS_PATH = "./benchmarks/"

# Function to extract accuracy from text files
def extract_accuracy(file_path):
    with open(file_path, "r") as f:
        content = f.read().strip()

    # Extract accuracy value using regex
    match = re.search(r"Accuracy: ([0-9.]+)", content)
    if match:
        try:
            return float(match.group(1).rstrip("."))  # Strip trailing dot if present
        except ValueError:
            print(f"Warning: Could not convert accuracy value from {file_path}")
            return None
    return None

# Collect accuracy results
results = []
for file in os.listdir(RESULTS_PATH):
    if file.startswith("acc_") and file.endswith(".txt"):
        file_path = os.path.join(RESULTS_PATH, file)
        acc = extract_accuracy(file_path)

        if acc is not None:
            # Extract dataset and run type from filename
            parts = file.replace(".txt", "").split("_")  # Remove .txt before splitting
            dataset_name = "_".join(parts[1:-1])  # All parts except first (`acc`) and last (`run type`)
            run_name = parts[-1]  # The last part is the run type (e.g., `qwen-fed`)

            results.append((dataset_name, run_name, acc))

# Convert results to a Pandas DataFrame
df = pd.DataFrame(results, columns=["Dataset", "Run", "Accuracy"])

# Check if DataFrame is empty
if df.empty:
    print("No valid results found to plot.")
    exit()

# **Fix duplicate dataset-run combinations by averaging their accuracy**
df = df.groupby(["Dataset", "Run"], as_index=False).agg({"Accuracy": "mean"})

# Pivot the data: index = Dataset, columns = Run, values = Accuracy
pivot_df = df.pivot(index="Dataset", columns="Run", values="Accuracy")

# Create the plot
plt.figure(figsize=(12, 6))
pivot_df.plot(kind="bar", ax=plt.gca())

# Add plot labels and title
plt.title("Accuracy by Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Dataset")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Run", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save the plot
plot_path = os.path.join(RESULTS_PATH, "accuracy_plot_all_datasets.png")
plt.savefig(plot_path)
print(f"✅ Saved combined plot: {plot_path}")

# Show the plot
plt.show()