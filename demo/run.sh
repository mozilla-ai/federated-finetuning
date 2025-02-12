#!/bin/bash

# Define results directory
RESULTS_DIR="results"

echo "Running Federated Fine-Tuning..."
flwr run .

echo "Running Local Fine-Tuning..."
python src/fine-tune-local.py 

# Find the latest federated fine-tuning run (first folder in sorted list, excluding "_local")
LATEST_FEDERATED=$(ls -t "$RESULTS_DIR" | grep -v "_local" | head -n 1)

# Find the latest local fine-tuning run (folder containing "_local", sorted by timestamp)
LATEST_LOCAL=$(ls -t "$RESULTS_DIR" | grep "_local" | head -n 1)

# Validate that the necessary results exist
if [ -z "$LATEST_FEDERATED" ]; then
  echo "Error: No federated fine-tuned model found in $RESULTS_DIR"
  exit 1
fi

if [ -z "$LATEST_LOCAL" ]; then
  echo "Error: No local fine-tuned model found in $RESULTS_DIR"
  exit 1
fi

# Define paths
FEDERATED_PATH="$RESULTS_DIR/$LATEST_FEDERATED/peft_100/"
LOCAL_PATH="$RESULTS_DIR/$LATEST_LOCAL/checkpoint-100/"

echo "Evaluating Federated Fine-Tuning..."
python ./src/benchmarks/general-nlp/eval.py \
  --base-model-name-path=Qwen/Qwen2-0.5B-Instruct \
  --peft-path="$FEDERATED_PATH" \
  --run-name=qwen-fed \
  --batch-size=16 \
  --quantization=4 \
  --category=stem,social_sciences

python ./src/benchmarks/general-nlp/eval.py \
  --base-model-name-path=Qwen/Qwen2-0.5B-Instruct \
  --peft-path="$FEDERATED_PATH" \
  --run-name=qwen-fed \
  --batch-size=16 \
  --quantization=4 \
  --category=stem,social_sciences

python ./src/benchmarks/finance/eval.py \
  --base-model-name-path=Qwen/Qwen2-0.5B-Instruct \
  --peft-path="$FEDERATED_PATH" \
  --run-name=qwen-fed \
  --batch-size=16 \
  --quantization=4 \
  --datasets=fpb,fiqa,tfns

echo "Evaluating Local Fine-Tuning..."
python ./src/benchmarks/finance/eval.py \
  --base-model-name-path=Qwen/Qwen2-0.5B-Instruct \
  --peft-path="$LOCAL_PATH" \
  --run-name=qwen-local \
  --batch-size=16 \
  --quantization=4 \
  --datasets=fpb,fiqa,tfns

echo "Generating Visualizations..."
python ./src/plot_results.py  # Adjust if necessary

echo "All tasks completed!"