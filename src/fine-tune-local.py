import os
import toml
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load configuration from pyproject.toml
def load_config(toml_path="pyproject.toml"):
    config = toml.load(toml_path)
    app_config = config["tool"]["flwr"]["app"]["config"]
    model_name = app_config["model"]["name"]
    dataset_name = app_config["dataset"]["name"]
    num_rounds = app_config["num-server-rounds"]
    num_clients = config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
    return model_name, dataset_name, num_clients, num_rounds

# Read model, dataset name, and number of clients
MODEL_NAME, DATASET_NAME, NUM_CLIENTS, NUM_ROUNDS = load_config()

# Create output directory with timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_PATH = f"results/{current_time}_local"
os.makedirs(SAVE_PATH, exist_ok=True)

# Load dataset
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)
subset_size = len(dataset["train"]) // NUM_CLIENTS  # ✅ Use only 1/n of the dataset
dataset["train"] = dataset["train"].select(range(subset_size))
print(f"Using {subset_size} samples out of {len(dataset['train'])} for fine-tuning.")

# Rename "output" to "response" if needed
if "output" in dataset["train"].column_names:
    dataset = dataset.rename_column("output", "response")

# Formatting function
def formatting_prompts_func(example):
    return [f"### Instruction:\n{example['instruction']}\n### Response: {example['response']}"]

# Load tokenizer
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForCompletionOnlyLM(tokenizer.encode("\n### Response:", add_special_tokens=False)[2:], tokenizer=tokenizer)

# Load model with LoRA adapters
print(f"Loading model: {MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    torch_dtype=torch.bfloat16
)
base_model = prepare_model_for_kbit_training(base_model)

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.075, 
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(base_model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=NUM_ROUNDS,
    save_steps=5,
    logging_dir=f"{SAVE_PATH}/logs",
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
)

# Fine-tune the model
print("Starting centralized fine-tuning on 1/n of the dataset...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],  # ✅ Uses only 1/n fraction of dataset
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
)
trainer.train()

# ✅ Save PEFT Adapter & Tokenizer
adapter_path = f"{SAVE_PATH}/peft_adapter"
print(f"Saving fine-tuned LoRA adapter to {adapter_path}")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("✅ Fine-tuning complete. Adapter saved successfully.")