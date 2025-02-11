import streamlit as st
import torch
import argparse
import time
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Streamlit Federated Fine-Tuned Model Demo")
parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
parser.add_argument("--template", type=str, default="vicuna_v1.1", help="Conversation template")
args = parser.parse_args()

MODEL_PATH = args.model_path

# Load model and tokenizer (cached to avoid reloading on every request)
@st.cache_resource
def load_model(model_path):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    base_model = model.peft_config["default"].base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    return model, tokenizer

# Streamlit UI
st.title("Federated Fine-Tuned Model Demo")
st.write(f"Model loaded from: `{MODEL_PATH}`")
st.write("Ask a question and get a response from the fine-tuned OpenLLaMA model.")

# User input field
question = st.text_input("Enter your prompt:", "How does federated learning work?")
temperature = st.slider("Temperature (higher = more creative responses)", 0.1, 1.5, 0.7)

if st.button("Generate Response"):
    st.write("Generating response...")

    model, tokenizer = load_model(MODEL_PATH)

    prompt = question  # Directly use user input
    input_ids = tokenizer([prompt]).input_ids

    output_ids = model.generate(
        input_ids=torch.as_tensor(input_ids).to(model.device),
        do_sample=True,
        temperature=temperature,
        max_new_tokens=1024,
    )

    output_ids = (
        output_ids[0]
        if model.config.is_encoder_decoder
        else output_ids[0][len(input_ids[0]):]
    )

    # Streaming output with correct spacing
    output_placeholder = st.empty()
    output_text = ""

    for token_id in output_ids:
        token = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text += token + " "
        output_placeholder.write(output_text.strip())
        time.sleep(0.02)

    output_placeholder.write(output_text.strip())