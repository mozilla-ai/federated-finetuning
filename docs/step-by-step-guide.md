# **Step-by-Step Guide: How this Blueprint Works**

---

## **Overview**
This guide walks you through the process of setting up and using the Federated Fine-Tuning Blueprint with Flower. It covers prerequisites, installation, running fine-tuning, and customizing your workflow.

---

## **Step 1: Prerequisites**
### System Requirements
- **OS:** Linux
- **Python Version:** 3.10 or higher
- **Minimum RAM:** 8GB (recommended for LLM fine-tuning)
- **Dependencies:** Listed in `pyproject.toml`

---

## **Step 2: Installation & Setup**
### Clone the Project
```bash
git clone https://github.com/mozilla-ai/blueprint-federated-finetuning.git
cd blueprint-federated-finetuning
```

### Install Dependencies
```bash
pip install -e .  # Install root project dependencies
```

---

## **Step 3: Running Federated Fine-Tuning**
### Run with the Simulation Engine (Recommended)
```bash
flwr run . 

# Run for 10 rounds with 25% client participation per round
flwr run . --run-config "num-server-rounds=10 strategy.fraction-fit=0.25"
```

> **Note:** Learn more about simulations in the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html).

### Run with the Deployment Engine
Follow the [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run this app with Flower's Deployment Engine.

---

## 🎨 **Customizing the Blueprint**
To better understand how you can tailor this Blueprint to suit your specific needs, please visit the **[Customization Guide](customization.md)**.

---

## 🤝 **Contributing to the Blueprint**
Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!

