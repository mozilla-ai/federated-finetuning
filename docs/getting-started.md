# **Getting Started with the Blueprint**

Get started with this Blueprint using one of the options below:

---

### **Option 1:**
Follow this option if you want to quickly set up and run the Blueprint with minimal configuration.

1. Clone the repository:
```bash
git clone https://github.com/mozilla-ai/blueprint-federated-finetuning.git
cd blueprint-federated-finetuning
```
2. Install dependencies:
```bash
pip install -e .
```
3. Run the default simulation:
```bash
flwr run .
```

---

### **Option 2:**
Choose this option if you want to customize the setup for specific requirements.

1. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/mozilla-ai/blueprint-federated-finetuning.git
cd blueprint-federated-finetuning
```
2. Modify `pyproject.toml` to adjust parameters like number of rounds, client participation, and resource allocation.
3. Install dependencies and run the customized setup:
```bash
pip install -e .
flwr run . --run-config "num-server-rounds=10 strategy.fraction-fit=0.25"
```

For more details on customization, refer to the **[Customization Guide](customization.md)**.
