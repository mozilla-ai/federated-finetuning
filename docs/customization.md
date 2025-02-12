# 🎨 **Customization Guide**

This Blueprint is designed to be flexible and easily adaptable to your specific needs. This guide will walk you through some key areas you can customize to make the Blueprint your own.

---

## 🧠 **Changing the Model**

To swap out the default model for a different one, update the model name in `pyproject.toml` as long as it is a model from HuggingFace.

```bash
model.name = "<YOUR_CUSTOM_MODEL>"
```

Ensure that the new model supports the same fine-tuning methods as the original.

---

## 📝 **Modifying the Streamlit App**

Make your own application of a federated fine-tuned model and launch it in an application in the browser. Add graphics, new ways of prompting the model, and more.

---

## 💡 **Other Customization Ideas**

- Adjust **training parameters** in `pyproject.toml`, such as batch size, number of rounds, and learning rate.
- Modify **dataset preprocessing** in `src/flowertune_llm/dataset.py` to support different data formats.
- Implement **custom evaluation metrics** in `src/benchmarks/eval.py` to better assess performance based on your specific requirements.

---

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!

Moreover, join Flower's [Slack](https://flower.ai/join-slack/) where you can chat with the developers and maintainers of this Blueprint.
