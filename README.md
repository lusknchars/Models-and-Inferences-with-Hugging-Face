# 🤗 Loading Models and Inference with Hugging Face

> **Hands-on Lab** from IBM AI Engineering Professional Certificate

[![IBM Skills Network](https://img.shields.io/badge/IBM-Skills_Network-052FAD?style=flat&logo=ibm)](https://www.ibm.com/training/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.40.0-FFD21E?style=flat)](https://huggingface.co/transformers/)

---

## 📚 About This Lab

This repository contains practical exercises from the **IBM AI Engineering Professional Certificate**, specifically from the module on loading models and performing inference using the Hugging Face Transformers library.

**⚠️ Important:** This is a guided laboratory for educational purposes, not an original project. The goal is to learn and practice fundamental NLP concepts with pre-trained models.

---

## 🎯 Learning Objectives

In this laboratory, the following concepts were explored:

### 1️⃣ **Manual Inference (Low-Level Approach)**
- Manual loading of models and tokenizers
- Text preprocessing (tokenization)
- Running inference with `torch.no_grad()`
- Processing logits and model outputs
- Decoding results

### 2️⃣ **Pipeline API (High-Level Approach)**
- Using Hugging Face's `pipeline()` function
- Simplifying NLP tasks with just a few lines of code
- Comparing manual vs. automated approaches

---

## 🧪 Implemented Tasks

### **Text Classification (Sentiment Analysis)**
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Task:** Sentiment classification (POSITIVE/NEGATIVE)
- **Implementation:** Manual + Pipeline

### **Text Generation**
- **Model:** `gpt2`
- **Task:** Text generation from a prompt
- **Implementation:** Manual + Pipeline

### **Language Detection**
- **Model:** `papluca/xlm-roberta-base-language-detection`
- **Task:** Language detection for text inputs
- **Implementation:** Pipeline

### **Translation (Text-to-Text)**
- **Model:** `t5-small`
- **Task:** Translation from English to French
- **Implementation:** Pipeline

### **Fill-Mask**
- **Model:** `bert-base-uncased`
- **Task:** Filling in masked tokens
- **Implementation:** Pipeline

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12 | Programming language |
| **PyTorch** | 2.3.1 | Deep learning framework |
| **Transformers** | 4.40.0 | Pre-trained models library |
| **Torchvision** | 0.18.0 | Computer vision utilities |

---

## 📦 Installation

### Requirements
```bash
pip install torch~=2.3.1
pip install torchvision~=0.18.0
pip install transformers~=4.40.0
```

### Running the Notebook
```bash
jupyter notebook Loading_Models_and_Inference_with_Hugging_Face.ipynb
```

---

## 📖 Notebook Structure

```
1. Setup and Library Installation
2. Text Classification with DistilBERT
   ├── Manual model loading
   ├── Text preprocessing
   ├── Inference and output processing
   └── Implementation with pipeline()
3. Text Generation with GPT-2
   ├── Manual model loading
   ├── Step-by-step text generation
   └── Implementation with pipeline()
4. Hugging Face Pipeline API
   ├── Text Classification
   ├── Language Detection
   ├── Text Generation (T5)
   └── Fill-Mask (BERT)
5. Practical Exercise: Fill-Mask
```

---

## 💡 Key Learnings

### **Manual Approach vs. Pipeline**

| Aspect | Manual | Pipeline |
|--------|--------|----------|
| **Code** | ~15-20 lines | 3-5 lines |
| **Control** | Full | Limited |
| **Flexibility** | High | Medium |
| **Ease of use** | Requires knowledge | Very easy |
| **Recommended use** | Custom production | Rapid prototyping |

### **When to use each approach?**

✅ **Use `pipeline()` when:**
- Rapid prototyping
- Common NLP tasks
- Simplicity is priority
- Quick deployment

✅ **Use manual approach when:**
- Fine-grained control needed
- Performance optimization
- Custom tasks
- Complex integration

---

## 🔍 Code Examples

### Text Classification (Sentiment)

**Manual Approach:**
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

text = "I love this product!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1).item()
```

**With Pipeline:**
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## 📊 Results and Observations

### **Model Performance**

- **DistilBERT:** High accuracy in sentiment classification, fast
- **GPT-2:** Coherent generation, but with occasional repetitions
- **T5:** Excellent for translation tasks
- **BERT:** Great for fill-mask and MLM tasks

### **Efficiency Comparison**

| Task | Manual Time | Pipeline Time | Lines of Code (Manual) | Lines of Code (Pipeline) |
|------|-------------|---------------|------------------------|--------------------------|
| Classification | ~2-3s | ~1-2s | 15-20 | 3-5 |
| Generation | ~3-5s | ~2-3s | 20-25 | 3-5 |
| Translation | N/A | ~2-3s | N/A | 3-5 |

---

## 🎓 Certification

This laboratory is part of the **IBM AI Engineering Professional Certificate** offered via IBM Skills Network / Coursera.

**Course Topics:**
- Machine Learning with Python
- Deep Learning and Neural Networks
- Computer Vision
- Natural Language Processing
- Generative AI and LLMs

---

## 📝 Important Notes

1. **Hardware Requirements:** Some models may require GPU for optimized execution
2. **Download Time:** First execution downloads models (may take minutes)
3. **Warnings:** Warnings about unused weights are normal for fine-tuned models
4. **Versions:** Tested with Python 3.12 and specified libraries

---

## 🔗 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [IBM Skills Network](https://skills.network/)

---

## 📄 License

This material is part of IBM's educational content and is used for learning purposes only.

---

## ✍️ Author

Completed as part of the **IBM AI Engineering Professional Certificate** studies

**Context:** Guided laboratory - Learning exercise, not an original project

---

## 🏷️ Tags

`#IBM` `#AI` `#MachineLearning` `#NLP` `#HuggingFace` `#Transformers` `#PyTorch` `#DeepLearning` `#SentimentAnalysis` `#TextGeneration` `#Education` `#Certification`
