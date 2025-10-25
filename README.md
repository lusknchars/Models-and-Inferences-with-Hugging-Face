# 🤗 Loading Models and Inference with Hugging Face

> **Laboratório prático** do curso IBM AI Engineering Professional Certificate

[![IBM Skills Network](https://img.shields.io/badge/IBM-Skills_Network-052FAD?style=flat&logo=ibm)](https://www.ibm.com/training/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.40.0-FFD21E?style=flat)](https://huggingface.co/transformers/)

---

## 📚 Sobre este Lab

Este repositório contém exercícios práticos do **IBM AI Engineering Professional Certificate**, especificamente do módulo sobre carregamento de modelos e inferência usando a biblioteca Hugging Face Transformers.

**⚠️ Importante:** Este é um laboratório guiado para fins educacionais, não um projeto original. O objetivo é aprender e praticar conceitos fundamentais de NLP com modelos pré-treinados.

---

## 🎯 Objetivos de Aprendizado

Neste laboratório, foram explorados os seguintes conceitos:

### 1️⃣ **Inferência Manual (Baixo Nível)**
- Carregamento manual de modelos e tokenizers
- Pré-processamento de texto (tokenização)
- Execução de inferência com `torch.no_grad()`
- Processamento de logits e saídas do modelo
- Decodificação de resultados

### 2️⃣ **Pipeline API (Alto Nível)**
- Uso da função `pipeline()` do Hugging Face
- Simplificação de tarefas de NLP com poucas linhas de código
- Comparação entre abordagens manual vs. automatizada

---

## 🧪 Tarefas Implementadas

### **Text Classification (Sentiment Analysis)**
- **Modelo:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Tarefa:** Classificação de sentimentos (POSITIVE/NEGATIVE)
- **Implementação:** Manual + Pipeline

### **Text Generation**
- **Modelo:** `gpt2`
- **Tarefa:** Geração de texto a partir de um prompt
- **Implementação:** Manual + Pipeline

### **Language Detection**
- **Modelo:** `papluca/xlm-roberta-base-language-detection`
- **Tarefa:** Detecção de idioma de textos
- **Implementação:** Pipeline

### **Translation (Text-to-Text)**
- **Modelo:** `t5-small`
- **Tarefa:** Tradução de inglês para francês
- **Implementação:** Pipeline

### **Fill-Mask**
- **Modelo:** `bert-base-uncased`
- **Tarefa:** Preenchimento de tokens mascarados
- **Implementação:** Pipeline

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| **Python** | 3.12 | Linguagem de programação |
| **PyTorch** | 2.3.1 | Framework de deep learning |
| **Transformers** | 4.40.0 | Biblioteca de modelos pré-treinados |
| **Torchvision** | 0.18.0 | Utilitários de visão computacional |

---

## 📦 Instalação

### Requisitos
```bash
pip install torch~=2.3.1
pip install torchvision~=0.18.0
pip install transformers~=4.40.0
```

### Executar o Notebook
```bash
jupyter notebook Loading_Models_and_Inference_with_Hugging_Face.ipynb
```

---

## 📖 Estrutura do Notebook

```
1. Setup e Instalação de Bibliotecas
2. Text Classification com DistilBERT
   ├── Carregamento manual do modelo
   ├── Pré-processamento do texto
   ├── Inferência e processamento de saídas
   └── Implementação com pipeline()
3. Text Generation com GPT-2
   ├── Carregamento manual do modelo
   ├── Geração de texto passo a passo
   └── Implementação com pipeline()
4. Hugging Face Pipeline API
   ├── Text Classification
   ├── Language Detection
   ├── Text Generation (T5)
   └── Fill-Mask (BERT)
5. Exercício Prático: Fill-Mask
```

---

## 💡 Principais Aprendizados

### **Abordagem Manual vs. Pipeline**

| Aspecto | Manual | Pipeline |
|---------|--------|----------|
| **Código** | ~15-20 linhas | 3-5 linhas |
| **Controle** | Total | Limitado |
| **Flexibilidade** | Alta | Média |
| **Facilidade** | Requer conhecimento | Muito fácil |
| **Uso recomendado** | Produção customizada | Prototipagem rápida |

### **Quando usar cada abordagem?**

✅ **Use `pipeline()` quando:**
- Prototipando rapidamente
- Tarefas comuns de NLP
- Simplicidade é prioridade
- Deploy rápido

✅ **Use abordagem manual quando:**
- Precisa de controle fino
- Otimização de performance
- Tarefas customizadas
- Integração complexa

---

## 🔍 Exemplos de Código

### Text Classification (Sentimento)

**Abordagem Manual:**
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

**Com Pipeline:**
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## 📊 Resultados e Observações

### **Performance dos Modelos**

- **DistilBERT:** Alta acurácia em classificação de sentimentos, rápido
- **GPT-2:** Geração coerente, porém com repetições ocasionais
- **T5:** Excelente para tarefas de tradução
- **BERT:** Ótimo para fill-mask e MLM tasks

### **Comparação de Eficiência**

| Tarefa | Tempo Manual | Tempo Pipeline | Linhas de Código (Manual) | Linhas de Código (Pipeline) |
|--------|--------------|----------------|---------------------------|----------------------------|
| Classificação | ~2-3s | ~1-2s | 15-20 | 3-5 |
| Geração | ~3-5s | ~2-3s | 20-25 | 3-5 |
| Tradução | N/A | ~2-3s | N/A | 3-5 |

---

## 🎓 Certificação

Este laboratório faz parte do **IBM AI Engineering Professional Certificate** oferecido via IBM Skills Network / Coursera.

**Tópicos do Curso:**
- Machine Learning com Python
- Deep Learning e Neural Networks
- Computer Vision
- Natural Language Processing
- Generative AI e LLMs

---

## 📝 Notas Importantes

1. **Requisitos de Hardware:** Alguns modelos podem requerer GPU para execução otimizada
2. **Tempo de Download:** Primeira execução baixa os modelos (pode levar minutos)
3. **Warnings:** Warnings sobre weights não utilizados são normais para modelos fine-tuned
4. **Versões:** Testado com Python 3.12 e bibliotecas especificadas

---

## 🔗 Recursos Adicionais

- [Documentação Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [IBM Skills Network](https://skills.network/)

---

## 📄 Licença

Este material é parte do conteúdo educacional da IBM e é usado apenas para fins de aprendizado.

---

## ✍️ Autor

Desenvolvido como parte dos estudos para a **IBM AI Engineering Professional Certificate**

**Contexto:** Laboratório guiado - Exercício de aprendizado, não projeto original

---

## 🏷️ Tags

`#IBM` `#AI` `#MachineLearning` `#NLP` `#HuggingFace` `#Transformers` `#PyTorch` `#DeepLearning` `#SentimentAnalysis` `#TextGeneration` `#Education` `#Certification`
