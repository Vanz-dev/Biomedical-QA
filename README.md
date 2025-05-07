# Biomedical Question Answering with PubMedQA and PubMedBERT

This project implements a biomedical question answering pipeline using the PubMedQA dataset and PubMedBERT. It uses BM25 for context retrieval and fine-tunes a pretrained biomedical transformer to classify answers. We also evaluate semantic similarity using Sentence-BERT.

## 🔍 Overview

**Key components:**
- **Dataset**: `pubmed_qa` (`pqa_labeled` subset)
- **Retrieval**: BM25 for sentence/document selection
- **Model**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **Evaluation**: Accuracy, Macro F1 Score, Semantic Similarity using Sentence-BERT

## ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/Vanz-dev/Biomedical-QA.git
```

Install requirements:

```bash
pip install -r requirements.txt
```

## 📦 Usage
Open and manually execute the Jupyter Notebook: qa_final.ipynb

## 📁 Dataset
We use the Hugging Face datasets library to load the PubMedQA dataset.
```python
load_dataset("pubmed_qa", "pqa_labeled")
```

## 🧠 Results Summary
| Model       | Accuracy | Macro F1 | Semantic Similarity |
|------------|---------|---------|--------------------|
| Pretrained | 0.1750   | 0.1558  | 0.59              |
| Fine-tuned | 0.4750   | 0.3110  | 0.83              |

