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

In the notebook interface, run the cells step-by-step by clicking Run (or pressing Shift + Enter) on each cell in sequence. This will:
1. Load the necessary libraries and datasets.
2. Perform data preprocessing and context retrieval using BM25.
3. Pretrain, finetune, and evaluate the PubMedBERT model.

> **⏳ Note**: Fine-tuning the PubMedBERT model is computationally intensive and may take significant time depending on your hardware. For faster results, consider using a GPU.

## 📁 Dataset
We use the Hugging Face datasets library to load the PubMedQA dataset.
```python
load_dataset("pubmed_qa", "pqa_labeled")
```

## 🚀 Methodology

This project implements a biomedical question-answering system using the PubMedQA dataset and PubMedBERT. The key steps include:

1. **Dataset Preparation**: The PubMedQA dataset is loaded, shuffled, and split into training and testing sets.
2. **BM25 Retrieval**: A BM25 model retrieves relevant contexts for questions based on tokenized documents.
3. **Preprocessing**: Questions and retrieved contexts are tokenized with PubMedBERT for classification tasks.
4. **Model Training**:
   - **Pretraining**: The PubMedBERT model is evaluated on the test set to establish a baseline.
   - **Fine-tuning**: The model is fine-tuned on the training set to improve performance.
5. **Evaluation**:
   - Metrics such as accuracy and F1-score are computed.
   - Semantic similarity between predicted and reference answers is calculated using Sentence-BERT. 


## 🧠 Results Summary
| Model       | Accuracy | Macro F1 | Semantic Similarity |
|------------|---------|---------|--------------------|
| Pretrained | 0.1750   | 0.1558  | 0.59              |
| Fine-tuned | 0.4750   | 0.3110  | 0.83              |


## 📚 References

- **PubMedQA Dataset**: [PubMedQA on Hugging Face](https://huggingface.co/datasets/pubmed_qa)
- **PubMedBERT**: [Microsoft's BiomedNLP-PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- **Semantic Similarity with Sentence-BERT**: [SBERT Documentation](https://www.sbert.net/)
- **BM25 Retrieval**: [Introduction to BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

