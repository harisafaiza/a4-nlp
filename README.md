# README: BERT Training, Sentence Embeddings, and Web Application

## Overview
This repository contains the implementation of three key tasks related to Natural Language Understanding (NLU) using BERT and Sentence-BERT models. These tasks include:
1. **Training BERT from Scratch**
2. **Sentence Embedding with Sentence-BERT**
3. **Evaluation, Analysis, and Web Application for Text Similarity**

Each of these tasks builds upon pre-trained transformers and adapts them for specific NLP objectives, such as text classification and semantic similarity.

---

## Task 1: Training BERT from Scratch
### Description
In this task, we implement and train a **Bidirectional Encoder Representations from Transformers (BERT)** model from scratch using the **BookCorpus** dataset. The model learns to generate contextual embeddings through a masked language modeling objective.

### Steps:
1. Load and preprocess a subset of the BookCorpus dataset.
2. Tokenize text using the **BERT tokenizer**.
3. Train a custom **BERT model** for masked language modeling (MLM).
4. Save the trained model for further use in downstream tasks.

### Files:
- `train_bert_from_scratch.py` - Contains code for training BERT.
- `trained_bert/` - Directory where the trained model is stored.

---

## Task 2: Sentence Embedding with Sentence-BERT
### Description
This task fine-tunes a **Sentence-BERT (SBERT)** model using a **Siamese network structure** to generate meaningful sentence embeddings. The embeddings are compared using **cosine similarity** for sentence-pair classification tasks.

### Steps:
1. Load the **SNLI** dataset for Natural Language Inference (NLI).
2. Tokenize premise-hypothesis pairs.
3. Fine-tune a **BERT-based Siamese network** with a **classification objective function**.
4. Train the model using **Softmax Loss**.
5. Save the trained model for further evaluation.

### Files:
- `sentence_bert_training.py` - Contains code for fine-tuning Sentence-BERT.
- `trained_sentence_bert/` - Directory where the trained model is stored.

---

## Task 3: Evaluation, Analysis, and Web Application for Text Similarity
### Description
In this task, we evaluate the trained **Sentence-BERT** model using NLI datasets and deploy a simple web application for demonstrating its capabilities.

### Steps:
1. Evaluate the **Sentence-BERT model** on the **SNLI** and **MNLI** datasets.
2. Compute and analyze model performance using accuracy and F1-score.
3. Develop a simple **web application** that allows users to compare two text inputs for **semantic similarity**.
4. Use a trained Sentence-BERT model to classify relationships (Entailment, Neutral, Contradiction).

### Files:
- `evaluate_model.py` - Contains evaluation metrics and analysis.
- `app/` - Directory containing web application files.

---

## Setup Instructions
### Requirements:
- Python 3.8+
- PyTorch
- Hugging Face `transformers` and `datasets`
- Flask (for web app deployment)

### Installation:
```bash
pip install torch transformers datasets flask
```

### Running the Training Scripts:
- To train BERT from scratch:
  ```bash
  python train_bert_from_scratch.py
  ```
- To fine-tune Sentence-BERT:
  ```bash
  python sentence_bert_training.py
  ```
- To evaluate the trained model:
  ```bash
  python evaluate_model.py
  ```
- To launch the web application:
  ```bash
  cd app
  python app.py
  ```

---

## Conclusion
This project implements BERT-based models for various NLP tasks, including training from scratch, sentence similarity learning, and deploying a web-based text similarity application. The trained models can be extended for various downstream applications like semantic search, paraphrase detection, and entailment recognition.

