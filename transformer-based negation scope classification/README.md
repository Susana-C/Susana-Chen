# Negation Scope Detection with BERT

This project implements a **BERT-based model for negation scope detection** in text.  
The goal is to identify which tokens in a sentence fall within the scope of a negation cue (e.g. *not*, *no*, *never*).

Negation detection is a crucial component in many NLP applications such as information extraction, sentiment analysis, and clinical or biomedical text processing, where negation can completely reverse the meaning of statements.



## Task Overview

- **Task**: Negation scope detection (token-level classification)
- **Input**: Tokenized sentences containing negation cues
- **Output**: For each token, a label indicating whether it is inside or outside the negation scope
- **Model**: BERT fine-tuned for sequence labeling

The task is framed as a supervised learning problem using a transformer-based encoder.



## Motivation

Negation is a well-known challenge in NLP.  
While cue detection is relatively straightforward, determining the **scope of negation** (which parts of the sentence are affected) is much harder.

This project explores how a pretrained BERT model can be adapted to learn negation scope boundaries directly from annotated data, without hand crafted rules.



## Model Architecture

- **Base model**: BERT
- **Task head**: Token-level classification head
- **Training objective**: Cross-entropy loss over token labels
- **Framework**: Hugging Face Transformers (via PyTorch)

The model is fine-tuned end-to-end on labeled negation scope data.



## Tokenization and Preprocessing

- Sentences are tokenized using a BERT tokenizer
- Word-level labels are aligned with subword tokens
- Special tokens ([CLS], [SEP]) are handled explicitly
- Padding and attention masks are applied for batching

The preprocessing logic is separated into a dedicated tokenizer module for clarity and reuse.



## Training and Inference

The project includes:
- A training pipeline for fine-tuning BERT
- Inference code to generate predictions on unseen data
- Export of predictions to a '.tsv' file for inspection and evaluation

Training and evaluation are demonstrated in the provided Jupyter notebook.



## Outputs

- **Notebook**:  
  'BERT negation scope detection.ipynb'  
  Contains:
  - Data loading
  - Model training
  - Evaluation
  - Example predictions

- **Prediction file**:  
  'prediction.tsv'  
  Stores token-level predictions produced by the trained model.
