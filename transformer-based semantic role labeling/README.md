# Semantic Role Labeling with BERT

This project implements a **BERT-based Semantic Role Labeling (SRL)** system.  
The model identifies semantic roles (arguments) associated with predicates in a sentence, enabling structured interpretation of “who did what to whom, when, and how”.

The project focuses on **token-level classification using a pretrained transformer**, with a complete pipeline covering tokenization, training, inference, and result export.

---

## Task Overview

- **Task**: Semantic Role Labeling (token-level classification)
- **Input**: Sentences with predicate information
- **Output**: Semantic role labels assigned to individual tokens
- **Model**: BERT fine-tuned for SRL

Each token in a sentence is classified into a semantic role (e.g. argument labels) or marked as non-argument.



## Motivation

Semantic Role Labeling is a core task in semantic parsing and information extraction.  
Unlike surface-level tasks such as POS tagging or NER, SRL requires understanding **predicate–argument structure** and long-range dependencies.

This project explores how a pretrained BERT model can be adapted to perform SRL effectively without explicit syntactic parsing.



## Model Architecture

- **Base encoder**: BERT
- **Task head**: Token-level classification layer
- **Training objective**: Cross-entropy loss over role labels
- **Framework**: Hugging Face Transformers with PyTorch

The model is fine-tuned end-to-end on labeled SRL data.



## Tokenization and Label Alignment

- Text is tokenized using a BERT tokenizer
- Word-level SRL labels are aligned with subword tokens
- Special tokens ([CLS], [SEP]) are handled explicitly
- Attention masks and padding are applied for batching

The trained tokenizer is saved for consistent inference.



## Training and Inference

The notebook demonstrates:
- Loading and preprocessing SRL data
- Fine-tuning BERT for semantic role classification
- Running inference on unseen data
- Saving token-level predictions to disk

A trained model checkpoint and tokenizer are included for reuse.



## Outputs

- **Notebook**:  
  'srl_BERT.ipynb'  
  Contains:
  - Data preparation
  - Model training
  - Evaluation
  - Prediction examples

- **Prediction file**:  
  'bert_prediction.tsv'  
  Token-level semantic role predictions.

- **Saved artifacts**:
  - 'bert_model_trained/' – fine-tuned BERT model
  - 'Tokenizer_saved/' – tokenizer used during training
