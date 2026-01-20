# Named Entity Recognition with Feature-Based Models

This project implements a **feature-based Named Entity Recognition (NER)** system using classic machine learning models.  
The focus is on **feature engineering, ablation studies, and error analysis**, rather than end-to-end neural architectures.

The project uses the **CoNLL-2003 dataset** and evaluates how different linguistic and distributional features affect NER performance.

---

## Task Overview

- **Task**: Token-level Named Entity Recognition
- **Dataset**: CoNLL-2003
- **Labels**:  
  'PER', 'LOC', 'ORG', 'MISC' (BIO tagging scheme)
- **Models**:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)

Each token is classified independently using a rich set of handcrafted features.

---

## Motivation

While transformer-based NER models dominate current benchmarks, classic feature-based approaches remain valuable for:
- understanding **what signals matter** in NER
- working in **low-resource or constrained environments**
- building **interpretable models**

This project explores how much performance can be gained through **careful feature engineering**, without relying on end-to-end neural sequence models.

---

## Feature Engineering

The following features are extracted per token:

- **Lexical**
  - Token identity
  - Capitalization ('is_capitalized')
  - Token frequency in corpus

- **Syntactic**
  - POS tag (via NLTK)

- **Contextual**
  - Previous-token bigram
  - Next-token bigram

- **Distributional**
  - Pretrained word embeddings (Google News Word2Vec)

All features are represented using a 'DictVectorizer' to produce sparse feature vectors.

---

## Feature Ablation Experiments

A systematic **feature ablation study** is conducted to assess the contribution of each feature group.

### Procedure
1. Start from a minimal baseline ('token' only)
2. Incrementally add one feature type
3. Measure **macro F1-score** on the development set
4. Repeat across different models

Feature combinations explored include:
- 'token'
- 'token + capitalization'
- 'token + POS'
- 'token + bigrams'
- 'token + embeddings'
- Hybrid combinations (e.g. embeddings + contextual features)

---

## Models Evaluated

### Logistic Regression
- Strong baseline
- Performs best with combined lexical, contextual, and embedding features
- Used for detailed error analysis

### Naive Bayes
- Fast but sensitive to feature independence assumptions
- Performs best with limited contextual features

### Support Vector Machine
- Evaluated with and without hyperparameter tuning
- Grid search over regularization parameter 'C'
- Competitive performance with compact feature sets

---

## Data Analysis

Before training, the dataset is analyzed to understand:
- Label distribution
- Class imbalance (dominance of 'O' label)
- Differences between training and development splits

Label frequency plots highlight the sparsity of non-'O' entity tags and motivate careful evaluation beyond accuracy.

---

## Evaluation

- **Metric**: Macro-averaged F1-score
- **Outputs**:
  - Classification reports
  - Confusion matrices
  - Per-model prediction files

Final evaluation compares the best-performing feature sets across models.

---

## Error Analysis

A targeted error analysis is performed on the Logistic Regression model:

- Correctly predicted 'O' tokens are excluded
- A random sample of remaining instances is analyzed
- Misclassifications are inspected alongside their active features

This reveals common error patterns such as:
- Confusion between 'ORG' and 'LOC'
- Boundary errors between 'B-*' and 'I-*' tags
- Dependence on capitalization and context for disambiguation
