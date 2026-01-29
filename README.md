# NLP Portfolio — Language & AI (MSc)

This repository contains a collection of **Natural Language Processing (NLP)** projects completed during my Master's degree in **Language and AI**.  
The projects span classic machine learning approaches and modern transformer-based models, with a focus on **information extraction, semantic analysis, and clinical NLP**.

Each folder represents a self-contained project with code, experiments, and documentation.



## Projects Overview

### 'clinical_classification_medroberta'
**Clinical text classification with MedRoBERTa**

A Master's thesis project conducted in collaboration with a hospital, focusing on **multi-label classification of Dutch clinical notes** using transformer-based models.  
The project explores **LLM-assisted weak supervision** to extend the label space (ICF categories) with minimal manual annotation, and evaluates sentence and note-level performance.

**Key topics:**  
Clinical NLP · Text classification · Transformers · Weak supervision · Evaluation



### 'ML_name_entity_recognition'
**Named Entity Recognition with feature-based ML models**

An NLP project implementing **token-level NER** on the CoNLL-2003 dataset using classic machine learning models.  
The project emphasizes **feature engineering**, **ablation studies**, and **error analysis**, comparing Logistic Regression, Naive Bayes, and SVM approaches.

**Key topics:**  
NER · Feature engineering · Classical ML · Model analysis



### 'transformer-based_negation_scope_classification'
**Negation scope detection with BERT**

A transformer-based sequence labeling project addressing the task of **negation scope detection**.  
The model identifies which tokens fall under the influence of negation cues (e.g. *not*, *never*), using a fine-tuned BERT encoder and token-level classification.

**Key topics:**  
Negation handling · Sequence labeling · BERT · Token-level NLP



### 'transformer-based_semantic_role_labeling'
**Semantic Role Labeling (SRL) with BERT**

An implementation of **semantic role labeling** using a pretrained BERT model fine-tuned for token-level classification.  
The project focuses on learning **predicate–argument structure** directly from text, without explicit syntactic parsing.

**Key topics:**  
Semantic parsing · SRL · Transformers · Argument identification
