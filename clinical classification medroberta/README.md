# Extending ICF Classification in Dutch Clinical Text using LLM-Assisted Weak Supervision

The project focuses on extending a clinical text classification system to recognize a broader range of **ICF (International Classification of Functioning, Disability and Health)** categories in Dutch rehabilitation notes, using **LLM-assisted weak supervision** and transformer-based models.



## Problem Overview

Clinical notes contain rich information about a patient's functioning (mobility, cognition, pain, social context), but this information is typically unstructured and difficult to analyze at scale.

Previous work successfully classified Dutch clinical sentences into **9 ICF categories**, but many important aspects of daily functioning (e.g. sleep, pain, stress handling, family relationships) were missing due to lack of annotated data.

**Goal of this thesis**:
> Extend an existing sentence-level ICF classifier from **10 labels (9 + None)** to **18 labels** using minimal additional manual annotation, while maintaining or improving performance.



## Key Idea

We use a **large language model (GPT-4o)** as an *automated annotator* to generate **weak labels** for new ICF categories, and then distill this knowledge into a **smaller, domain-specific model** that can run locally.

This combines the strengths of:
- LLMs: label expansion and semantic coverage
- Transformer encoders: efficient, privacy-preserving inference



## Task Definition

- **Task**: Multi-label sentence-level classification  
- **Language**: Dutch  
- **Domain**: Clinical rehabilitation notes  
- **Labels**:  
  - 17 ICF categories (body functions, activities, participation)
  - 1 "None" label for irrelevant sentences  
- A sentence may belong to **multiple categories** or none.



## Data Construction Pipeline

1. **Original Dataset**
   - Expert-annotated clinical sentences
   - 9 ICF categories + "None"

2. **New Data Collection**
   - Unlabeled Dutch clinical notes
   - Keyword-based retrieval for underrepresented ICF concepts

3. **LLM-Assisted Annotation**
   - GPT-4o used in a few-shot setup
   - Prompt includes category definitions and examples
   - Generates synthetic labels for 8 new categories

4. **Dataset Merging & Cleaning**
   - Combine expert-labeled and GPT-labeled data
   - Deduplication at sentence level
   - Stratified train/dev split at note level

5. **Gold Standard Test Set**
   - Manually annotated by clinical experts
   - Covers all 18 labels
   - Completely disjoint from training data



## Model Architecture

- **Encoder**: MedRoBERTa.nl  
  - Domain-specific Dutch clinical transformer
- **Training Setup**:
  - Multi-label classification
  - Binary cross-entropy loss (per label)
  - No class weighting
  - Threshold calibration per label using validation data
- **Training Library**: SimpleTransformers (Hugging Face backend)



## Threshold Calibration

Instead of a global threshold (e.g. 0.5), we apply **label-specific thresholds** derived from validation data:

- For each label:
  - Compare confidence distributions of true positives and true negatives
  - Select a low but discriminative threshold (≈ 0.08–0.09)
- This significantly improves recall for rare categories while controlling false positives.



## Evaluation

Evaluation is performed at two levels:

### Sentence-Level
- Macro-averaged precision, recall, and F1
- Per-category analysis
- Confusion patterns and error examples

### Note-Level
- Sentence predictions aggregated using logical OR
- Measures whether a note contains evidence for each ICF category

### Baseline Comparison
- GPT-4o inference (zero-shot and few-shot)
- Same test set and constraints
- Shows that the fine-tuned model meets or exceeds GPT-4o performance



## Results Summary

- Incorporating GPT-labeled data **improves performance on new categories**
- No degradation on original categories
- Fine-tuned MedRoBERTa.nl:
  - Outperforms zero-shot GPT-4o
  - Approaches or exceeds few-shot GPT-4o
- Confirms that **LLM-assisted weak supervision is effective** for label expansion in low-resource clinical NLP
