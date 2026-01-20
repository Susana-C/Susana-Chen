#!/usr/bin/env python
# coding: utf-8
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import gensim
import nltk
from sklearn.model_selection import train_test_split
import seaborn as sns
import random

import functions

# ## Load and inspect the data
#

# trainingfile =  "../../data/conll2003/conll2003.train.conll"
# inputfile = "../../data/conll2003/conll2003.dev.conll"
# outputfile = "../../data/conll2003/cosnll2003.pred.conll"



data, targets = functions.extract_features_and_labels_inspect(trainingfile)
data_test, targets_test = functions.extract_features_and_labels_inspect(inputfile)

unique_labels = set(targets)
print(f"Unique NER Labels: {unique_labels}")


# ## Data analysis: feature and label distributions


count_labels = Counter(targets)
print(count_labels)
labels = []
values = []
for label in count_labels:
    if label == 'O':
        continue
    else:
        labels.append(label)
        values.append(count_labels[label])

labels_plt = list(count_labels.keys())
values_plt = list(count_labels.values())
functions.plot_labels_with_counts(labels_plt, values_plt)



count_labels_test = Counter(targets_test)
print(count_labels_test)
labels_test = []
values_test = []
for label in count_labels_test:
    if label == 'O':
        continue
    else:
        labels_test.append(label)
        values_test.append(count_labels_test[label])


labels_plt_test = list(count_labels_test.keys())
values_plt_test = list(count_labels_test.values())
functions.plot_labels_with_counts(labels_plt_test, values_plt_test)


# # **Feature Ablation**


# # Word Embedding Model
# from gensim.models import KeyedVectors
# word_embedding_model = KeyedVectors.load_word2vec_format('/kaggle/input/final-project/GoogleNews-vectors-negative300.bin', binary=True)


# Word Embedding Model
from gensim.models import KeyedVectors
word_embedding_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


# # ***Feature Ablation: Logistic Regression Model***


# First Step: Test baseline Feature (token) paired with one of the other features
feature_combinations1 = [
    ['token'],
    ['token', 'is_capitalized'],
    ['token', 'frequency'],
    ['token', 'pos'],
    ['token', 'bigram_prev'],
    ['token', 'bigram_next'],
    ['token', 'embeddings']
]

results1 = functions.evaluate_features(word_embedding_model, feature_combinations1, 'logreg')


# Second Step: Choice among token itself and word embeddings
feature_combinations2 = [
    ['token'],
    ['embeddings'],
    ['token', 'embeddings']
]

results2= functions.evaluate_features(word_embedding_model, feature_combinations2, 'logreg')


# Third Step: Decided to preserve word embeddings with token itself, test three-feature combinations with other one-hot features

feature_combinations3 = [
    ['token', 'embeddings', 'is_capitalized'],
    ['token', 'embeddings', 'bigram_prev'],
    ['token', 'embeddings', 'bigram_next'],
    ['token', 'embeddings', 'pos'],
    ['token', 'embeddings', 'frequency']
]


results3= functions.evaluate_features(word_embedding_model, feature_combinations3, 'logreg')


# Fourth Step: Decided to preserve ['token', 'embedding', 'bigram_prev'], test combinations with the rest of the one-hot features
feature_combinations4 = [
    ['token', 'embeddings', 'bigram_prev', 'bigram_next'],
    ['token', 'embeddings', 'bigram_prev', 'is_capitalized'],
    ['token', 'embeddings', 'bigram_prev', 'pos']
]

# Run Feature Ablation
results4= functions.evaluate_features(word_embedding_model, feature_combinations4, 'logreg')


# Fifth Step: Preserve ['token', 'embeddings', 'bigram_prev', 'bigram_next'], test combinations with the rest of the one-hot features
feature_combinations5 = [
    ['token', 'embeddings', 'bigram_prev', 'bigram_next','is_capitalized'],
    ['token', 'embeddings', 'bigram_prev', 'bigram_next','pos'],
    ['token', 'embeddings', 'bigram_prev', 'bigram_next','is_capitalized','pos']
]

results5= functions.evaluate_features(word_embedding_model, feature_combinations5, 'logreg')


# # ***Feature Ablation: Naive Bayes Model & SVM Model***


# Same as the first step for Logistic Regression model, test on combinations of one-hot features only
feature_combinations_1 = [
    ['token'],
    ['token', 'is_capitalized'],
    ['token', 'frequency'],
    ['token', 'pos'],
    ['token', 'bigram_prev'],
    ['token', 'bigram_next']
]

# Run Feature Ablation
results_nb1 = functions.evaluate_features(word_embedding_model, feature_combinations_1, 'nb')
results_svm1 = functions.evaluate_features(word_embedding_model, feature_combinations_1, 'svm')


# Naive Bayes: Preserve 'token' and 'bigram_prev', test combinations with other features (exclude 'frequency')
feature_combinations_nb = [
    ['token', 'bigram_prev', 'is_capitalized'],
    ['token', 'bigram_prev', 'pos'],
    ['token', 'bigram_prev', 'is_capitalized', 'pos']
]

results_nb2 = functions.evaluate_features(word_embedding_model, feature_combinations_nb, 'nb')


# SVM: Preserve 'token' and 'is_capitalized', test combinations with other features (exclude 'frequency')
feature_combinations_svm = [
    ['token', 'is_capitalized', 'pos'],
    ['token', 'is_capitalized', 'bigram_prev'],
    ['token', 'is_capitalized', 'bigram_prev', 'pos']
]

results_svm2 = functions.evaluate_features(word_embedding_model, feature_combinations_svm, 'svm')


# # **Final Evaluation on Chosen Feature Combinations**

model_names = ['nb', 'svm']
for model_name in model_names:
    if model_name == 'logreg':
        feature_combination = ['token', 'embeddings', 'is_capitalized', 'bigram_prev', 'bigram_next', 'pos']
        results_log = functions.evaluate_features_final(word_embedding_model, feature_combination, 'logreg')
        print()
    elif model_name == 'nb':
        feature_combination = ['token', 'bigram_prev']
        results_nb = functions.evaluate_features_final(word_embedding_model, feature_combination, 'nb')
        print()
    elif model_name == 'svm':
        feature_combination = ['token', 'is_capitalized', 'bigram_prev']
        results_svm = functions.evaluate_features_final(word_embedding_model, feature_combination, 'svm')


# # Hyper-parameter Tuning for SVM Model


feature_combination_tuned = ['token', 'is_capitalized', 'bigram_prev']
results_svm_tuned = functions.evaluate_features_svm_tuned(word_embedding_model, feature_combination_tuned)


# # **Error Analysis**


feature_set_ea = ['token', 'embeddings', 'is_capitalized', 'bigram_prev', 'bigram_next', 'pos']

data_train, labels_train = functions.extract_features_and_labels_ablation(trainingfile, word_embedding_model, feature_set_ea)
data_test, labels_test = functions.extract_features_and_labels_ablation(inputfile, word_embedding_model, feature_set_ea)

vec = DictVectorizer()
data_train_vectorized = vec.fit_transform(data_train)
data_test_vectorized = vec.transform(data_test)

model = LogisticRegression(max_iter=10000)
model.fit(data_train_vectorized, labels_train)
predictions = model.predict(data_test_vectorized)

data = pd.DataFrame({
    'features': data_test,
    'true_label': labels_test,
    'predicted_label': predictions
})

# Exclude correctly classified 'O' class samples
data['is_correct'] = data['true_label'] == data['predicted_label']
data_filtered = data[~((data['is_correct']) & (data['true_label'] == 'O'))]

sample_size = 100
sample = data_filtered.sample(n=min(sample_size, len(data_filtered)), random_state=42)

# Separate correct and incorrect predictions within the sampled data
sample['is_correct'] = sample['true_label'] == sample['predicted_label']
correct_samples = sample[sample['is_correct']]
incorrect_samples = sample[~sample['is_correct']]

num_correct = len(correct_samples)
num_incorrect = len(incorrect_samples)
print(f"Correctly classified instances (excluding 'O'): {num_correct} ({num_correct / len(sample) * 100:.2f}%)")
print(f"Misclassified instances: {num_incorrect} ({num_incorrect / len(sample) * 100:.2f}%)")
print()


print("Examples of Misclassified Instances:")
for idx, row in incorrect_samples.iterrows():
    print(f"Index: {idx}")
    print(f"True Label: {row['true_label']}, Predicted: {row['predicted_label']}")
    ex_emb = dict(list(row['features'].items())[:5]) # Exclude word embeddings in this case to make the output cleaner
    print(f"One-hot features: {ex_emb}")
    print("-" * 50)

print()

print("Examples of Correctly Classified Instances (Excluding 'O'):")
for idx, row in correct_samples.head(20).iterrows():
    print(f"Index: {idx}")
    print(f"True Label: {row['true_label']}, Predicted: {row['predicted_label']}")
    ex_emb = dict(list(row['features'].items())[:5])
    print(f"One-hot features: {ex_emb}")
    print("-" * 50)
