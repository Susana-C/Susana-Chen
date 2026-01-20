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



def extract_features_and_labels_inspect(trainingfile):
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token,
                                'Is_Capitalized': token[0].istitle()
                               }
                data.append(feature_dict)
                # NOTE: you can add inline comments when you feel the need, e.g. "gold is in the last column"
                targets.append(components[-1])
    return data, targets


def plot_labels_with_counts(labels, values):
    total = 0
    for v in values:
        total+=v
    print('Total of values', total)
    ax = sns.barplot(x=labels, y=values)
    # Add values above bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.2, str(int((v/total*100)))+'%', ha='center')
    plt.show()


def extract_features_and_labels_ablation(file, word_embedding_model, feature_set=None):
    """
    Extract specified features and labels.

    Arguments:
        file (str): Path to the training file.
        word_embedding_model (gensim.KeyedVectors): Word embedding model.
        feature_set (list): List of features to include (e.g., ['token', 'is_capitalized']).

    Returns:
        data (list): List of feature dictionaries.
        targets (list): List of corresponding labels.
    """
    data = []
    targets = []

    if feature_set is None:
        feature_set = ['token', 'is_capitalized', 'frequency', 'pos', 'bigram_prev', 'bigram_next', 'embeddings']

    with open(file, 'r', encoding='utf8') as infile:
        lines = [line.strip() for line in infile if line.strip()]
        tokens = [line.split()[0] for line in lines]
        token_frequencies = Counter(tokens)

        for idx, line in enumerate(lines):
            components = line.split()
            if len(components) > 1:
                token = components[0]
                label = components[-1]
                pos_tags = nltk.pos_tag([token])
                pos_tag = pos_tags[0][1]

                prev_token = tokens[idx - 1] if idx > 0 else '<START>'
                next_token = tokens[idx + 1] if idx < len(tokens) - 1 else '<END>'

                features = {}

                # Include features conditionally
                if 'token' in feature_set:
                    features['token'] = token
                if 'is_capitalized' in feature_set:
                    features['is_capitalized'] = token[0].istitle()
                if 'frequency' in feature_set:
                    features['frequency'] = token_frequencies[token]
                if 'pos' in feature_set:
                    features['pos'] = pos_tag
                if 'bigram_prev' in feature_set:
                    features['bigram_prev'] = f"{prev_token}_{token}"
                if 'bigram_next' in feature_set:
                    features['bigram_next'] = f"{token}_{next_token}"
                if 'embeddings' in feature_set:
                    if token in word_embedding_model:
                        embedding_features = word_embedding_model[token]
                    else:
                        embedding_features = [0.0] * word_embedding_model.vector_size
                    for i, value in enumerate(embedding_features):
                        features[f'embedding_{i}'] = value

                data.append(features)
                targets.append(label)


def evaluate_features(word_embedding_model, feature_combinations, model):
    results = {}
    for feature_set in feature_combinations:
        data_train, labels_train = extract_features_and_labels_ablation(trainingfile, word_embedding_model, feature_set)
        data_test, labels_test = extract_features_and_labels_ablation(inputfile, word_embedding_model, feature_set)

        vec = DictVectorizer()
        data_train_vectorized = vec.fit_transform(data_train)
        data_test_vectorized = vec.transform(data_test)

        if model == 'logreg':
            model = LogisticRegression(max_iter=10000)
            print("Testing Logistic Regression Model...")
        elif model == 'nb':
            model = MultinomialNB()
            print("Testing Naive Bayes Model...")
        elif model == 'svm':
            model = SVC()
            print("Testing SVM Model...")
        model.fit(data_train_vectorized, labels_train)

        print(f"Feature set: {feature_set}")

        predictions = model.predict(data_test_vectorized)
        report = classification_report(labels_test, predictions, output_dict=True)
        report_view = classification_report(labels_test, predictions)
        print(report_view)

        dict = {'Gold':    labels_test, 'Predicted': predictions    }
        df = pd.DataFrame(dict, columns=['Gold','Predicted'])

        confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
        print(confusion_matrix)

        macro_f1 = report['macro avg']['f1-score']

        results[tuple(feature_set)] = macro_f1
        print(f"Macro F1-score: {macro_f1:.4f}\n")

    return results


def evaluate_features_final_output(word_embedding_model, feature_combination, model):
    results = {}

    data_train, labels_train = extract_features_and_labels_ablation(trainingfile, word_embedding_model, feature_combination)
    data_test, labels_test = extract_features_and_labels_ablation(inputfile, word_embedding_model, feature_combination)

    vec = DictVectorizer()
    data_train_vectorized = vec.fit_transform(data_train)
    data_test_vectorized = vec.transform(data_test)

    if model == 'logreg':
        model = LogisticRegression(max_iter=10000)
        print("Testing Logistic Regression Model...")
    elif model == 'nb':
        model = MultinomialNB()
        print("Testing Naive Bayes Model...")
    elif model == 'svm':
        model = SVC()
        print("Testing SVM Model...")
    model.fit(data_train_vectorized, labels_train)

    print(f"Feature set: {feature_combination}")

    predictions = model.predict(data_test_vectorized)
    report = classification_report(labels_test, predictions, output_dict=True, zero_division=0)
    report_view = classification_report(labels_test, predictions, zero_division=0)
    print(report_view)

    all_classes = np.unique(labels_train + labels_test)
    conf_matrix = confusion_matrix(labels_test, predictions, labels=all_classes)
    confusion_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)
    print("Confusion Matrix:")
    print(confusion_df)

    macro_f1 = report['macro avg']['f1-score']

    results[tuple(feature_combination)] = macro_f1
    print(f"Macro F1-score: {macro_f1:.4f}\n")

    # Save predictions to file
    output_file = f"{model}_predictions.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Predictions for {model}\n")
        f.write("True_Label\tPredicted_Label\n")
        for true_label, predicted_label in zip(labels_test, predictions):
            f.write(f"{true_label}\t{predicted_label}\n")
    print(f"Predictions saved to {output_file}")

    return results


def evaluate_features_svm_tuned(word_embedding_model, feature_combination):
    results = {}

    data_train, labels_train = extract_features_and_labels_ablation(trainingfile, word_embedding_model, feature_combination)
    data_test, labels_test = extract_features_and_labels_ablation(inputfile, word_embedding_model, feature_combination)

    vec = DictVectorizer()
    data_train_vectorized = vec.fit_transform(data_train)
    data_test_vectorized = vec.transform(data_test)

    param_grid = {'C': [0.1, 1, 10, 100]}
    svc = SVC(random_state=42)
    print("Testing SVM Model...")
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring='f1_macro')
    model = grid_search
    model.fit(data_train_vectorized, labels_train)

    print(f"Feature set: {feature_combination}")

    predictions = model.predict(data_test_vectorized)
    report = classification_report(labels_test, predictions, output_dict=True, zero_division=0)
    report_view = classification_report(labels_test, predictions, zero_division=0)
    print(report_view)

    all_classes = np.unique(labels_train + labels_test)
    conf_matrix = confusion_matrix(labels_test, predictions, labels=all_classes)
    confusion_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)
    print("Confusion Matrix:")
    print(confusion_df)

    macro_f1 = report['macro avg']['f1-score']

    results[tuple(feature_combination)] = macro_f1
    print(f"Macro F1-score: {macro_f1:.4f}\n")

    return results



def main(argv=None):
    """
    Main function for training and evaluating models on NER tasks.

    Command-line arguments:
        argv[1]: Path to the training file.
        argv[2]: Path to the input (test) file.
        argv[3]: Directory for output files (predictions and reports).
    """
    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]
    inputfile = argv[2]
    output_dir = argv[3]

    os.makedirs(output_dir, exist_ok=True)

    feature_set = ['token', 'embeddings', 'is_capitalized', 'bigram_prev', 'bigram_next', 'pos']
    models = {
        "logreg": LogisticRegression(max_iter=10000),
        "nb": MultinomialNB(),
        "svm": SVC()
    }

    print("Extracting features and labels...")
    training_features, gold_labels = extract_features_and_labels_ablation(trainingfile, word_embedding_model, feature_set)
    test_features, test_labels = extract_features_and_labels_ablation(inputfile, word_embedding_model, feature_set)

    vec = DictVectorizer()
    training_vectorized = vec.fit_transform(training_features)
    test_vectorized = vec.transform(test_features)

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name.upper()} model...")
        model.fit(training_vectorized, gold_labels)
        predictions = model.predict(test_vectorized)

        report = classification_report(test_labels, predictions, output_dict=True)
        print(f"Classification Report for {model_name.upper()}:\n", classification_report(test_labels, predictions))

        # Save predictions to file
        predictions_file = os.path.join(output_dir, f"{model_name}_predictions.txt")
        with open(predictions_file, 'w') as f:
            f.write(f"# Predictions for {model_name.upper()}\n")
            f.write("True_Label\tPredicted_Label\n")
            for true_label, pred_label in zip(test_labels, predictions):
                f.write(f"{true_label}\t{pred_label}\n")
        print(f"Predictions saved to {predictions_file}")

        # Save classification report to file
        report_file = os.path.join(output_dir, f"{model_name}_report.txt")
        with open(report_file, 'w') as f:
            f.write(f"Classification Report for {model_name.upper()}:\n")
            f.write(classification_report(test_labels, predictions))
        print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()

# In[ ]:


# remember that the first element of the list is not used
# (since this is the `python command when the args are read from sys.argv)
# make sure to complete the rest of the list assigned to args correctly
args = ['python']
main(args)
