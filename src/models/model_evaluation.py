import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def load_model(model_path):

    with open(model_path, 'rb') as f:

        clf = pickle.load(f)

    return clf


def load_data(data_path):

    test_data = pd.read_csv(data_path)

    return test_data


def split_data(test_data):

    X_test = test_data.iloc[:, 0:-1].values

    y_test = test_data.iloc[:, -1].values

    return X_test, y_test


def prediction(model, X_test):

    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(
        X_test
    )[:, 1]

    return y_pred, y_pred_proba


def evaluate_model(y_test,
                   y_pred,
                   y_pred_proba):

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_pred_proba
    )

    metrics_dict = {

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

    return metrics_dict


def save_metrics(metrics_dict, file_path):

    with open(file_path, 'w') as f:

        json.dump(
            metrics_dict,
            f,
            indent=4
        )


def main():

    clf = load_model(
        'models/model.pkl'
    )

    test_data = load_data(
        './data/features/test_bow.csv'
    )

    X_test, y_test = split_data(
        test_data
    )

    y_pred, y_pred_proba = prediction(
        clf,
        X_test
    )

    metrics_dict = evaluate_model(
        y_test,
        y_pred,
        y_pred_proba
    )

    save_metrics(
        metrics_dict,
        'metrics.json'
    )


if __name__ == "__main__":
    main()