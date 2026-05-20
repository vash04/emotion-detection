import numpy as np
import pandas as pd
import pickle
import json
import os
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# =========================
# Logging Configuration
# =========================

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    "model_evaluation.log"
)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# Load Trained Model
# =========================

def load_model(model_path):

    try:

        logging.info(
            "Loading trained model"
        )

        with open(model_path, 'rb') as f:

            clf = pickle.load(f)

        logging.info(
            "Model loaded successfully"
        )

        return clf

    except FileNotFoundError as e:

        logging.error(
            f"Model file not found: {e}"
        )

        raise

    except pickle.UnpicklingError as e:

        logging.error(
            f"Error unpickling model: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error loading model: {e}"
        )

        raise


# =========================
# Load Test Dataset
# =========================

def load_data(data_path):

    try:

        logging.info(
            "Loading test dataset"
        )

        test_data = pd.read_csv(data_path)

        logging.info(
            "Test dataset loaded successfully"
        )

        return test_data

    except FileNotFoundError as e:

        logging.error(
            f"Dataset file not found: {e}"
        )

        raise

    except pd.errors.EmptyDataError as e:

        logging.error(
            f"Dataset file is empty: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error loading dataset: {e}"
        )

        raise


# =========================
# Split Dataset
# =========================

def split_data(test_data):

    try:

        logging.info(
            "Splitting dataset into features and labels"
        )

        X_test = test_data.iloc[
            :,
            0:-1
        ].values

        y_test = test_data.iloc[
            :,
            -1
        ].values

        logging.info(
            "Dataset split completed"
        )

        return X_test, y_test

    except Exception as e:

        logging.error(
            f"Error splitting dataset: {e}"
        )

        raise


# =========================
# Make Predictions
# =========================

def prediction(model, X_test):

    try:

        logging.info(
            "Making predictions on test data"
        )

        y_pred = model.predict(
            X_test
        )

        y_pred_proba = model.predict_proba(
            X_test
        )[:, 1]

        logging.info(
            "Predictions completed successfully"
        )

        return y_pred, y_pred_proba

    except Exception as e:

        logging.error(
            f"Error during prediction: {e}"
        )

        raise


# =========================
# Evaluate Model
# =========================

def evaluate_model(
        y_test,
        y_pred,
        y_pred_proba
):

    try:

        logging.info(
            "Evaluating model performance"
        )

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

        logging.info(
            f"Evaluation metrics: {metrics_dict}"
        )

        return metrics_dict

    except Exception as e:

        logging.error(
            f"Error evaluating model: {e}"
        )

        raise


# =========================
# Save Metrics
# =========================

def save_metrics(
        metrics_dict,
        file_path
):

    try:

        logging.info(
            "Saving evaluation metrics"
        )

        with open(file_path, 'w') as f:

            json.dump(
                metrics_dict,
                f,
                indent=4
            )

        logging.info(
            f"Metrics saved successfully at {file_path}"
        )

    except Exception as e:

        logging.error(
            f"Error saving metrics: {e}"
        )

        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logging.info(
            "Model evaluation pipeline started"
        )

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

        logging.info(
            "Model evaluation pipeline completed successfully"
        )

    except Exception as e:

        logging.error(
            f"Pipeline failed: {e}"
        )

        raise


if __name__ == "__main__":
    main()