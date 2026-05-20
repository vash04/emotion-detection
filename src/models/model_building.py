import numpy as np
import pandas as pd
import pickle
import yaml
import os
import logging

from sklearn.ensemble import GradientBoostingClassifier


# =========================
# Logging Configuration
# =========================

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    "model_building.log"
)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# Load Parameters
# =========================

def load_params(params_path):

    try:

        logging.info(
            "Loading model parameters"
        )

        with open(params_path, 'r') as file:

            params = yaml.safe_load(
                file
            )['model_building']

        logging.info(
            f"Parameters loaded successfully: {params}"
        )

        return params

    except FileNotFoundError as e:

        logging.error(
            f"Params file not found: {e}"
        )

        raise

    except KeyError as e:

        logging.error(
            f"Key missing in params.yaml: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error loading parameters: {e}"
        )

        raise


# =========================
# Load Training Data
# =========================

def load_data(data_path):

    try:

        logging.info(
            "Loading training dataset"
        )

        train_data = pd.read_csv(data_path)

        logging.info(
            "Training dataset loaded successfully"
        )

        return train_data

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

def split_data(train_data):

    try:

        logging.info(
            "Splitting dataset into features and labels"
        )

        X_train = train_data.iloc[
            :,
            0:-1
        ].values

        y_train = train_data.iloc[
            :,
            -1
        ].values

        logging.info(
            "Dataset split completed"
        )

        return X_train, y_train

    except Exception as e:

        logging.error(
            f"Error splitting dataset: {e}"
        )

        raise


# =========================
# Train Model
# =========================

def train_model(
        X_train,
        y_train,
        params
):

    try:

        logging.info(
            "Initializing GradientBoostingClassifier"
        )

        clf = GradientBoostingClassifier(
            n_estimators=params[
                'n_estimators'
            ],
            learning_rate=params[
                'learning_rate'
            ]
        )

        logging.info(
            "Model training started"
        )

        clf.fit(
            X_train,
            y_train
        )

        logging.info(
            "Model training completed successfully"
        )

        return clf

    except KeyError as e:

        logging.error(
            f"Missing parameter in params.yaml: {e}"
        )

        raise

    except ValueError as e:

        logging.error(
            f"Value error during training: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error training model: {e}"
        )

        raise


# =========================
# Save Model
# =========================

def save_model(
        model,
        model_path
):

    try:

        logging.info(
            "Saving trained model"
        )

        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        with open(model_path, 'wb') as file:

            pickle.dump(
                model,
                file
            )

        logging.info(
            f"Model saved successfully at {model_path}"
        )

    except Exception as e:

        logging.error(
            f"Error saving model: {e}"
        )

        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logging.info(
            "Model building pipeline started"
        )

        params = load_params(
            'params.yaml'
        )

        train_data = load_data(
            "./data/features/train_bow.csv"
        )

        X_train, y_train = split_data(
            train_data
        )

        clf = train_model(
            X_train,
            y_train,
            params
        )

        save_model(
            clf,
            "models/model.pkl"
        )

        logging.info(
            "Model building pipeline completed successfully"
        )

    except Exception as e:

        logging.error(
            f"Pipeline failed: {e}"
        )

        raise


if __name__ == "__main__":
    main()