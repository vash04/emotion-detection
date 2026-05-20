import pandas as pd
import yaml
import os
import logging

from sklearn.feature_extraction.text import CountVectorizer


# =========================
# Logging Configuration
# =========================

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    "feature_engineering.log"
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
            "Loading parameters from params.yaml"
        )

        with open(params_path, 'r') as file:

            max_features = yaml.safe_load(
                file
            )['feature_engineering'][
                'max_features'
            ]

        logging.info(
            f"Max features loaded: {max_features}"
        )

        return max_features

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
# Load Dataset
# =========================

def load_data(train_path, test_path):

    try:

        logging.info(
            "Loading processed train and test data"
        )

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logging.info(
            "Data loaded successfully"
        )

        return train_data, test_data

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
# Preprocess Dataset
# =========================

def preprocess_data(train_data, test_data):

    try:

        logging.info(
            "Starting data preprocessing"
        )

        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        X_train = train_data[
            'content'
        ].values

        y_train = train_data[
            'sentiment'
        ].values

        X_test = test_data[
            'content'
        ].values

        y_test = test_data[
            'sentiment'
        ].values

        logging.info(
            "Data preprocessing completed"
        )

        return (
            X_train,
            y_train,
            X_test,
            y_test
        )

    except KeyError as e:

        logging.error(
            f"Column missing in dataset: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error preprocessing data: {e}"
        )

        raise


# =========================
# Apply Bag Of Words
# =========================

def apply_bow(
        X_train,
        X_test,
        max_features
):

    try:

        logging.info(
            "Applying CountVectorizer"
        )

        vectorizer = CountVectorizer(
            max_features=max_features
        )

        X_train_bow = vectorizer.fit_transform(
            X_train
        )

        X_test_bow = vectorizer.transform(
            X_test
        )

        logging.info(
            "Bag of Words transformation completed"
        )

        return X_train_bow, X_test_bow

    except ValueError as e:

        logging.error(
            f"Value error in CountVectorizer: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error applying Bag of Words: {e}"
        )

        raise


# =========================
# Create DataFrame
# =========================

def create_dataframe(
        X_train_bow,
        y_train,
        X_test_bow,
        y_test
):

    try:

        logging.info(
            "Creating train and test dataframes"
        )

        train_df = pd.DataFrame(
            X_train_bow.toarray()
        )

        train_df['label'] = y_train

        test_df = pd.DataFrame(
            X_test_bow.toarray()
        )

        test_df['label'] = y_test

        logging.info(
            "Dataframes created successfully"
        )

        return train_df, test_df

    except Exception as e:

        logging.error(
            f"Error creating dataframe: {e}"
        )

        raise


# =========================
# Save Dataset
# =========================

def save_data(
        train_df,
        test_df,
        data_path
):

    try:

        logging.info(
            "Saving feature engineered datasets"
        )

        os.makedirs(
            data_path,
            exist_ok=True
        )

        train_df.to_csv(
            os.path.join(
                data_path,
                "train_bow.csv"
            ),
            index=False
        )

        test_df.to_csv(
            os.path.join(
                data_path,
                "test_bow.csv"
            ),
            index=False
        )

        logging.info(
            "Datasets saved successfully"
        )

    except Exception as e:

        logging.error(
            f"Error saving datasets: {e}"
        )

        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logging.info(
            "Feature engineering pipeline started"
        )

        max_features = load_params(
            'params.yaml'
        )

        train_data, test_data = load_data(
            './data/processed/processed_train.csv',
            './data/processed/processed_test.csv'
        )

        (
            X_train,
            y_train,
            X_test,
            y_test
        ) = preprocess_data(
            train_data,
            test_data
        )

        X_train_bow, X_test_bow = apply_bow(
            X_train,
            X_test,
            max_features
        )

        train_df, test_df = create_dataframe(
            X_train_bow,
            y_train,
            X_test_bow,
            y_test
        )

        save_data(
            train_df,
            test_df,
            os.path.join(
                "data",
                "features"
            )
        )

        logging.info(
            "Feature engineering pipeline completed successfully"
        )

    except Exception as e:

        logging.error(
            f"Pipeline failed: {e}"
        )

        raise


if __name__ == "__main__":
    main()