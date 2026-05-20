import numpy as np
import pandas as pd

import os
import re
import nltk
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================
# Logging Configuration
# =========================

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    "data_preprocessing.log"
)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# Download NLTK Resources
# =========================

try:

    nltk.download('wordnet')
    nltk.download('stopwords')

    logging.info(
        "NLTK resources downloaded successfully"
    )

except Exception as e:

    logging.error(
        f"Error downloading NLTK resources: {e}"
    )

    raise


# =========================
# Load Dataset
# =========================

def load_data(train_path, test_path):

    try:

        logging.info(
            "Loading training and testing data"
        )

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        logging.info(
            "Data loaded successfully"
        )

        return train_data, test_data

    except FileNotFoundError as e:

        logging.error(
            f"File not found: {e}"
        )

        raise

    except pd.errors.EmptyDataError as e:

        logging.error(
            f"CSV file is empty: {e}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error while loading data: {e}"
        )

        raise


# =========================
# Lemmatization
# =========================

def lemmatization(text):

    try:

        lemmatizer = WordNetLemmatizer()

        text = text.split()

        text = [
            lemmatizer.lemmatize(word)
            for word in text
        ]

        return " ".join(text)

    except Exception as e:

        logging.error(
            f"Error in lemmatization: {e}"
        )

        return text


# =========================
# Remove Stopwords
# =========================

def remove_stop_words(text):

    try:

        stop_words = set(
            stopwords.words("english")
        )

        text = [
            word for word in str(text).split()
            if word not in stop_words
        ]

        return " ".join(text)

    except Exception as e:

        logging.error(
            f"Error removing stop words: {e}"
        )

        return text


# =========================
# Remove Numbers
# =========================

def removing_numbers(text):

    try:

        text = ''.join(
            [
                char for char in text
                if not char.isdigit()
            ]
        )

        return text

    except Exception as e:

        logging.error(
            f"Error removing numbers: {e}"
        )

        return text


# =========================
# Convert To Lowercase
# =========================

def lower_case(text):

    try:

        text = text.split()

        text = [
            word.lower()
            for word in text
        ]

        return " ".join(text)

    except Exception as e:

        logging.error(
            f"Error converting text to lowercase: {e}"
        )

        return text


# =========================
# Remove Punctuations
# =========================

def removing_punctuations(text):

    try:

        text = re.sub(
            '[%s]' % re.escape(
                """!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""
            ),
            ' ',
            text
        )

        text = text.replace('؛', "")

        text = re.sub(
            '\s+',
            ' ',
            text
        )

        text = " ".join(text.split())

        return text.strip()

    except Exception as e:

        logging.error(
            f"Error removing punctuations: {e}"
        )

        return text


# =========================
# Remove URLs
# =========================

def removing_urls(text):

    try:

        url_pattern = re.compile(
            r'https?://\S+|www\.\S+'
        )

        return url_pattern.sub(
            r'',
            text
        )

    except Exception as e:

        logging.error(
            f"Error removing URLs: {e}"
        )

        return text


# =========================
# Normalize Text
# =========================

def normalize_text(df):

    try:

        logging.info(
            "Starting text normalization"
        )

        if 'content' not in df.columns:

            raise ValueError(
                "'content' column not found"
            )

        df.content = df.content.apply(
            lambda content: lower_case(
                str(content)
            )
        )

        df.content = df.content.apply(
            lambda content: remove_stop_words(
                content
            )
        )

        df.content = df.content.apply(
            lambda content: removing_numbers(
                content
            )
        )

        df.content = df.content.apply(
            lambda content: removing_punctuations(
                content
            )
        )

        df.content = df.content.apply(
            lambda content: removing_urls(
                content
            )
        )

        df.content = df.content.apply(
            lambda content: lemmatization(
                content
            )
        )

        logging.info(
            "Text normalization completed"
        )

        return df

    except Exception as e:

        logging.error(
            f"Error during text normalization: {e}"
        )

        raise


# =========================
# Save Processed Data
# =========================

def save_data(
        processed_train_data,
        processed_test_data,
        data_path
):

    try:

        logging.info(
            "Saving processed data"
        )

        os.makedirs(
            data_path,
            exist_ok=True
        )

        processed_train_data.to_csv(
            os.path.join(
                data_path,
                "processed_train.csv"
            ),
            index=False
        )

        processed_test_data.to_csv(
            os.path.join(
                data_path,
                "processed_test.csv"
            ),
            index=False
        )

        logging.info(
            "Processed data saved successfully"
        )

    except Exception as e:

        logging.error(
            f"Error saving data: {e}"
        )

        raise


# =========================
# Main Function
# =========================

def main():

    try:

        logging.info(
            "Data preprocessing pipeline started"
        )

        train_data, test_data = load_data(
            "./data/raw/train.csv",
            "./data/raw/test.csv"
        )

        processed_train_data = normalize_text(
            train_data
        )

        processed_test_data = normalize_text(
            test_data
        )

        save_data(
            processed_train_data,
            processed_test_data,
            os.path.join(
                "data",
                "processed"
            )
        )

        logging.info(
            "Data preprocessing pipeline completed successfully"
        )

    except Exception as e:

        logging.error(
            f"Pipeline failed: {e}"
        )

        raise


if __name__ == "__main__":
    main()