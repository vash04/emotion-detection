import numpy as np
import pandas as pd

import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')
nltk.download('stopwords')


def load_data(train_path, test_path):

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def lemmatization(text):

    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(word) for word in text]

    return " ".join(text)


def remove_stop_words(text):

    stop_words = set(stopwords.words("english"))

    text = [
        word for word in str(text).split()
        if word not in stop_words
    ]

    return " ".join(text)


def removing_numbers(text):

    text = ''.join(
        [char for char in text if not char.isdigit()]
    )

    return text


def lower_case(text):

    text = text.split()

    text = [word.lower() for word in text]

    return " ".join(text)


def removing_punctuations(text):

    text = re.sub(
        '[%s]' % re.escape(
            """!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""
        ),
        ' ',
        text
    )

    text = text.replace('؛', "")

    text = re.sub('\s+', ' ', text)

    text = " ".join(text.split())

    return text.strip()


def removing_urls(text):

    url_pattern = re.compile(
        r'https?://\S+|www\.\S+'
    )

    return url_pattern.sub(r'', text)


def normalize_text(df):

    df.content = df.content.apply(
        lambda content: lower_case(content)
    )

    df.content = df.content.apply(
        lambda content: remove_stop_words(content)
    )

    df.content = df.content.apply(
        lambda content: removing_numbers(content)
    )

    df.content = df.content.apply(
        lambda content: removing_punctuations(content)
    )

    df.content = df.content.apply(
        lambda content: removing_urls(content)
    )

    df.content = df.content.apply(
        lambda content: lemmatization(content)
    )

    return df


def save_data(processed_train_data,
              processed_test_data,
              data_path):

    os.makedirs(data_path, exist_ok=True)

    processed_train_data.to_csv(
        os.path.join(data_path, "processed_train.csv"),
        index=False
    )

    processed_test_data.to_csv(
        os.path.join(data_path, "processed_test.csv"),
        index=False
    )


def main():

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
        os.path.join("data", "processed")
    )


if __name__ == "__main__":
    main()