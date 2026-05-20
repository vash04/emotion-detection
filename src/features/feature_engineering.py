import pandas as pd
import yaml
import os

from sklearn.feature_extraction.text import CountVectorizer


def load_params(params_path):

    max_features = yaml.safe_load(
        open(params_path, 'r')
    )['feature_engineering']['max_features']

    return max_features


def load_data(train_path, test_path):

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def preprocess_data(train_data, test_data):

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)

    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    return X_train, y_train, X_test, y_test


def apply_bow(X_train, X_test, max_features):

    vectorizer = CountVectorizer(
        max_features=max_features
    )

    X_train_bow = vectorizer.fit_transform(X_train)

    X_test_bow = vectorizer.transform(X_test)

    return X_train_bow, X_test_bow


def create_dataframe(X_train_bow, y_train,
                     X_test_bow, y_test):

    train_df = pd.DataFrame(
        X_train_bow.toarray()
    )

    train_df['label'] = y_train

    test_df = pd.DataFrame(
        X_test_bow.toarray()
    )

    test_df['label'] = y_test

    return train_df, test_df


def save_data(train_df, test_df, data_path):

    os.makedirs(data_path, exist_ok=True)

    train_df.to_csv(
        os.path.join(data_path, "train_bow.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(data_path, "test_bow.csv"),
        index=False
    )


def main():

    max_features = load_params('params.yaml')

    train_data, test_data = load_data(
        './data/processed/processed_train.csv',
        './data/processed/processed_test.csv'
    )

    X_train, y_train, X_test, y_test = preprocess_data(
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
        os.path.join("data", "features")
    )


if __name__ == "__main__":
    main()