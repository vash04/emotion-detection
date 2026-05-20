import numpy as np
import pandas as pd
import pickle
import yaml
import os

from sklearn.ensemble import GradientBoostingClassifier


def load_params(params_path):

    params = yaml.safe_load(
        open(params_path, 'r')
    )['model_building']

    return params


def load_data(data_path):

    train_data = pd.read_csv(data_path)

    return train_data


def split_data(train_data):

    X_train = train_data.iloc[:, 0:-1].values

    y_train = train_data.iloc[:, -1].values

    return X_train, y_train


def train_model(X_train, y_train, params):

    clf = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate']
    )

    clf.fit(X_train, y_train)

    return clf


def save_model(model, model_path):

    os.makedirs(os.path.dirname(model_path),
                exist_ok=True)

    pickle.dump(
        model,
        open(model_path, 'wb')
    )


def main():

    params = load_params('params.yaml')

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


if __name__ == "__main__":
    main()