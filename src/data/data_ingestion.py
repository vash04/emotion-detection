import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

#logging configuration
logger=logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size

    except FileNotFoundError:
        logger.error('File not Found')
        raise 

    except KeyError as e:
        logger.error('missing key error in yaml')
        raise

    except yaml.YAMLError as e:
        logger.error('error while reading yaml file')
        raise


def read_data(url):

    try:
        df = pd.read_csv(url)
        return df

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")

    except pd.errors.ParserError:
        raise ValueError("Error while parsing CSV file")

    except Exception as e:
        raise Exception(f"Error while reading data: {e}")


def process_data(df):

    try:
        if 'tweet_id' not in df.columns:
            raise KeyError("'tweet_id' column not found in dataframe")

        if 'sentiment' not in df.columns:
            raise KeyError("'sentiment' column not found in dataframe")

        df.drop(columns=['tweet_id'], inplace=True)

        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])].copy()

        final_df['sentiment'] = final_df['sentiment'].replace({
            'neutral': 1,
            'sadness': 0
        })

        return final_df

    except Exception as e:
        raise Exception(f"Error during data processing: {e}")


def save_data(data_path, train_data, test_data):

    try:
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(data_path, "train.csv"),
            index=False
        )

        test_data.to_csv(
            os.path.join(data_path, "test.csv"),
            index=False
        )

        print("Train and test data saved successfully")

    except Exception as e:
        raise Exception(f"Error while saving data: {e}")


def main():

    try:
        test_size = load_params('params.yaml')

        df = read_data(
            'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        )

        final_df = process_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        data_path = os.path.join("data", "raw")

        save_data(data_path, train_data, test_data)

        print("Data ingestion completed successfully")

    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":

    main()