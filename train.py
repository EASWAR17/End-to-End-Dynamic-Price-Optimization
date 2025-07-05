from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import argparse
import mlflow


def main(args):

    # Start MLflow run
    with mlflow.start_run():

        # Read and preprocess data
        df = get_data(args.training_data)

        # Split into training and test sets
        X_train, X_test, y_train, y_test = split_data(df)

        # Train model
        model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

        # Explicitly log model
        mlflow.xgboost.log_model(model, artifact_path="modelv1", registered_model_name="optimized-pricing-model")


# function that reads the data
def get_data(path):
    #print("Reading data...")
    df = pd.read_csv(path)
    df.drop(['weather', 'date', 'time'], axis=1, inplace=True)
    df['customer_subscription'] = df['customer_subscription'].map({'Free': 0, 'Silver': 1, 'Gold': 2})
    df.drop(['product_name'], axis=1, inplace=True)  # assume product_id encodes product
    return df


def split_data(df):
    #print("Splitting data...")
    X = df.drop('optimized_price', axis=1)
    y = df['optimized_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    early_stopping_rounds=10,
    eval_metric='rmse'
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    return model 
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, dest='training_data')
    parser.add_argument("--reg_rate", type=float, default=0.01, dest='reg_rate')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)


