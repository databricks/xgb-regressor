import click
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import xgboost as xgb
import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    return (mae, rmse, r2)


@click.command()
@click.option("--training_data")
@click.option("--test_data")
@click.option("--label_col")
@click.option("--ntrees", default=200)
@click.option("--lr", default=0.005)
def main(training_data, test_data, label_col, ntrees, lr):
    trainDF = pd.read_parquet(training_data)
    testDF = pd.read_parquet(test_data)
    yTrain = trainDF[[label_col]]
    XTrain = trainDF.drop([label_col], axis=1)
    yTest = testDF[[label_col]]
    XTest = testDF.drop([label_col], axis=1)
    
    print("Running XGBoost regressor")
    mlflow.log_parameter("ntrees", ntrees)
    mlflow.log_parameter("lr", lr)

    xgbRegressor = xgb.XGBRegressor(
        n_estimators=ntrees,
        learning_rate=lr,
        random_state=42,
        seed=42,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=1,
        gamma=1)
    pipeline = Pipeline(steps=[("regressor", xgbRegressor)])
    pipeline.fit(XTrain, yTrain)
    yPred = pipeline.predict(XTest)
    
    (mae, rmse, r2) = eval_metrics(yTest, yPred)
    
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    
    mlflow.sklearn.log_model(pipeline, "model")


if __name__ == "__main__":
    main()
