import click
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import xgboost as xgb
import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)


@click.command()
@click.option("--training_data")
@click.option("--test_data")
@click.option("--label_col")
@click.option("--max_depth", default=7)
@click.option("--ntrees", default=200)
@click.option("--learning_rate", default=0.005)
def main(training_data, test_data, label_col, max_depth, ntrees, learning_rate):
    trainDF = pd.read_parquet(training_data)
    testDF = pd.read_parquet(test_data)
    yTrain = trainDF[[label_col]]
    XTrain = trainDF.drop([label_col], axis=1)
    yTest = testDF[[label_col]]
    XTest = testDF.drop([label_col], axis=1)

    xgbRegressor = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=ntrees,
        learning_rate=learning_rate,
        random_state=42,
        seed=42,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=1,
        gamma=1)
    pipeline = Pipeline(steps=[("regressor", xgbRegressor)])

    pipeline.fit(XTrain, yTrain)
    yPred = pipeline.predict(XTest)
    
    (rmse, mae, r2) = eval_metrics(yTest, yPred)
    
    print("XGBoost tree model (max_depth=%f, trees=%f, lr=%f):" % (max_depth, ntrees, learning_rate))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("ntrees", ntrees)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    mlflow.sklearn.log_model(pipeline, "model")
    #print("Model saved in run %s" % mlflow.active_run_id())


if __name__ == "__main__":
    main()
