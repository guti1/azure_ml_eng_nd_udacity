import lightgbm as lgbm
import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
import joblib
from sklearn.metrics import accuracy_score


run = Run.get_context()


def clean_data(data):
    # Dict for cleaning data
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
              "nov": 11, "dec": 12}
    weekdays = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)

    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help="Learning rate param for lgbm")
    parser.add_argument('--max_depth', type=int, default=10, help="Limit the tree depth explicitly")
    parser.add_argument('--num_leaves', type=int, default=255, help="Control the complexity of the tree model")
    parser.add_argument('--min_data_in_leaf', type=int, default=3, help="Large value can avoid growing too deep a tree")
    parser.add_argument('--num_iterations', type=int, default=500, help="Number of boosting iterations")

    args = parser.parse_args()
    run.log("learning-rate:", np.float(args.learning_rate))
    run.log("max_depth:", np.int(args.max_depth))
    run.log("num_leaves", np.int(args.num_leaves))
    run.log("min_data_in_leaf", np.int(args.min_data_in_leaf))
    run.log("num_iterations", np.int(args.num_iterations))

    factory = TabularDatasetFactory()
    train_data_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    valid_data_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_validate.csv"

    train_ds = factory.from_delimited_files(train_data_path)
    valid_ds = factory.from_delimited_files(valid_data_path)

    X_train, y_train = clean_data(train_ds)
    X_valid, y_valid = clean_data(valid_ds)
    
    d_train = lgbm.Dataset(X_train, label=y_train)

    lgbm_params = {}

    lgbm_params['learning_rate'] = args.learning_rate
    lgbm_params['boosting_type'] = 'gbdt'
    lgbm_params['objective'] = 'binary'
    lgbm_params['metric'] = 'binary_logloss'
    lgbm_params['max_depth'] = args.max_depth
    lgbm_params['num_leaves'] = args.num_leaves
    lgbm_params['min_data_in_leaf'] = args.min_data_in_leaf
    lgbm_params['colsample_bytree'] = 1.0,

    model = lgbm.train(lgbm_params, d_train, args.num_iterations)

    accuracy = accuracy_score(model.predict(X_valid).round(0).astype(int), y_valid)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(model, 'outputs/bankmarketing-lgbm-model.joblib')

if __name__ == '__main__':
    main()