from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


def load_data():
    df = pd.read_csv('cbcl_1_5-2023-07-21.csv')

    # data cleaning -------------

    df['cbcl_validity_flag'] = df['cbcl_validity_flag'].fillna(0)
    df = df[(df['cbcl_validity_flag'] != 1)]

    # drop rows where dsm5_autism_spectrum_problems_t_score = NaN
    df = df.dropna(subset=['dsm5_autism_spectrum_problems_t_score'])

    # Drop rows where any cell in the subset contains NaN
    subset_columns = df.iloc[:, 11:110]
    df = df.dropna(subset=subset_columns.columns)

    # define X and y -------------
    X = df.iloc[:, 11:110].copy()
    y = df['dsm5_autism_spectrum_problems_t_score']

    return X, y


def create_model(eval_metrics):
    return XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric=eval_metrics)


def main():
    X, y = load_data()

    # Split the data into training and test sets. Further split the training set into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    eval_set = [(X_train, y_train), (X_val, y_val)]
    eval_metrics = ["rmse", "mae"]

    model = create_model(eval_metrics)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True, early_stopping_rounds=2)

    # Perform cross validation
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")

    print("\nCross Validation:")
    print(cv_results)


if __name__ == "__main__":
    main()