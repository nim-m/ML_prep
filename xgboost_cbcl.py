from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv('cbcl_1_5-2023-07-21.csv')

    # data cleaning -------------
    df = df[(df['cbcl_validity_flag'] != 1)]

    # drop rows where dsm5_autism_spectrum_problems_t_score = NaN
    df = df.dropna(subset=['dsm5_autism_spectrum_problems_t_score'])

    # (REMOVED) Drop rows where any cell in the subset contains NaN
    # Instead let xgb deal with missing values (is this okay?)

    # define X and y -------------
    X = df.iloc[:, 11:110].copy()
    y = df['dsm5_autism_spectrum_problems_t_score']

    return X, y


def create_model(eval_metrics):
    return XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric=eval_metrics, early_stopping_rounds=5)


def main():
    X, y = load_data()

    # Perform cross validation (manually, using a for loop, for finer control)
    # place the entire block of dividing datasets, model fitting, and testing within loop

    k = 10

    eval_metrics = ["rmse", "mae"]
    results = []

    # Loop over each fold
    for i in range(k):
        # Split data into training and test sets for current fold
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)

        eval_set = [(X_train, y_train), (X_val, y_val)]

        # Create and fit model for current fold
        model = create_model(eval_metrics)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        # Evaluate model on test set for current fold
        score = model.score(X_test, y_test)
        results.append(score)
    
    # Calculate average score and standard deviation
    avg_score = np.mean(results)
    std_dev = np.std(results)

    print("\n R2 scores: \n")

    print(f"Average score: {avg_score}")
    print(f"Standard deviation: {std_dev}")


if __name__ == "__main__":
    main()