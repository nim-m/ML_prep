from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def load_data():
    df = pd.read_csv('cbcl_1_5-2023-07-21.csv')

    # data cleaning -------------
    df = df[(df['cbcl_validity_flag'] != 1)]  # (2828, 175)

    # drop rows where dsm5_autism_spectrum_problems_t_score = NaN
    df = df.dropna(subset=['dsm5_autism_spectrum_problems_t_score'])  # (2675, 175)

    # define X and y -------------
    X = df.iloc[:, 11:110].copy()  # all categorical features
    y = df['dsm5_autism_spectrum_problems_t_score']


    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)

    # create a new dataframe to store the encoded columns
    encoded_X = pd.DataFrame()

    # iterate over the categorical columns in X
    for col in X.columns:
        # one-hot encode the current column
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_col = encoder.fit_transform(X[[col]])

        # add the encoded columns to a list
        encoded_cols = [encoder.categories_[0][i] for i in range(encoded_col.shape[1])]
        encoded_cols = [col + '_' + str(i) for i in range(encoded_col.shape[1])]
        encoded_cols = pd.DataFrame(encoded_col, columns=encoded_cols)

        # concatenate the encoded columns with the existing dataframe
        encoded_X = pd.concat([encoded_X, encoded_cols], axis=1)

    # store to-be-dropped columns just in case
    dropped = encoded_X.loc[:, encoded_X.columns.str.contains("_3")]

    # drop columns with "_3" in their names
    encoded_X = encoded_X.loc[:, ~encoded_X.columns.str.contains("_3")]

    return encoded_X, y


def create_model(eval_metrics):
    return XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric=eval_metrics, early_stopping_rounds=5)


def main():
    X, y = load_data()
    print(X.shape)

    # Perform cross validation (manually, using a for loop, for finer control)
    # place the entire block of dividing datasets, model fitting, and testing within loop

    k = 10

    eval_metrics = ["rmse", "mae"]
    results = []

    # Initialize StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Loop over each fold
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        # Split data into training and test sets for current fold
        X_train, X_test = X.iloc[train_index], X.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)

        eval_set = [(X_train, y_train), (X_val, y_val)]

        # Create and fit model for current fold
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric=eval_metrics, early_stopping_rounds=10)
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
