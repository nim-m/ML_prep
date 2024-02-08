from sklearn.model_selection import cross_val_score, cross_val_predict, RepeatedKFold
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_data():
    cbcl = pd.read_csv('cbcl_1_5-2023-07-21.csv')

    # drop rows where cbcl_validity_flag = 1
    cbcl['cbcl_validity_flag'] = cbcl['cbcl_validity_flag'].fillna(0)
    cbcl = cbcl[(cbcl['cbcl_validity_flag'] != 1)]

    # drop rows where dsm5_autism_spectrum_problems_t_score = NaN
    cbcl = cbcl.dropna(subset=['dsm5_autism_spectrum_problems_t_score'])

    # Drop rows where any cell in the subset contains NaN
    subset_columns = cbcl.iloc[:, 11:110]
    cbcl = cbcl.dropna(subset=subset_columns.columns)

    # Extract selected features and target variable
    X = cbcl.iloc[:, 11:110].copy()
    y = cbcl['dsm5_autism_spectrum_problems_t_score']

    return X, y


def create_model():
    return Lasso(alpha=0.01)


def k_fold_cross_validation(model, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

    mse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    mae = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
    r2 = cross_val_score(model, X, y, scoring='r2', cv=cv)

    print("K-Fold Cross Validation:\n")
    print("MSE:\n", mse)
    print("\nMAE:\n", mae)
    print("\nR2:\n", r2)

    print("\nMSE: Mean = {:.6f}, SD = {:.6f}".format(np.mean(mse), np.std(mse)))
    print("MAE: Mean = {:.6f}, SD = {:.6f}".format(np.mean(mse), np.std(mse)))
    print("R2: Mean = {:.6f}, SD = {:.6f}".format(np.mean(r2), np.std(r2)))


def main():
    X, y = load_data()
    model = create_model()

    k_fold_cross_validation(model, X, y)

    y_pred = cross_val_predict(model, X, y, cv=10)
    print("\ny_pred: \n", y_pred)


if __name__ == "__main__":
    main()