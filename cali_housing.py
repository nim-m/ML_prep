from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

ch_data = fetch_california_housing()

df = pd.DataFrame(data=ch_data.data, columns=ch_data.feature_names)

# study the dataset
print(df.head())
print(df.describe())

# check available features
print(df.columns)

# define target feature
y = ch_data.target

# select feature to be used
X = df[['MedInc']]

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a model, fit, and predict
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# model evaluation
# accuracy = accuracy_score(y_test, y_pred)
# print("\nAccuracy:", accuracy)
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error:", mae, "\n")

r2_value = r2_score(y_test, y_pred)
print("R-squared:", r2_value)

# print actual comparison
print("y_test:\n", y_test)
print("y_pred:\n", y_pred)

# Plot
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.xlabel('MedInc')
plt.ylabel('Target (y_test)')
plt.show()
