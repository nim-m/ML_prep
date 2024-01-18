from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

bc_data = load_breast_cancer()

df = pd.DataFrame(data=bc_data.data, columns=bc_data.feature_names)

# study the dataset
print(df.head())
print(df.describe())

# check available features
print(df.columns)

# define target feature
y = bc_data.target

# select features to be used for training (optional)
# trainList = []
X = df

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create a model, fit, and predict
model = LogisticRegression(random_state=0)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error:", mae, "\n")

# print actual comparison
print("y_test:\n", y_test)
print("y_pred:\n", y_pred)
