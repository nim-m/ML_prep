from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def load():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


def main(X,y):
    # Data standardisation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

    # create a model, fit, and predict
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)

    # print actual comparison
    print("y_test:\n", y_test)
    print("y_pred:\n", y_pred)

if __name__ == "__main__":
    X, y = load()
    main(X, y)