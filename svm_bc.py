from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def create_model():
    return SVC(random_state=0)


def k_fold_cross_validation(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
    return accuracy_scores


def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    cnf_matrix = confusion_matrix(y_true, y_pred)

    # SPECIFICITY (True Negative Rate)
    specificity = confusion_matrix(y, y_pred)[0, 0] / (confusion_matrix(y, y_pred)[0, 0] + confusion_matrix(y, y_pred)[0, 1])

    print("Accuracy: {:.6f}".format(accuracy))
    print("AUC: {:.6f}".format(auc))
    print("Sensitivity: {:.6f}".format(sensitivity))
    print("Specificity: {:.6f}".format(specificity))
    print("Precision: {:.6f}".format(precision))

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    X, y = load_data()
    model = create_model()

    accuracy_scores = k_fold_cross_validation(model, X, y)

    print("K-Fold Cross Validation:")
    print('Mean Accuracy (SD): %.6f (%.6f)' % (np.mean(accuracy_scores), np.std(accuracy_scores)))

    y_pred = cross_val_predict(model, X, y, cv=10)

    evaluate_classification(y, y_pred)


if __name__ == "__main__":
    main()
