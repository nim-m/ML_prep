import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.preprocessing import MinMaxScaler

import shap
from imblearn.over_sampling import RandomOverSampler


def load_data():
    df2 = pd.read_csv('cbcl_cog_imp.csv')

    # TOTAL PREDICTION DATASET
    X2 = df2.drop(columns=['subject_sp_id','deriv_cog_impair','cbcl_validity_flag']) 
    y2 = df2['deriv_cog_impair']

    # MODEL TRAINING DATASET 
    # drop rows where target variable deriv_cog_impair is missing
    df = df2.dropna(subset=['deriv_cog_impair'])

    # Drop rows where 'cbcl_validity_flag' column is 1.0 ----------
    df = df[df['cbcl_validity_flag'] != 1.0]
    
    # Drop rows where 'deriv_cog_impair' column is NaN ------------
    df = df.dropna(subset=['deriv_cog_impair'])

    # Separate features and target variable
    X = df.drop(columns=['subject_sp_id','deriv_cog_impair','cbcl_validity_flag']) 
    y = df['deriv_cog_impair']

    # Apply random oversampling to balance the classes
    # oversampler = RandomOverSampler(random_state=42)
    # X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # return X_resampled, y_resampled, df, df2, y2, X2
    return X, y, df, df2, y2, X2


def create_model(eval_metrics):
    return XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric=eval_metrics, early_stopping_rounds=5)


def calculate_metrics(y_true, y_pred):
    
    auc_score = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return auc_score, cm, report

def main():
    X, y, df, df2, y2, X2 = load_data()

    # Perform cross validation using StratifiedKFold
    k = 10

    eval_metrics = ["auc"]
    confusion_matrices = []

    prec_list = []
    acc_list = []
    sens_list = []
    spec_list = []
    f1_list = []
    auc_list = []

    # Loop over each fold
    for i in range(k):
        # Split data into training and test sets for current fold
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)

        # Initialize MinMaxScaler ---------------
        scaler = MinMaxScaler()

        # Scale the relevant features
        columns_to_scale = [
            "emotionally_reactive_raw_score",
            "emotionally_reactive_t_score",
            "emotionally_reactive_percentile",
            "anxious_depressed_raw_score",
            "anxious_depressed_t_score",
            "anxious_depressed_percentile",
            "somatic_complaints_raw_score",
            "somatic_complaints_t_score",
            "somatic_complaints_percentile",
            "withdrawn_raw_score",
            "withdrawn_t_score",
            "withdrawn_percentile",
            "sleep_problems_raw_score",
            "sleep_problems_t_score",
            "sleep_problems_percentile",
            "attention_problems_raw_score",
            "attention_problems_t_score",
            "attention_problems_percentile",
            "aggressive_behavior_raw_score",
            "aggressive_behavior_t_score",
            "aggressive_behavior_percentile",
            "internalizing_problems_raw_score",
            "internalizing_problems_t_score",
            "internalizing_problems_percentile",
            "externalizing_problems_raw_score",
            "externalizing_problems_t_score",
            "externalizing_problems_percentile",
            "total_problems_raw_score",
            "total_problems_t_score",
            "total_problems_percentile",
            "stress_problems_raw_score",
            "stress_problems_t_score",
            "stress_problems_percentile",
            "other_problems_raw_score",
            "dsm5_depressive_problems_raw_score",
            "dsm5_depressive_problems_t_score",
            "dsm5_depressive_problems_percentile",
            "dsm5_anxiety_problems_raw_score",
            "dsm5_anxiety_problems_t_score",
            "dsm5_anxiety_problems_percentile",
            "dsm5_autism_spectrum_problems_raw_score",
            "dsm5_autism_spectrum_problems_t_score",
            "dsm5_autism_spectrum_problems_percentile",
            "dsm5_attention_deficit_hyperactivity_raw_score",
            "dsm5_attention_deficit_hyperactivity_t_score",
            "dsm5_attention_deficit_hyperactivity_percentile",
            "dsm5_oppositional_defiant_raw_score",
            "dsm5_oppositional_defiant_t_score",
            "dsm5_oppositional_defiant_percentile"
        ]
        
        X_train_scaled = X_train.copy()
        X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_scaled[columns_to_scale])
        X_val_scaled = X_val.copy()
        X_val_scaled[columns_to_scale] = scaler.transform(X_val_scaled[columns_to_scale])
        X_test_scaled = X_test.copy()
        X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])

        #oversampler = RandomOverSampler(random_state=42)
        #X_train_scaled, y_train = oversampler.fit_resample(X_train_scaled, y_train)
        
        # ---------------
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

        # scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # Create and fit model for current fold
        model = create_model(eval_metrics)
        model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=True)

        # Predict probabilities on test set for current fold --------------
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Predict on test set for current fold
        y_pred = model.predict(X_test_scaled)
    
        # Calculate metrics for current fold
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    
        # Append metrics to lists
        auc_list.append(auc_score)
        prec_list.append(precision)
        acc_list.append(accuracy)
        sens_list.append(sensitivity)
        spec_list.append(specificity)
        f1_list.append(f1)
        confusion_matrices.append(cm)

        # Plot ROC curve for current fold
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'Fold {i+1}')

    # Calculate average AUC score and standard deviation --------------    
    avg_auc_score = np.mean(auc_list)
    std_dev = np.std(auc_list)

    print("\n AUC scores: \n")

    print("Average AUC score:", avg_auc_score)
    print(f"Standard deviation: {std_dev}")

    # Print average confusion matrix ------------------------    
    avg_cm = np.mean(confusion_matrices, axis=0)
    print("\n Average Confusion Matrix: \n")
    print(avg_cm)

    # Print average classification report -----------------------------    
    # Calculate average and standard deviation of metrics
    avg_prec = np.mean(prec_list)
    avg_acc = np.mean(acc_list)
    avg_sens = np.mean(sens_list)
    avg_spec = np.mean(spec_list)
    avg_f1 = np.mean(f1_list)
    
    std_prec = np.std(prec_list)
    std_acc = np.std(acc_list)
    std_sens = np.std(sens_list)
    std_spec = np.std(spec_list)
    std_f1 = np.std(f1_list)
    
    # Print the average and standard deviation
    print("Average Precision:", avg_prec)
    print("Average Accuracy:", avg_acc)
    print("Average Sensitivity:", avg_sens)
    print("Average Specificity:", avg_spec)
    print("Average F1-Score:", avg_f1)
    
    print("SD Precision:", std_prec)
    print("SD Accuracy:", std_acc)
    print("SD Sensitivity:", std_sens)
    print("SD Specificity:", std_spec)
    print("SD F1-Score:", std_f1)

    # Plot ROC curve for all folds ------------------    
    plt.title('ROC Curve for all Folds')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    # -------------- Explain model predictions using SHAP ---------------
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Plot SHAP values
    shap.summary_plot(shap_values, X_test_scaled, max_display=50)

    # Retrieve the feature names and SHAP values
    feature_names = X_test_scaled.columns
    shap_values_abs = np.abs(shap_values).mean(axis=0)  # Take the mean absolute SHAP values across samples
    
    # Combine feature names and SHAP values into a dictionary
    feature_shap_dict = dict(zip(feature_names, shap_values_abs))
    
    # Sort the dictionary by SHAP values in descending order
    sorted_feature_shap = dict(sorted(feature_shap_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Retrieve the sorted feature names
    sorted_feature_names = list(sorted_feature_shap.keys())
    print(sorted_feature_names)
    

    # ---------------- Store predictions in a new column in the original dataframe ----------
    df2.loc[X2.index, 'ml_pred_cog_impair'] = model.predict(X2)

    # Save dataframe with new column to csv file
    df2.to_csv('cbcl_cog_imp_ML_pred_minmax.csv', index=False)


if __name__ == "__main__":
    main()
