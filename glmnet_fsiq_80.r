library(caret)
library(glmnet)
library(e1071)  # For SVM
library(randomForest)
library(class)  # For kNN
library(xgboost) 
library(pROC)


### DATA LOADING AND SPLITTING
### ------------------------------------------------------

df <- read.csv("ref129_less18_fsiq_1554.csv")

dropped_paras = c('subject_sp_id','derived_cog_impair','asd', 'fsiq', 'fsiq_score', 'fsiq_70')
df <- df[, !(names(df) %in% dropped_paras)]

df$fsiq_80 <- as.logical(df$fsiq_80)
df$fsiq_80 <- as.numeric(df$fsiq_80)

parameters <- c(
  "q02_catch_ball_1.0", "q02_catch_ball_2.0", "q02_catch_ball_3.0", "q02_catch_ball_4.0", "q02_catch_ball_5.0",
  "q05_run_fast_similar_1.0", "q05_run_fast_similar_2.0", "q05_run_fast_similar_3.0", "q05_run_fast_similar_4.0", "q05_run_fast_similar_5.0",
  "q06_plan_motor_activity_1.0", "q06_plan_motor_activity_2.0", "q06_plan_motor_activity_3.0", "q06_plan_motor_activity_4.0", "q06_plan_motor_activity_5.0",
  "q09_appropriate_tension_printing_writing_1.0", "q09_appropriate_tension_printing_writing_2.0", "q09_appropriate_tension_printing_writing_3.0", "q09_appropriate_tension_printing_writing_4.0", "q09_appropriate_tension_printing_writing_5.0",
  "q10_cuts_pictures_shapes_1.0", "q10_cuts_pictures_shapes_2.0", "q10_cuts_pictures_shapes_3.0", "q10_cuts_pictures_shapes_4.0", "q10_cuts_pictures_shapes_5.0",
  "q11_likes_sports_motors_skills_1.0", "q11_likes_sports_motors_skills_2.0", "q11_likes_sports_motors_skills_3.0", "q11_likes_sports_motors_skills_4.0", "q11_likes_sports_motors_skills_5.0",
  "q13_quick_competent_tidying_up_1.0", "q13_quick_competent_tidying_up_2.0", "q13_quick_competent_tidying_up_3.0", "q13_quick_competent_tidying_up_4.0", "q13_quick_competent_tidying_up_5.0",
  "q01_whole_body_0.0", "q01_whole_body_1.0", "q01_whole_body_2.0", "q01_whole_body_3.0",
  "q03_hand_finger_0.0", "q03_hand_finger_1.0", "q03_hand_finger_2.0", "q03_hand_finger_3.0",
  "q07_hits_self_body_0.0", "q07_hits_self_body_1.0", "q07_hits_self_body_2.0", "q07_hits_self_body_3.0",
  "q08_hits_self_against_object_0.0", "q08_hits_self_against_object_1.0", "q08_hits_self_against_object_2.0", "q08_hits_self_against_object_3.0",
  "q09_hits_self_with_object_0.0", "q09_hits_self_with_object_1.0", "q09_hits_self_with_object_2.0", "q09_hits_self_with_object_3.0",
  "q12_rubs_0.0", "q12_rubs_1.0", "q12_rubs_2.0", "q12_rubs_3.0",
  "q16_complete_0.0", "q16_complete_1.0", "q16_complete_2.0", "q16_complete_3.0",
  "q18_checking_0.0", "q18_checking_1.0", "q18_checking_2.0", "q18_checking_3.0",
  "q19_counting_0.0", "q19_counting_1.0", "q19_counting_2.0", "q19_counting_3.0",
  "q22_touch_tap_0.0", "q22_touch_tap_1.0", "q22_touch_tap_2.0", "q22_touch_tap_3.0",
  "q32_insists_walking_0.0", "q32_insists_walking_1.0", "q32_insists_walking_2.0", "q32_insists_walking_3.0",
  "q27_play_0.0", "q27_play_1.0", "q27_play_2.0", "q27_play_3.0",
  "q28_communication_0.0", "q28_communication_1.0", "q28_communication_2.0", "q28_communication_3.0",
  "q29_things_same_place_0.0", "q29_things_same_place_1.0", "q29_things_same_place_2.0", "q29_things_same_place_3.0",
  "q31_becomes_upset_0.0", "q31_becomes_upset_1.0", "q31_becomes_upset_2.0", "q31_becomes_upset_3.0",
  "q34_dislikes_changes_0.0", "q34_dislikes_changes_1.0", "q34_dislikes_changes_2.0", "q34_dislikes_changes_3.0",
  "q35_insists_door_0.0", "q35_insists_door_1.0", "q35_insists_door_2.0", "q35_insists_door_3.0",
  "q36_likes_piece_music_0.0", "q36_likes_piece_music_1.0", "q36_likes_piece_music_2.0", "q36_likes_piece_music_3.0",
  "q39_insists_time_0.0", "q39_insists_time_1.0", "q39_insists_time_2.0", "q39_insists_time_3.0",
  "q41_strongly_attached_0.0", "q41_strongly_attached_1.0", "q41_strongly_attached_2.0", "q41_strongly_attached_3.0",
  "q43_fascination_movement_0.0", "q43_fascination_movement_1.0", "q43_fascination_movement_2.0", "q43_fascination_movement_3.0"
)


df[parameters] <- lapply(df[parameters], function(x) as.integer(as.logical(x)))
# df <- as.data.frame(lapply(df, as.numeric))


print("Data loading and type conversion done.")




# Set seed for reproducibility
set.seed(123)

# Create a vector of indices for stratified sampling
train_index <- createDataPartition(df$fsiq_80, p = 0.6, list = FALSE, times = 1)
remaining_data <- df[-train_index, ]

# Now create test and validation sets from the remaining data
test_index <- createDataPartition(remaining_data$fsiq_80, p = 0.5, list = FALSE, times = 1)
valid_index <- setdiff(seq(nrow(remaining_data)), test_index)

# Split the dataset
train_data <- df[train_index, ]
valid_data <- df[valid_index, ]
test_data <- df[test_index, ]

print("Straified splitting done.")



### MINMAX SCALING
### ------------------------------------------------------

# Define the parameters for Min-Max scaling
parameters_to_scale <- c("age_onset_mos", "fed_self_spoon_age_mos", "smiled_age_mos", "fine_motor_handwriting")  

# Perform Min-Max scaling on the training set
train_data_orig <- train_data  # Make a copy of the training data
train_data[, parameters_to_scale] <- apply(train_data[, parameters_to_scale], 2, function(x) (x - min(x)) / (max(x) - min(x)))

# Perform Min-Max scaling on the validation set
valid_data_orig <- valid_data  # Make a copy of the validation data
valid_data[, parameters_to_scale] <- apply(valid_data[, parameters_to_scale], 2, function(x) (x - min(x)) / (max(x) - min(x)))

# Perform Min-Max scaling on the testing set
test_data_orig <- test_data  # Make a copy of the testing data
test_data[, parameters_to_scale] <- apply(test_data[, parameters_to_scale], 2, function(x) (x - min(x)) / (max(x) - min(x)))

print("Minmax scaling done.")


# --------------------------------------------------
# Data Splitting

# For train set
X_train <- train_data[, !(names(train_data) %in% "fsiq_80")]
y_train <- train_data$fsiq_80
y_train_num <- train_data$fsiq_80
y_train <- factor(y_train)  # factorise

# For validation set
X_val <- valid_data[, !(names(valid_data) %in% "fsiq_80")]
y_val <- valid_data$fsiq_80
y_val_num <- valid_data$fsiq_80
y_val <- factor(y_val)  # factorise


# For test set
X_test <- test_data[, !(names(test_data) %in% "fsiq_80")]
y_test <- test_data$fsiq_80
y_test_num <- test_data$fsiq_80
y_test <- factor(y_test)  # factorise


print("Splitting done.")


# -----------------------------------------------
# Training Data - Train, and choose best model in each category based on lambda

# MODEL 01:
# ---------
glmnet_model = cv.glmnet(x = as.matrix(X_train), y = y_train, family = 'binomial', alpha = 0.1, type.measure = "auc")

# Get the best lambda value
best_lambda <- glmnet_model$lambda.min

# Create a new model using the best lambda
best_glmnet_model <- glmnet(x = as.matrix(X_train), y = y_train, family = 'binomial', alpha = 0.1, lambda = best_lambda)


print("Training and model selection (based on lambda) done.")


# -----------------------------------------------
# Validation data - Choose best model among all 5 using AUC score

# Make predictions on the validation set
glmnet_pred_val <- predict(best_glmnet_model, newx = as.matrix(X_val), type = 'response')

# Extract predicted probabilities for the positive class
glmnet_pred_val_vector <- as.vector(glmnet_pred_val)

# Calculate AUC-ROC
auc_glmnet_valid <- roc(y_val_num, glmnet_pred_val_vector)$auc

cat("AUC glmnet on Validation Data = ", auc_glmnet_valid, "\n")


print("Validation and model selection (based on AUC) done.")


# -------------------------------------------------------
# Test data - Cutoff selection and Model evaluation
# ROC curve, AUC, accuracy, Cohen’s kappa, sensitivity, specificity and 
# positive predictive values (PPV), and negative predictive values (NPV)

# Make predictions on test set using the final model
glmnet_pred_test <- predict(best_glmnet_model, newx = as.matrix(X_test), type = 'response')
glmnet_pred_test_vector <- as.vector(glmnet_pred_test)

auc_glmnet_test <- roc(y_test_num, glmnet_pred_test_vector)$auc

cat("AUC glmnet on Test Data = ", auc_glmnet_test, "\n")


print("Test data AUC evaluation done.")


# Cutoff selection using Test Data

cutoff_values <- seq(0.1, 0.9, by = 0.01)

# Initialize vectors 
sensitivity <- numeric(length(cutoff_values))
specificity <- numeric(length(cutoff_values))
accuracy <- numeric(length(cutoff_values))

# Iterate over the range of cutoff values
for (i in seq_along(cutoff_values)) {
  # Convert predicted probabilities to binary predictions using the current cutoff
  binary_predictions <- ifelse(final_predictions_vector > cutoff_values[i], 1, 0)
  
  # Compute confusion matrix
  confusion <- table(binary_predictions, y_test_num)
  
  # Extract true positives, false positives, true negatives, false negatives
  TP <- confusion[2, 2]
  FP <- confusion[2, 1]
  TN <- confusion[1, 1]
  FN <- confusion[1, 2]
  
  # Calculate sensitivity, specificity, accuracy
  sensitivity[i] <- TP / (TP + FN)
  specificity[i] <- TN / (TN + FP)
  accuracy[i] <- (TP + TN) / sum(confusion)
}

# Choose the cutoff with the maximum combined sensitivity and specificity
combined_performance <- sensitivity + specificity
best_cutoff_index <- which.max(combined_performance)
best_cutoff <- cutoff_values[best_cutoff_index]

# Use the best cutoff to compute the confusion matrix
conf_matrix <- confusionMatrix(data = as.factor(ifelse(final_predictions_vector > best_cutoff, 1, 0)),
                               reference = as.factor(ifelse(y_test_num == 1, 1, 0)))

# Print the confusion matrix
cat("Best Cutoff selected = ", best_cutoff, "\n")
print(conf_matrix)

