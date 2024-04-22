library(caret)
library(glmnet)
library(e1071)  # For SVM
library(randomForest)
library(class)  # For kNN
library(xgboost) 


### DATA LOADING AND SPLITTING
### ------------------------------------------------------

df <- read.csv("ref129_less18_fsiq_1554.csv")

dropped_paras = c('subject_sp_id','derived_cog_impair','asd', 'fsiq', 'fsiq_score', 'fsiq_80')
df <- df[, !(names(df) %in% dropped_paras)]

df$fsiq_70 <- as.logical(df$fsiq_70)
df$fsiq_70 <- as.numeric(df$fsiq_70)

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
train_index <- createDataPartition(df$fsiq_70, p = 0.6, list = FALSE, times = 1)
remaining_data <- df[-train_index, ]

# Now create test and validation sets from the remaining data
test_index <- createDataPartition(remaining_data$fsiq_70, p = 0.5, list = FALSE, times = 1)
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


# For train set
X_train <- train_data[, !(names(train_data) %in% "fsiq_70")]
y_train <- train_data$fsiq_70
y_train_num <- train_data$fsiq_70
y_train <- factor(y_train)  # factorise

# For validation set
X_valid <- valid_data[, !(names(valid_data) %in% "fsiq_70")]
y_valid <- valid_data$fsiq_70
y_valid_num <- valid_data$fsiq_70
y_valid <- factor(y_valid)  # factorise


# For test set
X_test <- test_data[, !(names(test_data) %in% "fsiq_70")]
y_test <- test_data$fsiq_70
y_test_num <- test_data$fsiq_70
y_test <- factor(y_test)  # factorise


print("Splitting done.")

# Model Training
model_glmnet <- glmnet(x = as.matrix(X_train), y = y_train, family = "binomial")
model_svmRadial <- svm(x = as.matrix(X_train), y = y_train, kernel = "radial")
model_rf <- randomForest(x = as.matrix(X_train), y = y_train)
#model_kNN <- knn(train = as.matrix(X_train), test = as.matrix(X_valid), cl = y_train_num)
model_xgbTree <- xgboost(data = as.matrix(X_train), label = y_train, nrounds = 10)

print("Training done.")


# Validation set -> to select the best performing model
valid_perf_glmnet <- predict(model_glmnet, newx = as.matrix(X_valid), type = "class")
valid_perf_svmRadial <- predict(model_svmRadial, newdata = X_valid)
valid_perf_rf <- predict(model_rf, newdata = X_valid)
#valid_perf_kNN <- predict(model_kNN, newdata = as.matrix(X_valid), type = "prob")
valid_perf_xgbTree <- predict(model_xgbTree, newdata = as.matrix(X_valid))

print("Validation done.")

# Testing set -> to decide the predictive probability cutoff and model evaluation
test_perf_glmnet <- predict(model_glmnet, newx = as.matrix(X_test), type = "class")
test_perf_svmRadial <- predict(model_svmRadial, newdata = X_test)
test_perf_rf <- predict(model_rf, newdata = X_test)
# test_perf_kNN <- knn(train = as.matrix(X_train), test = as.matrix(X_test), cl = y_test)
test_perf_xgbTree <- predict(model_xgbTree, newdata = as.matrix(X_test), type="")

print("Testing done.")


# METRICS ------------------------------------


