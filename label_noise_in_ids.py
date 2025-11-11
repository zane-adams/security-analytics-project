import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
from sklearn.metrics import classification_report, f1_score


#---Preprocess the Data---

# train sample does not encapsulate all attacks so use stratified sample instead
train_df = pd.read_csv('train/train_stratified_sample.csv')

# Load validation and test ds
# val_df = pd.read_csv('validation/validation.csv')
test_df = pd.read_csv('test/test.csv')
test_df = test_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# reset indices after sampling
train_df = train_df.reset_index(drop=True)

# Separate Features & Labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

# X_val = val_df.drop(columns=["label"])
# y_val = val_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Handle Preprocessing
# Separate numeric and categorical columns
# categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# appears to be no categorical features
# print("Categorical columns:", categorical_cols.tolist())
# print("Numeric columns:", numeric_cols.tolist())

scaler = StandardScaler()
label_encoder = LabelEncoder()

# create copies to preserve indices
X_train_scaled = X_train.copy()
# X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# fit scaler on x_train numerical features
scaler.fit(X_train[numeric_cols])

# transform sets
X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
# X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Fit LabelEncoder on ALL labels to prevent 'unseen label' error
all_labels = pd.concat([y_train, y_test]).unique()
label_encoder.fit(all_labels)

# transform all three label sets
y_train_encoded = label_encoder.transform(y_train)
# y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# inspect mapping of attacks
# print("\nLabel mapping:")
# for i, class_name in enumerate(label_encoder.classes_):
#     print(f"{class_name}  ->  {i}")


#---Baseline Evaluation---

# convert to numpy arrays for scikit-learn
X_train_np = X_train_scaled.values
y_train_np = y_train_encoded
X_test_np = X_test_scaled.values
y_test_np = y_test_encoded

# random forest

rf_model = RandomForestClassifier(random_state=42, n_jobs=4, min_samples_leaf=5)

# train the model
rf_model.fit(X_train_np, y_train_np)

# Evaluate the model
y_pred = rf_model.predict(X_test_np)

report = classification_report(
    y_test_np, 
    y_pred, 
    target_names=label_encoder.classes_
)
print(report)
baseline_f1 = f1_score(y_test_np, y_pred, average='weighted')
print(f"Baseline Weighted F1-Score: {baseline_f1:.4f}")
