import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


#---Preprocess the Data---

# sample size
N_SAMPLES = 1_00_000

# load and take random sample of train data set
train_df = pd.read_csv('train/train.csv')
if len(train_df) > N_SAMPLES:
    train_df = train_df.sample(n=N_SAMPLES, random_state=42)

# Load validation and test ds
val_df = pd.read_csv('validation/validation.csv')
test_df = pd.read_csv('test/test.csv')

# reset indices after sampling
train_df = train_df.reset_index(drop=True)

# Separate Features & Labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Handle Preprocessing
# Separate numeric and categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# appears to be no categorical features
# print("Categorical columns:", categorical_cols.tolist())
# print("Numeric columns:", numeric_cols.tolist())

scaler = StandardScaler()
label_encoder = LabelEncoder()

# create copies to preserve indices
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# fit scaler on x_train numerical features
scaler.fit(X_train[numeric_cols])

# transform sets
X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# fit encoder on training labels
label_encoder.fit(y_train)

# transform all three label sets
y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# inspect mapping of attacks
# print("\nLabel mapping:")
# for i, class_name in enumerate(label_encoder.classes_):
#     print(f"{class_name}  ->  {i}")



#---Aqua Noise detection---

try:
    from aqua.data.process_data import Aqdata
except ImportError:
    print("Error: Could not import Aqdata.")
    print("Please ensure you have cloned 'autonlab/aqua' and run 'python setup.py install'")
    # Stop the script if it fails
    raise

# Wrap your data in the Aqdata object
train_data = Aqdata(
    data=X_train_scaled.values, 
    ground_labels=y_train_encoded
)

val_data = Aqdata(
    data=X_val_scaled.values, 
    ground_labels=y_val_encoded
)

test_data = Aqdata(
    data=X_test_scaled.values, 
    ground_labels=y_test_encoded
)

print("\nSuccessfully wrapped data in Aqdata objects.")
print(f"Training data object: {type(train_data)}")
