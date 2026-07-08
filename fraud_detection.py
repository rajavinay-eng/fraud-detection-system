import pandas as pd

# Load real fraud data
df = pd.read_csv(r'C:/Users/RAJA VINAY KUMAR/Downloads/job preparation/onlinefraud.csv')

# Check data
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass balance:")
print(df['isFraud'].value_counts())
print("\nFraud percentage:", round(df['isFraud'].mean()*100, 2), "%")

# Remove useless columns
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

print("After removing useless columns:")
print(df.columns.tolist())
print("New shape:", df.shape)

# Encode text column — type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
print("\nAfter encoding type column:")
print(df['type'].value_counts())
print("\nFirst 5 rows:")
print(df.head())

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# stratify=y means split keeps same fraud ratio
# in both train and test sets

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("\nFraud in training:", y_train.sum())
print("Fraud in test:", y_test.sum())

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling done.")

# Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("SMOTE done.")
print("\nAfter SMOTE:")
print("Training genuine:", sum(y_train==0))
print("Training fraud:", sum(y_train==1))

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("\nTraining XGBoost model...")
print("This will take a few minutes. Please wait.")

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)
print("Training complete.")

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n=== MODEL RESULTS ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred), 4))

import shap

print("\nGenerating SHAP explanations...")

# Use sample of test data — full test set too large for SHAP
X_test_sample = X_test[:1000]

# Create explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test_sample)

# Show which features matter most
print("\nTop features causing fraud predictions:")
shap.summary_plot(
    shap_values,
    X_test_sample,
    plot_type="bar",
    feature_names=['step','type','amount','oldbalanceOrg',
                   'newbalanceOrig','oldbalanceDest','newbalanceDest'],
    show=True
)
print("SHAP done.")

import joblib

# Save trained model
joblib.dump(model, r'C:/Users/RAJA VINAY KUMAR/Downloads/job preparation/fraud_model.pkl')
print("\nModel saved successfully as fraud_model.pkl")

# Save scaler too — needed for new predictions
joblib.dump(scaler, r'C:/Users/RAJA VINAY KUMAR/Downloads/job preparation/fraud_scaler.pkl')
print("Scaler saved successfully as fraud_scaler.pkl")

# Test loading
loaded_model = joblib.load(r'C:/Users/RAJA VINAY KUMAR/Downloads/job preparation/fraud_model.pkl')
print("Model loaded successfully.")
print("Model is ready for deployment.")