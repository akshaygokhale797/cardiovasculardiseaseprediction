import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load cardiovascular disease dataset
# Features include:
    # Age in days and years
    # Gender (1: Female, 2: Male)
    # Cholesterol (1: normal, 2: above normal, 3: well above normal)
    # Height of the patient in centimeters
    # Weight in kilograms
    # Systolic blood pressure (ap_hi) - top num in reading representing max pressure in arteries
    # Diastolic blood pressure (ap_lo) - bottom num representing min pressure prior to next contraction
    # Cholesterol levels (1: Normal, 2: Above Normal, 3: Well Above Normal)
    # Glucose levels (1: Normal, 2: Above Normal, 3: Well Above Normal)
    # Smoking status (0: Non-smoker, 1: Smoker)
    # Alcohol intake (0: Does not consume alcohol, 1: Consumes alcohol)
    # Physical activity (0: Not physically active, 1: Physically active)
    # Presence or absence of cardiovascular disease (0: Absence, 1: Presence)
    # BMI from weight and height
    # Blood pressure category (Normal, Elevated, Hypertension Stage 1/2, Hypertensive Crisis)
df = pd.read_csv("C:\\Users\\aksha\\OneDrive\\Desktop\\Python\\cardio_disease_prediction\\cardio_data_processed.csv")
df = df.drop(columns=['id'])
# BMI calculations from kg and cm
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
# Defined bins for feature engineered gender/weight categories
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
df['age_range_gender'] = df.apply(
    lambda row: f"{'Male' if row['gender'] == 2 else 'Female'}-{pd.cut([row['age_years']], bins=bins, labels=labels)[0]}",
    axis=1
)
X = df.drop(['cardio'], axis=1)  
y = df['cardio']  # Target variable
categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_range_gender', 'bp_category_encoded']
numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']

# PREPROCESSING
# Impute missing values and scale features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
# Encoder to transform categorical data to numerical frequency columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
# Define the XGBClassifier with the best parameters
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.2,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
# Pipeline for preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Train the model
pipeline.fit(X_train, y_train)
# Predictions
y_pred = pipeline.predict(X_val)
# Analysis metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
