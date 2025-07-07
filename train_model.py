import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('diabetes.csv')

# Replace 0s with median in specific columns
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
X_res, y_res = SMOTE().fit_resample(X_scaled, y)

# Train model
model = GradientBoostingClassifier()
params = {'n_estimators':[100,200], 'learning_rate':[0.01,0.1], 'max_depth':[3,5]}
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_res, y_res)

# Save model & scaler
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model & Scaler saved successfully.")
