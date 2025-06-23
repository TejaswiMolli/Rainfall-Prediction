import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_csv('Rainfall.csv')

# Data Preprocessing
for col in df.select_dtypes(include=[np.number]).columns:
    df.loc[:, col] = df[col].fillna(df[col].mean())
    df = df.replace({'yes':1, 'no':0}).infer_objects(copy=False)
    df.info()
features = df.drop(['day', 'rainfall'], axis=1)
target = df['rainfall']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)

# Handling class imbalance
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Normalizing data
scaler = StandardScaler()
#print("Scaled Inputs:", scaler.transform([50, 1015, 5])) # Example input
print("x test", type(X_test), X_test)
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel='rbf', probability=True)
model.fit(X_resampled, y_resampled)

# Save model & scaler
pickle.dump(model, open('rainfall_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))