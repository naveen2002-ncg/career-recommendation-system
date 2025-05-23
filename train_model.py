import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/career_data.csv')

# Parse multi-values separated by ';'
df['skills'] = df['Skills'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
df['certifications'] = df['Certifications'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
df['industries'] = df['Preferred_Industry'].apply(lambda x: x.split(';') if isinstance(x, str) else [])

# Rename for consistency
df['education'] = df['Education_Level']
df['career'] = df['Recommended_Career']

# Show class balance
print("\n🎯 Class Distribution:\n", df['career'].value_counts())

# Encode multi-label features
skill_mlb = MultiLabelBinarizer()
skill_encoded = skill_mlb.fit_transform(df['skills'])

cert_mlb = MultiLabelBinarizer()
cert_encoded = cert_mlb.fit_transform(df['certifications'])

industry_mlb = MultiLabelBinarizer()
industry_encoded = industry_mlb.fit_transform(df['industries'])

# Encode education and career
edu_le = LabelEncoder()
edu_encoded = edu_le.fit_transform(df['education'])

target_le = LabelEncoder()
y = target_le.fit_transform(df['career'])

# Combine features
X = np.hstack([
    skill_encoded,
    cert_encoded,
    industry_encoded,
    edu_encoded.reshape(-1, 1),
    df['Experience'].values.reshape(-1, 1)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model with class balancing
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Save model and encoders
joblib.dump(clf, 'model/random_forest_model.joblib')
joblib.dump(skill_mlb, 'model/skill_mlb.joblib')
joblib.dump(cert_mlb, 'model/cert_mlb.joblib')
joblib.dump(industry_mlb, 'model/industry_mlb.joblib')
joblib.dump(edu_le, 'model/edu_le.joblib')
joblib.dump(target_le, 'model/target_le.joblib')

# Feature importance plot
importances = clf.feature_importances_
plt.figure(figsize=(12, 4))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("model/feature_importance.png")
print("📊 Feature importance plot saved as model/feature_importance.png")

print("\n✅ Training complete and models saved.")
