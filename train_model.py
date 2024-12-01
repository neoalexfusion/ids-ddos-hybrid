import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv('CICDDoS2019.csv')  # Replace with your dataset file name

# Preprocessing: Select key features
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
            'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Flow Bytes/s']
target = 'Class'  # Column with "Attack" or "Benign" labels

# Drop rows with missing values
data = data.dropna()

# Define features (X) and target (y)
X = data[features]
y = data[target].map({'Attack': 0, 'Benign': 1})  # Encode target labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 1: Train Isolation Forest with adjusted contamination
isolation_forest = IsolationForest(contamination=0.02, max_samples=256, random_state=42)
isolation_forest.fit(X_train)

# Predict anomalies in the test set
anomalies = isolation_forest.predict(X_test)

# Analyze anomalies
anomaly_df = pd.DataFrame({
    'Anomaly': anomalies,
    'True Label': y_test
})

anomaly_counts = anomaly_df.groupby(['True Label', 'Anomaly']).size().unstack(fill_value=0)
print("Anomaly Counts:\n", anomaly_counts)

# Convert Isolation Forest outputs
anomalies = anomalies  # Use raw output (-1 for anomaly, 1 for normal)

# Step 2: Train Random Forest Classifier
rf = RandomForestClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=20,
    class_weight={0: 1, 1: 3},  # Increase weight for benign
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predict probabilities with the Random Forest
rf_probs = rf.predict_proba(X_test)[:, 1]  # Probabilities for benign class

# Adjust the decision threshold
threshold = 0.7  # Increased threshold
rf_predictions = (rf_probs >= threshold).astype(int)

# Alternative Decision Logic
final_predictions = []
for anomaly, rf_pred in zip(anomalies, rf_predictions):
    if anomaly == -1:  # If flagged as anomaly
        # Classify as attack only if Random Forest also predicts attack
        if rf_pred == 0:
            final_predictions.append(0)  # Attack
        else:
            final_predictions.append(1)  # Benign
    else:
        final_predictions.append(rf_pred)  # Use Random Forest prediction

# Evaluate the hybrid model
print("Classification Report:\n", classification_report(y_test, final_predictions))

# Confusion matrix
cm = confusion_matrix(y_test, final_predictions, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Attack', 'Benign'])
disp.plot()
plt.show()

# Feature Importance Analysis
importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)

# Save the trained model and features
joblib.dump(rf, 'hybrid_model_rf.pkl')
joblib.dump(isolation_forest, 'hybrid_model_isolation_forest.pkl')
joblib.dump(features, 'hybrid_model_features.pkl')
print("Hybrid model saved successfully!")
