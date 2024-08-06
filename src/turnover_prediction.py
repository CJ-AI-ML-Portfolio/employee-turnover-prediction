import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Extract the ZIP files
def extract_zip(file_path, extract_to="."):
    with ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


# Paths to the uploaded files
churn_path = "/data/churn.csv"
comments_zip_path = "/data/comments_clean_anonimized.csv.zip"
votes_zip_path = "/data/votes.csv.zip"
interactions_zip_path = "/data/commentInteractions.csv.zip"

# Extract the contents of the ZIP files
extract_zip(comments_zip_path)
extract_zip(votes_zip_path)
extract_zip(interactions_zip_path)

# Load the datasets
churn_df = pd.read_csv(churn_path)
comments_df = pd.read_csv("comments_clean_anonimized.csv")
votes_df = pd.read_csv("votes.csv")
interactions_df = pd.read_csv("commentInteractions.csv")

# Display the first few rows of each dataset
print("Churn Dataset:")
print(churn_df.head())

print("\nComments Dataset:")
print(comments_df.head())

print("\nVotes Dataset:")
print(votes_df.head())

print("\nInteractions Dataset:")
print(interactions_df.head())
# Example: Merge datasets on a common key (e.g., 'user_id')
# Ensure to replace 'user_id' with the actual common column name if it differs

# Merging datasets
combined_df = pd.merge(churn_df, comments_df, on="user_id", how="inner")
combined_df = pd.merge(combined_df, votes_df, on="user_id", how="inner")
combined_df = pd.merge(combined_df, interactions_df, on="user_id", how="inner")

# Display combined dataset info
print("\nCombined Dataset Info:")
print(combined_df.info())

# Handle missing values and encode categorical variables if necessary
# Example preprocessing step: filling missing values
combined_df.fillna(method="ffill", inplace=True)

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
for column in combined_df.select_dtypes(include=["object"]).columns:
    combined_df[column] = le.fit_transform(combined_df[column])

# Display the first few rows of the combined DataFrame
print("\nCombined Dataset:")
print(combined_df.head())


# Define features and target (Replace 'turnover' with your target column name)
X = combined_df.drop(["turnover", "user_id"], axis=1, errors="ignore")
y = combined_df["turnover"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Hyperparameter tuning with Grid Search
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the best model
best_y_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_y_pred)
best_report = classification_report(y_test, best_y_pred)

print(f"\nTuned Model Accuracy: {best_accuracy:.2f}")
print("\nTuned Classification Report:")
print(best_report)

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Visualization: Feature Importance
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# Visualization: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(combined_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
