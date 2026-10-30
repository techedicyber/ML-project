
# =====================================================
# PHISHING WEBSITE DETECTION USING MACHINE LEARNING
# Algorithms: Logistic Regression, Decision Tree,
# Random Forest and AdaBoost
# =====================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)

print("Libraries imported successfully!")


# 2. UPLOAD DATASET

uploaded = files.upload()

file_name = "uci_raw.csv"
df = pd.read_csv(file_name)

print("\nDataset uploaded successfully!")
print("Dataset shape:", df.shape)

display(df.head())


# 3. DATASET INFORMATION

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# 4. DATA PREPROCESSING

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Check target column
if "Result" not in df.columns:
    raise ValueError("Target column 'Result' not found. Check your CSV column names.")

# Convert target into numeric values
df["Result"] = pd.to_numeric(df["Result"], errors="coerce")

# Remove rows with missing target values
df = df.dropna(subset=["Result"])

# Convert feature columns to numeric
X = df.drop("Result", axis=1).apply(pd.to_numeric, errors="coerce")

# Fill missing feature values using median
X = X.fillna(X.median())

# Target variable
y = df["Result"].astype(int)

# Check target labels
if not set(y.unique()).issubset({-1, 1}):
    raise ValueError("Expected target labels -1 and +1 in Result column.")

print("\nPreprocessing completed!")
print("Features:", X.shape)
print("Target:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# 5. SPLIT DATASET INTO TRAINING AND TESTING

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 6. CREATE MACHINE LEARNING MODELS

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    )
}


# 7. TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, pos_label=-1, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=-1, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=-1, zero_division=0
    )

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[-1, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))


# 8. DISPLAY MODEL COMPARISON

results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON:")
display(results_df.round(4))


# 9. ACCURACY COMPARISON GRAPH

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Accuracy Comparison of Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.xlabel("Machine Learning Algorithms")
plt.xticks(rotation=15)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# 10. PRECISION, RECALL AND F1 SCORE GRAPH

metrics_df = results_df.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=metrics_df,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Precision, Recall and F1 Score Comparison")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# 11. CONFUSION MATRICES

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[-1, 1]
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"]
    )

    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()


# 12. SELECT BEST MODEL BASED ON ACCURACY

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(), "Model"
]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

print(
    "Best Accuracy:",
    round(results_df["Accuracy"].max(), 2),
    "%"
)


# 13. SAMPLE PREDICTION

sample = X_test.iloc[[0]]

prediction = best_model.predict(sample)[0]

print("\nSample Website Prediction:")

if prediction == -1:
    print("Result: Phishing Website")
else:
    print("Result: Legitimate Website")


# 14. SAVE BEST MODEL

joblib.dump(best_model, "phishing_detection_model.pkl")

print("\nBest model saved successfully!")


# 15. DOWNLOAD MODEL AND RESULTS

results_df.to_csv("model_results.csv", index=False)

files.download("phishing_detection_model.pkl")
files.download("model_results.csv")

print("\nProject execution completed successfully!")
