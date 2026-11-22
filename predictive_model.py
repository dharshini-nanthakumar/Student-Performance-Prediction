import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
data = pd.read_csv("student_performance.csv")
print(data)
print("Dataset Shape:", data.shape)
data.info()
print(data.describe())
print(data.isnull().sum())
print("Duplicate Records:", data.duplicated().sum())
sns.countplot(x="Result", data=data)
plt.title("Student Pass and Fail Count")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.show()
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Study_Hours",
    y="Previous_Marks",
    hue="Result",
    data=data,
    s=100
)
plt.title("Study Hours vs Previous Marks")
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.show()
sns.countplot(x="Result", data=data)
plt.title("Pass vs Fail Students")
plt.show()
X = data.drop("Result", axis=1)
y = data["Result"]
print("Features:")
print(X)
print("\nTarget:")
print(y)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Encoded Target:")
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed successfully!")
y_pred = model.predict(X_test)
print("Actual Results:", y_test)
print("Predicted Results:", y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("Random Forest training completed!")
rf_pred = rf_model.predict(X_test)
print("Actual Results:", y_test)
print("Random Forest Predictions:", rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy * 100, "%")
print("\nMODEL COMPARISON")
print("Decision Tree Accuracy:", accuracy * 100, "%")
print("Random Forest Accuracy:", rf_accuracy * 100, "%")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Generate confusion matrix

cm = confusion_matrix(y_test, rf_pred)

print("Confusion Matrix:")
print(cm)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)
display.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report
print("Classification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)
from sklearn.metrics import roc_curve, roc_auc_score
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
new_student = pd.DataFrame(
    [[7, 90, 80, 85, 82]],
    columns=X.columns
)
prediction = rf_model.predict(new_student)
result = encoder.inverse_transform(prediction)
print("New Student Prediction:", result[0])