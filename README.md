# Student Performance Prediction Using Machine Learning

## Project Overview
This project uses supervised machine learning algorithms to predict student academic performance. Based on student-related features, the model classifies students into two categories: Pass or Fail.

## Objectives
- Understand and implement supervised machine learning.
- Train classification models using student performance data.
- Compare Decision Tree and Random Forest algorithms.
- Evaluate model performance using different evaluation metrics.
- Predict the academic result of a new student.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset Description

The dataset contains the following features:

| Feature | Description |
|---|---|
| Study_Hours | Number of hours spent studying |
| Attendance | Student attendance percentage |
| Previous_Marks | Previous examination marks |
| Assignment_Marks | Assignment marks |
| Internal_Marks | Internal assessment marks |
| Result | Target variable (Pass/Fail) |

**Dataset size:** 12 records.

Note: This is a small sample dataset created for learning and demonstration purposes. It is not representative of real-world student populations.

## Machine Learning Algorithms

### 1. Decision Tree Classifier
A supervised learning algorithm that makes predictions using a tree-like structure of decisions.

### 2. Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to make predictions.

## Model Evaluation

The models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score

### Results

| Evaluation Metric | Result |
|---|---:|
| Decision Tree Accuracy | 66.67% |
| Random Forest Accuracy | 66.67% |
| Random Forest AUC | 1.00 |

These results are based on a test set of only three records. Therefore, they should not be interpreted as reliable estimates of real-world model performance.

## Project Structure

```text
Student-Performance-Prediction/
│
├── predictive_model.py
├── student_performance.csv
└── README.md
```

## How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/dharshini-nanthakumar/Student-Performance-Prediction.git
```

2. Navigate to the project folder:

```bash
cd Student-Performance-Prediction
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the Python file:

```bash
python predictive_model.py
```

## Expected Outcome

The project demonstrates how supervised machine learning can be used to classify student performance and predict whether a student is likely to Pass or Fail.

## Author

**Dharshini Nanthakumar**

BE Computer Science and Engineering  
GitHub: [dharshini-nanthakumar](https://github.com/dharshini-nanthakumar)
