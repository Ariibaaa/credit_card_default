# Credit Default Prediction Model

This project demonstrates the development and evaluation of a logistic regression model to predict credit defaults based on a dataset.

## Dataset

The dataset used for this project is available on [GitHub](https://github.com/YBI-Foundation/Dataset/raw/main/Credit%20Default.csv). It contains the following columns:

- Income: The income of the individual.
- Age: The age of the individual.
- Loan: The loan amount.
- Loan to Income: The ratio of the loan amount to income.
- Default: The target variable, where 1 represents a default and 0 represents no default.

### Data Information

The dataset consists of 2000 rows and 5 columns. It includes both numerical features and a binary classification target.

## Model

We trained a logistic regression model to predict credit defaults using the provided dataset. Here's a summary of the model:

- Features: Income, Age, Loan, Loan to Income
- Target: Default

The model achieved the following results on the test set:

- Accuracy: 95%
- Precision: 83%
- Recall: 79%
- F1-Score: 81%

## Usage

1. Clone the repository or download the dataset from [here](https://github.com/YBI-Foundation/Dataset/raw/main/Credit%20Default.csv).
2. Install the required libraries if not already installed (`pandas`, `scikit-learn`).
3. Run the provided Jupyter Notebook or Python script to train and evaluate the model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
default = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Credit%20Default.csv')

# Split the data into training and testing sets
X = default.drop(['Default'], axis=1)
y = default['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
confusion_matrix_result = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(confusion_matrix_result)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)
