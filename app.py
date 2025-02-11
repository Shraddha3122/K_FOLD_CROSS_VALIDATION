import pandas as pd
from flask import Flask, jsonify
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/K FOLD CROSS VALIDATION/breast-cancer-wisconsin.data', header=None)

# Assign column names based on the dataset documentation
data.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# Drop the ID column
data = data.drop(columns=['ID'])

# Convert 'Bare Nuclei' to numeric, forcing errors to NaN
data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'], errors='coerce')

# Drop rows with any NaN values
data = data.dropna()

# Prepare the features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Convert target variable to numeric if it's not already
y = pd.to_numeric(y, errors='coerce')

# Ensure that the target variable does not contain NaN values
y = y.dropna()
X = X.loc[y.index]  # Align X with y after dropping NaNs

# Initialize K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Store accuracy scores
accuracy_scores = []

# K-Fold Cross Validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

# Calculate average accuracy
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

# Create Flask application
app = Flask(__name__)

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify({'average_accuracy': average_accuracy})

if __name__ == '__main__':
    app.run(debug=True)