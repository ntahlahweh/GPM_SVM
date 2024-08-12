import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# read data from csv file
training_data = pd.read_csv('train_20052022.csv')
test_data = pd.read_csv('test_20052022.csv')

# prepare data for applying it to svm
x_train = training_data.iloc[:, 1:].values  # data
y_train = training_data.iloc[:, 0].values  # target
x_test = test_data.iloc[:, 1:].values  # data
y_test = test_data.iloc[:, 0].values  # target

# Define the hyperparameter grid
param_grid = {
    'kernel': ('linear', 'rbf', 'poly'),
    'C': [5, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'degree': [1, 2, 3, 4, 5, 6]
}

# Set up the grid search with cross-validation
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)

# Best hyperparameters found by GridSearchCV
print(f"Best hyperparameters found: {grid.best_params_}")

# Use the best model to predict the test data
y_pred = grid.predict(x_test)

# Creating confusion matrix and calculating accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = float(cm.diagonal().sum()) / len(y_test)
print('Model accuracy is:', accuracy * 100, '%')

# Optional: You can also get the best estimator
best_model = grid.best_estimator_
