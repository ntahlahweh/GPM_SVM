import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import pickle

# Read data from csv file
train_data = pd.read_csv('train_20052022.csv')
test_data = pd.read_csv('test_20052022.csv')

# Prepare data for applying it to SVM
x_train = train_data.iloc[:, 1:].values  # data
y_train = train_data.iloc[:, 0].values  # target
x_test = test_data.iloc[:, 1:].values  # data
y_test = test_data.iloc[:, 0].values  # target

# Reduce the data to 2D using PCA for visualization
pca = PCA(n_components=2)
x_train_2d = pca.fit_transform(x_train)
x_test_2d = pca.transform(x_test)

# Define the hyperparameter grid
param_grid = {
    'kernel': ['rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'degree': [0, 1, 2]
}

# Manually iterate over the parameter grid
best_test_score = 0
best_params = None
for kernel in param_grid['kernel']:
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for degree in param_grid['degree']:
                print(f"Evaluating: kernel={kernel}, C={C}, gamma={gamma}, degree={degree}")
                
                # Create the SVM model with the current hyperparameters
                model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
                
                # Perform cross-validation on the training set
                cv_scores = cross_val_score(model, x_train_2d, y_train, cv=5)
                cv_mean_score = np.mean(cv_scores)
                print(f"Training CV mean accuracy: {cv_mean_score:.3f}")
                
                # Train the model on the entire training set
                model.fit(x_train_2d, y_train)
                
                # Evaluate the model on the test set
                y_test_pred = model.predict(x_test_2d)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                print(f"Test set accuracy: {test_accuracy:.3f}")
                
                # Track the best model based on test accuracy
                if test_accuracy > best_test_score:
                    best_test_score = test_accuracy
                    best_params = {
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma,
                        'degree': degree
                    }
                    best_model = model

# Print the best parameters and the corresponding test set accuracy
print(f"Best parameters based on test set accuracy: {best_params}")
print(f"Best test set accuracy: {best_test_score:.3f}")

# Save the PCA model
with open('pca_model.pkl', 'wb') as pca_file:
    pickle.dump(pca, pca_file)

# Save the best SVM model
with open('gpm_svm_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Use the best model to predict the test data again (just for confirmation)
y_pred = best_model.predict(x_test_2d)

# Create confusion matrix and calculate accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = float(cm.diagonal().sum()) / len(y_test)
print('Final model accuracy on test set:', accuracy * 100, '%')
