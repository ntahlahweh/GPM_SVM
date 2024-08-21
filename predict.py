import pandas as pd
import pickle
import os

# Define file paths
model_name = "gpm_svm_model.pkl"
pca_model_name = "pca_model.pkl"
csv_file_name = 'test_2.csv'  # Assuming new data for prediction is in this file

try:
    # Read data from CSV file
    new_data = pd.read_csv(csv_file_name)

    # Extract features from the data (assuming the first column is no longer the target)
    x_new = new_data.values  # all columns are features now

    # Load the saved PCA model
    with open(pca_model_name, 'rb') as pca_file:
        pca = pickle.load(pca_file)

    # Transform the new data using the loaded PCA model
    x_new_2d = pca.transform(x_new)

    # Load the saved SVM model
    with open(model_name, 'rb') as model_file:
        load_model = pickle.load(model_file)

    # Make predictions using the loaded SVM model
    prediction = load_model.predict(x_new_2d)

    # Print predictions
    print("Predicted classes:", prediction)

    # Optionally, you can save the predictions to a CSV file
    output_df = pd.DataFrame({'Prediction': prediction})
    output_df.to_csv('predictions.csv', index=False)

except FileNotFoundError as e:
    print(f"File not found: {e.filename}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
