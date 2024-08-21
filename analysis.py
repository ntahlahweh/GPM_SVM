import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the provided CSV file
file_path = 'analysis.csv'
df = pd.read_csv(file_path)

df_rbf = df[df['Kernel'] == 'rbf']
df_sigmoid = df[df['Kernel'] == 'sigmoid']

rbf_summary = df_rbf[['Training Accuracy', 'Test Accuracy']].describe()
sigmoid_summary = df_sigmoid[['Training Accuracy', 'Test Accuracy']].describe()

print("RBF Kernel Summary:\n", rbf_summary)
print("\nSigmoid Kernel Summary:\n", sigmoid_summary)
print(df_rbf.index)

# Boxplot for Training and Test Accuracy
plt.figure(figsize=(12, 6))
sns.boxplot(x='Kernel', y='Training Accuracy', data=df)
plt.title('Training Accuracy Comparison: RBF vs Sigmoid')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Kernel', y='Test Accuracy', data=df)
plt.title('Test Accuracy Comparison: RBF vs Sigmoid')
plt.show()

# Line plot for Test Accuracy across hyperparameters
# Reset index to ensure both start from 0 or have aligned indices
df_rbf = df_rbf.reset_index(drop=True)
df_sigmoid = df_sigmoid.reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(df_rbf.index, df_rbf['Test Accuracy'], label='RBF Test Accuracy', marker='o')
plt.plot(df_sigmoid.index, df_sigmoid['Test Accuracy'], label='Sigmoid Test Accuracy', marker='o')
plt.xlabel('Hyperparameter Combination Index')
plt.ylabel('Test Accuracy')
plt.title('RBF vs Sigmoid Test Accuracy Comparison')
plt.legend()
plt.show()


