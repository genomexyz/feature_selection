import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random seed for reproducibility
np.random.seed(0)

# Number of data points
num_samples = 10000

# Create the three features with some random noise
#feature1 = np.random.rand(num_samples) + np.random.normal(0, 0.5, num_samples)
#feature2 = np.random.rand(num_samples) + np.random.normal(0, 0.5, num_samples)
#feature3 = np.random.rand(num_samples) + np.random.normal(0, 0.5, num_samples)

feature1 = np.random.normal(0, 0.5, num_samples)
feature2 = np.random.normal(0, 0.5, num_samples)
feature3 = np.random.normal(0, 0.5, num_samples)

# Create the target as a combination of the features with noise
# Feature 1 influences the target by its absolute magnitude
target = 3 * np.abs(feature1) + 1 * feature2 - 1 * feature3 + np.random.normal(0, 0.05, num_samples)

# Combine the features and the target into a DataFrame
data = pd.DataFrame({
    'Feature1': feature1,
    'Feature2': feature2,
    'Feature3': feature3,
    'Target': target
})

# Display the first few rows of the dataset
print(data.head())

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()