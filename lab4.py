import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_excel(r"C:\Users\vijay\Documents\sem4\ML\lab4\training_mathbert 2.xlsx")

# Calculate class centroids, class spreads, and interclass distance
mathbert_columns = [col for col in dataset.columns if col.startswith("embed_")]
mathbert_data = dataset[mathbert_columns]
mathbert_data['output'] = dataset['output']

# Filter out invalid output values
valid_output_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
mathbert_data = mathbert_data[mathbert_data['output'].isin(valid_output_values)]

# Group data by output values and calculate class centroids and spreads
grouped_data = mathbert_data.groupby('output')
class_centroids = grouped_data.mean()
class_spreads = grouped_data.std()

# Calculate interclass distance between two classes
class_1 = class_centroids.iloc[0]
class_2 = class_centroids.iloc[1]
interclass_distance = np.linalg.norm(class_1 - class_2)

print("Class Centroids:")
print(class_centroids)
print("\nClass Spreads:")
print(class_spreads)
print("\nInterclass Distance between Class 1 and Class 2:", interclass_distance)

# Plot histogram of a selected feature
feature = 'embed_0'
feature_values = dataset[feature]

plt.figure(figsize=(10, 6))
plt.hist(feature_values, bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {feature}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate mean and variance of the selected feature
feature_mean = np.mean(feature_values)
feature_variance = np.var(feature_values)
print(f"Mean of {feature}: {feature_mean}")
print(f"Variance of {feature}: {feature_variance}")

# Calculate Minkowski distance for different values of r
feature_1 = 'embed_0'
feature_2 = 'embed_1'
X = dataset[[feature_1, feature_2]].values

# Split dataset into train and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Calculate Minkowski distance for different values of r
r_values = range(1, 11)
distances = []

for r in r_values:
    distance_r = np.linalg.norm(X_train[:, 0] - X_train[:, 1], ord=r)
    distances.append(distance_r)

# Plot the distance versus r
plt.figure(figsize=(10, 6))
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
