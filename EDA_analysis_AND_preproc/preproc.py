import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA 


#Min max for GA and standard scaler for EDA 

file_path = "Cor_data.csv"
df = pd.read_csv(file_path)

print(df.head())

# Encode target column ('Cath') to binary (Cad -> 1, Normal -> 0)
df['Target_class'] = df['Cath'].apply(lambda x: 1 if x.lower() == 'cad' else 0)

# Handling categorical columns (one-hot encoding for 'BBB' and 'VHD')
df = pd.get_dummies(df, drop_first=True)
print(df['Target_class'].value_counts())

# Handle missing values: Dropping rows with NaN values
df.dropna(inplace=True)

# EDA - Distribution of numerical columns
plt.figure(figsize=(14, 10))
df.hist(figsize=(14, 10), bins = 20)

plt.tight_layout()
plt.show()

# Splitting features and target
X = df.drop(columns=['Target_class'])  # Αφαιρούμε μόνο την 'Target_class' για να κρατήσουμε τα χαρακτηριστικά
y = df['Target_class']

# Standardizing the data (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA graph for visualizing the 2D components
pca = PCA(n_components=2)  # We only need the first 2 components for the 2D plot
principal_components = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Ensure the target column ('Target_class') is added to the PCA DataFrame for coloring
pc_df['Target_class'] = df['Target_class'].values

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Scatter plot for PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Target_class',  # The column used for coloring the points
    data=pc_df,
    palette='viridis',  # Select a color palette
    s=70,               # Size of the points
    alpha=0.7           # Transparency
)

# Add titles and labels
plt.title(f'PCA Scatter Plot - Principal Component 1 vs 2')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% variance)')
plt.legend(title='Target Class')
plt.grid(True)
plt.show()

# Print the shape of the preprocessed data
print(f"Dataset shape after preprocessing: {X_scaled.shape}")

# Printing explained variance ratios
print(f"Explained variance by PC1: {explained_variance[0]*100:.2f}%")
print(f"Explained variance by PC2: {explained_variance[1]*100:.2f}%")


# Saving the preprocessed data (X_scaled) and the target (y)
preprocessed_data = pd.DataFrame(X_scaled, columns=X.columns)
preprocessed_data['Target_class'] = y

# Export to CSV file
preprocessed_data.to_csv('preprocessed_data.csv', index=False)

print("The preprocessed data has been saved to 'preprocessed_data.csv'")
