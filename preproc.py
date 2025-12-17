import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns  # Εισαγωγή seaborn για το heatmap
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder


file_path = 'Cor_data.csv'
df = pd.read_csv(file_path)

print(df.head())

binary_map = {
    "n": 0, "no": 0, "0": 0, "false": 0, "f": 0, 
    "y": 1, "yes": 1, "1": 1, "true": 1, "t": 1, 
    "male": 1, "m": 1,
    "female": 0, "fem": 0, "woman": 0, "w": 0
}

ordinal_map = {
    "none": 0, "mild": 1, "moderate": 2, "severe": 3, "very severe": 4
}

LabelEncoder_insatnce = LabelEncoder()
df['Sex'] = LabelEncoder_insatnce.fit_transform(df['Sex'])
df['Cath'] = LabelEncoder_insatnce.fit_transform(df['Cath'])


print(df["Cath"].value_counts())
# Επιλογή μόνο των αριθμητικών στηλών για τυποποίηση
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df_numeric = df[numeric_cols].copy()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Δημιουργία νέου DataFrame για τα τυποποιημένα δεδομένα
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

print("--- Πρώτες 5 γραμμές του DataFrame μετά την τυποποίηση (Z-score) ---")
print(df_scaled.head())
print("-" * 40)

pca = PCA()
principal_components = pca.fit_transform(df_scaled)

pc_cols = [f'PC{i+1}' for i in range(len(df_scaled.columns))]
pc_df = pd.DataFrame(data=principal_components, columns=pc_cols)

print("--- Πρώτες 5 γραμμές του DataFrame μετά την PCA ---")
print(pc_df.head())
print("-" * 40)

# --- Προετοιμασία για Οπτικοποίηση ---
# Βεβαιωνόμαστε ότι ο στόχος (Cath) υπάρχει και είναι στον πίνακα
target_col_name = 'Cath'

# Αν δεν έχετε τρέξει τον προηγούμενο κώδικα PCA, βεβαιωθείτε ότι το pc_df υπάρχει:
# 1. Εφαρμογή PCA
pca = PCA(n_components=2) # Χρειαζόμαστε μόνο τις 2 πρώτες συνιστώσες για το 2D plot
principal_components = pca.fit_transform(df_scaled) # df_scaled είναι τα τυποποιημένα αριθμητικά δεδομένα

# 2. Δημιουργία DataFrame με PC1, PC2 και τον στόχο
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# 3. Προσθήκη της στήλης-στόχου (Cath) στο DataFrame των PC
# Χρησιμοποιούμε τις τιμές από το αρχικό DataFrame (df), καθώς οι δείκτες ταιριάζουν
if target_col_name in df.columns:
    pc_df[target_col_name] = df[target_col_name].values
else:
    print(f"Προσοχή: Η στήλη '{target_col_name}' δεν βρέθηκε στο αρχικό DataFrame για χρωματισμό.")
    exit()


# EDA - Distribution of numerical columns
plt.figure(figsize=(14, 10))
df.hist(figsize=(14, 10), bins = 20)

plt.tight_layout()
plt.show()


explained_variance = pca.explained_variance_ratio_

# --- Οπτικοποίηση (Scatter Plot) ---
plt.figure(figsize=(10, 8))

# Χρωματίζουμε τα σημεία ανάλογα με την κατηγορία (target_col_name)
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue=target_col_name, # Η στήλη που καθορίζει το χρώμα
    data=pc_df,
    palette='viridis', # Επιλέξτε μια παλέτα
    s=70,              # Μέγεθος σημείου
    alpha=0.7          # Διαφάνεια
)

plt.title(f'PCA Scatter Plot - Οπτικοποίηση Μεταξύ PC1 και PC2')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% Διακύμανσης)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% Διακύμανσης)')
plt.legend(title=target_col_name)
plt.grid(True)
plt.show()
