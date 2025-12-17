import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt 
import seaborn as sns 


# -------------------------------------------------------
# 1. Φόρτωση και Προεπεξεργασία Δεδομένων
# -------------------------------------------------------
print("--- Loading and Preprocessing Data ---")
df_raw = pd.read_csv("Cor_data.csv")
df = df_raw.copy()

# Κωδικοποίηση Στόχου: Cath (1 για 'Cad', 0 για 'Normal')
df['target'] = df['Cath'].apply(lambda x: 1 if x == 'Cad' else 0)
df = df.drop('Cath', axis=1)

# Δυαδική Κωδικοποίηση: Y/N στήλες και Sex
binary_map = {'Y': 1, 'N': 0, 'Male': 1, 'Fmale': 0}
object_cols = df.select_dtypes(include=['object']).columns

for col in object_cols:
    if len(df[col].unique()) <= 2: 
        df[col] = df[col].map(lambda x: binary_map.get(x, x))

# One-Hot Encoding: Πολυταξικές στήλες ('BBB', 'VHD')
df = pd.get_dummies(df, columns=['BBB', 'VHD'], prefix=['BBB', 'VHD'], drop_first=False)

# Διαχωρισμός χαρακτηριστικών (X) και στόχου (y)
X_df = df.drop('target', axis=1)
y = df['target'].to_numpy()
feature_names_new = list(X_df.columns)

# Τυποποίηση (Εδώ χρησιμοποιούμε MinMaxScaler όπως στον αρχικό κώδικα)
# Σημείωση: Αν θες Standard Scaler, άλλαξε το σε StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X_df)

N_FEATURES = X.shape[1]  # Αριθμός χαρακτηριστικών

print(f"Total features: {N_FEATURES}")
print(f"Total samples: {X.shape[0]}")

# -------------------------------------------------------
# 2. Ρυθμίσεις Γενετικού Αλγόριθμου (GA)
# -------------------------------------------------------
POP_SIZE = 50   # Μέγεθος πληθυσμού
N_GEN = 10      # Αριθμός γενιών
MUT_RATE = 0.1  # Πιθανότητα μετάλλαξης
CROSS_RATE = 0.75  # Πιθανότητα διασταύρωσης

# Συνάρτηση fitness
def fitness(individual):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected) == 0:
        return 0
    
    # SVC με Linear Kernel
    model = SVC(kernel='linear', C=1.0, random_state=42)
    
    # 5-Fold Cross Validation για την αξιολόγηση του ατόμου
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X[:, selected], y, cv=cv, scoring='accuracy')
    
    return scores.mean()

# Δημιουργία ατόμου
def create_individual():
    return [random.randint(0, 1) for _ in range(N_FEATURES)]

# Μετάλλαξη
def mutate(ind):
    for i in range(N_FEATURES):
        if random.random() < MUT_RATE:
            ind[i] = 1 - ind[i]

# Διασταύρωση
def crossover(a, b):
    if N_FEATURES < 2: return a, b # Ασφάλεια για λίγα features
    point = random.randint(1, N_FEATURES - 1)
    return a[:point] + b[point:], b[:point] + a[point:]

# -------------------------------------------------------
# 3. Εκτέλεση GA
# -------------------------------------------------------
print("\n--- Starting Genetic Algorithm ---")
population = [create_individual() for _ in range(POP_SIZE)]
best_ind = None
best_score = -1

for gen in range(N_GEN):
    scored = [(ind, fitness(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    current_best_ind, current_best_score = scored[0]
    
    if current_best_score > best_score:
        best_score = current_best_score
        best_ind = current_best_ind[:] # Copy list
        
    print(f"Generation {gen+1:02d} | Best Score: {best_score:.4f}")
    
    # Επιλογή γονέων (Top 50%)
    parents = [ind for ind, score in scored[:POP_SIZE // 2]]
    
    # Επόμενη γενιά
    next_pop = parents.copy()
    while len(next_pop) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        c1, c2 = crossover(p1, p2)
        mutate(c1)
        mutate(c2)
        next_pop.append(c1)
        next_pop.append(c2)
    
    population = next_pop[:POP_SIZE]

# -------------------------------------------------------
# 4. Αποτελέσματα GA
# -------------------------------------------------------
selected_indices = [i for i, bit in enumerate(best_ind) if bit == 1]
selected_features_names = [feature_names_new[i] for i in selected_indices]

print("\n" + "="*50)
print(f"GA Completed. Selected {len(selected_indices)}/{N_FEATURES} features.")
print(f"Best Validation Accuracy during GA: {best_score:.4f}")
print("Selected Features:", selected_features_names)
print("="*50)

# -------------------------------------------------------
# 5. Τελική Αξιολόγηση (10-Fold CV) στα ΕΠΙΛΕΓΜΕΝΑ Features
# -------------------------------------------------------
print("\n--- Running Final 10-Fold Cross-Validation ---")

# ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ ΤΕΛΙΚΟΥ DATASET ΜΟΝΟ ΜΕ ΤΑ ΕΠΙΛΕΓΜΕΝΑ FEATURES
X_final = X[:, selected_indices]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Λίστες για αποθήκευση των metrics
acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []

# Loop για κάθε Fold
for fold_idx, (train_index, test_index) in enumerate(cv.split(X_final, y)):
    
    X_train, X_test = X_final[train_index], X_final[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Εκπαίδευση
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Πρόβλεψη
    y_pred = model.predict(X_test)
    
    # Αποθήκευση σκορ
    acc_scores.append(accuracy_score(y_test, y_pred))
    prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
    rec_scores.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

# -------------------------------------------------------
# 6. Τελική Αναφορά
# -------------------------------------------------------
print("\nFINAL PERFORMANCE METRICS (Average of 10 Folds):")
print("-" * 40)
print(f"Accuracy:    {np.mean(acc_scores):.4f}  (±{np.std(acc_scores):.4f})")
print(f"Precision:   {np.mean(prec_scores):.4f}  (±{np.std(prec_scores):.4f})")
print(f"Recall:      {np.mean(rec_scores):.4f}  (±{np.std(rec_scores):.4f})")
print(f"F1 Score:    {np.mean(f1_scores):.4f}  (±{np.std(f1_scores):.4f})")
print("-" * 40)

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred) # Από το τελευταίο fold

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (Pred)', 'Cad (Pred)'],
            yticklabels=['Normal (True)', 'Cad (True)'])
plt.title('Confusion Matrix (Last Fold)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()