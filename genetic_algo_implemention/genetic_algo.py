import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc)

# -------------------------------------------------------
# 1. Φόρτωση και Προεπεξεργασία Δεδομένων
# -------------------------------------------------------
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

# Τυποποίηση (Normalization)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_df)
N_FEATURES = X.shape[1]

# -------------------------------------------------------
# 2. Ρυθμίσεις Γενετικού Αλγόριθμου (GA)
# -------------------------------------------------------
POP_SIZE = 50
N_GEN = 10
MUT_RATE = 0.1
CROSS_RATE = 0.75

def fitness(individual):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected) == 0: return 0
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X[:, selected], y, cv=cv, scoring='accuracy')
    return scores.mean()

def create_individual():
    return [random.randint(0, 1) for _ in range(N_FEATURES)]

def mutate(ind):
    for i in range(N_FEATURES):
        if random.random() < MUT_RATE:
            ind[i] = 1 - ind[i]

def crossover(a, b):
    point = random.randint(1, N_FEATURES - 1)
    return a[:point] + b[point:], b[:point] + a[point:]

# -------------------------------------------------------
# 3. Εκτέλεση GA
# -------------------------------------------------------
population = [create_individual() for _ in range(POP_SIZE)]
best_ind = None
best_score = -1

print("Starting Genetic Algorithm...")
for gen in range(N_GEN):
    scored = [(ind, fitness(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if scored[0][1] > best_score:
        best_score = scored[0][1]
        best_ind = scored[0][0]

    print(f"Generation {gen+1}: Best Accuracy = {best_score:.4f}")

    parents = [ind for ind, score in scored[:POP_SIZE // 2]]
    next_pop = parents.copy()
    while len(next_pop) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        c1, c2 = crossover(p1, p2)
        mutate(c1); mutate(c2)
        next_pop.extend([c1, c2])
    population = next_pop[:POP_SIZE]

selected_idx = [i for i, bit in enumerate(best_ind) if bit == 1]
print(f"\nGA Finished. Selected {len(selected_idx)} features.")

# -------------------------------------------------------
# 4. Τελική Αξιολόγηση (10-Fold CV & ROC)
# -------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_y_true, all_y_prob = [], []
metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

print("\nStarting 10-Fold Cross-Validation...")
for train_idx, test_idx in cv.split(X[:, selected_idx], y):
    X_train, X_test = X[train_idx][:, selected_idx], X[test_idx][:, selected_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = SVC(kernel='linear', C=0.1, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    all_y_true.extend(y_test)
    all_y_prob.extend(y_prob)
    
    metrics['acc'].append(accuracy_score(y_test, y_pred))
    metrics['prec'].append(precision_score(y_test, y_pred, zero_division=0))
    metrics['rec'].append(recall_score(y_test, y_pred, zero_division=0))
    metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))

# Mean Results
print("\n===== FINAL MEAN RESULTS (10-FOLD CV) =====")
for k, v in metrics.items():
    print(f"{k.upper()}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

# -------------------------------------------------------
# 5. Τελική Αξιολόγηση (10-Fold CV & Confusion Matrix)
# -------------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_y_true = []
all_y_prob = []
all_y_pred = [] # Νέα λίστα για το confusion matrix
metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

print("\nStarting 10-Fold Cross-Validation...")
for train_idx, test_idx in cv.split(X[:, selected_idx], y):
    X_train, X_test = X[train_idx][:, selected_idx], X[test_idx][:, selected_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Εκπαίδευση μοντέλου
    model = SVC(kernel='linear', C=0.1, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Προβλέψεις
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Αποθήκευση αποτελεσμάτων για συνολική αξιολόγηση
    all_y_true.extend(y_test)
    all_y_prob.extend(y_prob)
    all_y_pred.extend(y_pred) # Προσθήκη εδώ
    
    # Metrics ανά fold
    metrics['acc'].append(accuracy_score(y_test, y_pred))
    metrics['prec'].append(precision_score(y_test, y_pred, zero_division=0))
    metrics['rec'].append(recall_score(y_test, y_pred, zero_division=0))
    metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))

# 6. Classification Report
print("\nDetailed Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Cad']))

# 7. Confusion Matrix 
cm = confusion_matrix(all_y_true, all_y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'CAD'], yticklabels=['Normal', 'CAD'])
plt.title('Confusion Matrix (GA-Selected Features)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve Plot
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Final ROC Curve')
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.show()