import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay # ΝΕΑ ΕΙΣΑΓΩΓΗ
from typing import List

# Καταστολή προειδοποιήσεων για καθαρότερο output
import warnings
warnings.filterwarnings('ignore')


# ===================== PATHS & CONSTANTS =====================
DATA_PATH = "Cor_data.csv"
TARGET_COL = "Cath" # Στόχος: Cath (Cad/Normal)

# ===================== ENCODING MAPS =====================

binary_map = {
    "n": 0, "no": 0, "0": 0, "false": 0, "f": 0, "oxi": 0, "όχι": 0,
    "y": 1, "yes": 1, "1": 1, "true": 1, "t": 1, "ναι": 1, "nai": 1,
    "male": 1, "m": 1,
    "female": 0, "fem": 0, "woman": 0, "w": 0
}

ordinal_map = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "very severe": 4
}


def encode_col(s: pd.Series) -> pd.Series:
    """Μετατρέπει object στήλη σε αριθμητική (Binary, Ordinal, Factorize)."""
    if not pd.api.types.is_object_dtype(s):
        return s
    s_str = s.astype(str).str.strip().str.lower()
    
    m = s_str.map(binary_map)
    if m.notna().any() and m.dropna().isin([0, 1]).all():
        return m.astype("Int64")

    m = s_str.map(ordinal_map)
    if m.notna().any():
        return m.astype("Int64")

    codes, _ = pd.factorize(s_str)
    codes = pd.Series(codes, index=s.index).replace(-1, np.nan).astype("Int64")
    return codes


def run_preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Εφαρμόζει Encoding, καθαρισμό Target και διαχωρισμό X, y."""
    
    # 1. Target Encoding (Cath -> 0/1)
    cath_raw = df[TARGET_COL].astype(str).str.strip().str.lower()
    cath_map = {"cad": 1, "1": 1, "normal": 0, "0": 0}
    df['y_target'] = cath_raw.map(cath_map)

    # Καθαρισμός γραμμών με άκυρο/NaN target
    df = df.dropna(subset=['y_target'])
    df['y_target'] = df['y_target'].astype(int)

    # 2. Encoding όλων των μη-target στηλών
    X = df.drop(columns=[TARGET_COL, 'y_target'])
    y = df['y_target']

    X_encoded = X.apply(encode_col)
    
    return X_encoded, y


def evaluate_model(X, y, model, model_name: str, cv_folds, roc_plot_ax=None):
    """Εκτελεί 10-fold CV, εμφανίζει αποτελέσματα και υπολογίζει ROC/AUC."""
    
    # Ορίζουμε την Pipeline (Imputation και Scaling)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    print(f"\n--- EVALUATING MODEL: {model_name} ---")
    
    # 1. Υπολογισμός Cross-Validation Score (Accuracy)
    scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy') 
    
    # 2. Υπολογισμός Confusion Matrix και Classification Report
    y_pred = cross_val_predict(pipeline, X, y, cv=cv_folds)
    
    # 3. Υπολογισμός Probabilities για ROC Curve (μόνο για μοντέλα που υποστηρίζουν predict_proba)
    # Επειδή το SVC με probability=True είναι χρονοβόρο, το εφαρμόζουμε μόνο αν χρειάζεται.
    try:
        # Παίρνουμε τις πιθανότητες για τη θετική κλάση (1 - CAD)
        y_proba = cross_val_predict(pipeline, X, y, cv=cv_folds, method='predict_proba')[:, 1]
        
        # Υπολογισμός ROC Curve και AUC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Εκτύπωση AUC
        print(f"AUC (Area Under ROC Curve): {roc_auc:.4f}")
        
        # Σχεδίαση ROC Curve
        if roc_plot_ax:
            # Χρησιμοποιούμε RocCurveDisplay.from_predictions για να σχεδιάσουμε
            RocCurveDisplay.from_predictions(
                y, 
                y_proba, 
                name=f'{model_name} (AUC = {roc_auc:.2f})', 
                ax=roc_plot_ax
            )
            # Κλείνουμε το χρώμα της γραμμής για να μην σχεδιάζεται η περιοχή κάτω από την καμπύλη
            # RocCurveDisplay.from_predictions(..., color="blue", **kwargs)
            
    except AttributeError:
        # Εμφανίζεται αν το μοντέλο δεν υποστηρίζει predict_proba (π.χ. SVC χωρίς probability=True)
        y_proba = None
        roc_auc = None
        print("Note: Cannot calculate ROC/AUC (Model does not support predict_proba).")
        
    print(f"Accuracy Score (10-fold CV): {scores.mean():.4f} (+/- {scores.std()*2:.4f} [95% CI])")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y, y_pred))

    return roc_auc # Επιστροφή της AUC για κεντρική χρήση


def main_integrated():
    
    # =========================================================
    # 1. ΦΟΡΤΩΣΗ & ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
    # =========================================================
    print("--- 1. DATA LOADING AND PREPROCESSING ---")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: File {DATA_PATH} not found.")
        return

    # Εφαρμογή της προεπεξεργασίας (Encoding, Target Mapping, NaN Drop)
    X_encoded, y = run_preprocessing(df)
    
    # Ορισμός Cross-Validation (10-fold Stratified)
    cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    print(f"Total samples after cleaning: {len(X_encoded)}")
    print("10-fold Stratified CV initialized.")
    
    
    # =========================================================
    # 2. ΑΞΙΟΛΟΓΗΣΗ ΔΙΑΦΟΡΕΤΙΚΩΝ ΜΟΝΤΕΛΩΝ
    # =========================================================
    print("\n\n--- 2. MODEL EVALUATION (10-FOLD CV) ---")
    
    
    models = {
        # 1. Support Vector Machine (SVC) - ΠΡΟΣΟΧΗ: Προσθέτουμε probability=True
        "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale', probability=True), 
        
        # 2. Logistic Regression
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
        
        # 3. Random Forest Classifier
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Δημιουργία του γραφήματος (figure) για την καμπύλη ROC
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for name, model in models.items():
        # Καλούμε την evaluate_model, περνώντας το αντικείμενο ax για τη σχεδίαση
        evaluate_model(X_encoded, y, model, name, cv_folds, roc_plot_ax=ax)

    # =========================================================
    # 3. ΕΜΦΑΝΙΣΗ ROC CURVE
    # =========================================================
    print("\n\n--- 3. ROC CURVE VISUALIZATION ---")
    # Σχεδίαση της τυχαίας καμπύλης (diagonal line)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)') 
    
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main_integrated()