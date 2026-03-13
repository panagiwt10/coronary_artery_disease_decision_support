# CAD Prediction: SVM Feature Selection via Genetic Algorithm

## Project Overview
This project implements an advanced Machine Learning methodology for predicting **Coronary Artery Disease (CAD)**. It utilizes a **Genetic Algorithm (GA)** for optimal feature selection and evaluates performance using a **Support Vector Machine (SVM)** classifier.

The primary goal is to identify the most discriminative clinical markers while reducing diagnostic costs and computational complexity through intelligent feature subset selection.

---

## Methodology

### 1. Data Preprocessing
The model performs several automated cleaning and transformation steps:
* **Categorical Encoding:** Converts clinical observations (e.g., Sex, Smoking status) using Binary Mapping and One-Hot Encoding for multi-class variables (BBB, VHD).
* **Target Labeling:** Encodes the diagnostic result (`Cath`) into binary format (1 for 'Cad', 0 for 'Normal').
* **Normalization:** Scales all numerical features to a [0, 1] range using **MinMaxScaler** to ensure compatibility with distance-based algorithms like SVM.

### 2. Feature Selection (Genetic Algorithm)
A Genetic Algorithm is employed to navigate the high-dimensional feature space and find the most relevant clinical indicators:
* **Fitness Function:** Evaluates feature subsets based on the Mean Accuracy of an SVM with an RBF kernel using 5-Fold Cross-Validation.
* **Evolutionary Process:** Uses Selection, Crossover, and Mutation to "evolve" the feature list over multiple generations.
* **Benefit:** This reduces "noise" in the data and prevents overfitting by removing redundant variables.

### 3. Classification & Evaluation
The final optimized feature set is validated using a **10-Fold Stratified Cross-Validation** approach with a Linear SVM.

---

## Results & Key Metrics
The model provides a comprehensive performance report including:
* **Accuracy:** Overall percentage of correct predictions.
* **Precision:** The reliability of the model when predicting positive CAD cases.
* **Recall (Sensitivity):** The ability to detect all actual patients (critical in medical diagnostics).
* **F1 Score:** The harmonic mean of Precision and Recall.
* **ROC-AUC:** Measures the model's ability to distinguish between classes across all thresholds.

---

## Conclusion: Resource Efficiency
By integrating Genetic Algorithms with SVM, this project achieves:
1.  **Clinical Efficiency:** Identification of a "minimal" set of medical tests required for accurate diagnosis, saving time and medical resources.
2.  **Cost Reduction:** Minimizing the need for expensive, non-essential examinations.
3.  **Scalability:** A lightweight model that can be integrated into real-time digital health tools.

---

## Requirements
* Python 3.x
* NumPy & Pandas
* Scikit-learn
* Matplotlib & Seaborn (for visualization)

## How to Run
1. Ensure `Cor_data.csv` is in the project directory.
2. Run the Jupyter Notebook or Python script:
   ```bash
   python main.py
