# CAD Prediction: SVM Feature Selection via Genetic Algorithm

## 📋 Project Overview
This project implements an advanced Machine learning methodology for predicting **Coronary Artery Disease (CAD)**. It utilizes a **Genetic Algorithm (GA)** for optimal feature selection and evaluates performance using a **Support Vector Machine (SVM)** classifier.

The primary goal is to identify the most discriminative clinical markers, reducing diagnostic costs and computational complexity without compromising predictive reliability.
---

## 📂 Repository Structure 
The project is organized into modular components, seperating exploratory analysis from the algorythmic implementation:

```text
project_root/
├── EDA_analysis_AND_preproc/
│   ├── data/                 # Directory containing the dataset (e.g., Cor_data.csv)
│   ├── CAD_eda.ipynb         # Jupyter Notebook for Exploratory Data Analysis
│   └── preproc.py            # Script for data cleaning and transformation
├── genetic_algo_implemention/
│   └── CAD_GA.ipynb          # Core notebook with the GA implementation and SVM evaluation
└── README.md                 # Project documentation
```

## 🛠 Methodology

### 1. Data Preprocessing (`preproc.py` & Notebooks)
The raw clinical data undergoes rigorous transformation to be compatible with distance-based algorithms:
* **Target Encoding:** The 'Cath' diagnosis is converted into a binary target (1 for CAD, 0 for Normal).
* **Binary Mapping:** Variables like 'Sex' and 'Y/N' flags are explicitly mapped to 1/0.
* **One-Hot Encoding:** Applied to multi-class variables such as 'BBB' and 'VHD'.
* **Normalization:** All features are scaled to a [0, 1] range using `MinMaxScaler`.

### 2. Feature Selection via Genetic Algorithm (GA)
To navigate the high-dimensional feature space (60 initial features), a GA simulates natural selection:
* **Fitness Function:** Evaluates feature subsets based on the Mean Accuracy of an SVM (RBF kernel) using 5-Fold Cross-Validation.
* **Evolutionary Process:** Uses a Population of 50, trained over 10 Generations with Selection, Crossover (75%), and Mutation (10%) rates.
* **Outcome:** The GA effectively reduced the feature space by selecting only the most significant clinical indicators (e.g., 29 out of 60 features).

### 3. Classification & Evaluation
The optimized feature subset is validated to ensure generalizability to unseen data:
* **Model:** Support Vector Machine (Linear Kernel).
* **Validation:** 10-Fold Stratified Cross-Validation.

---

## 📊 Results & Performance
The hybrid GA-SVM approach yielded highly reliable metrics, particularly in detecting positive CAD cases. Below are the mean results from the 10-Fold CV:

* **Accuracy:** ~87.4%
* **Precision:** ~90.5%
* **Recall (Sensitivity):** ~92.1% *(Crucial for medical diagnostics to minimize false negatives)*
* **F1-Score:** ~91.2%
* **ROC-AUC:** Demonstrated high discriminative power distinguishing between Normal and CAD patients.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/cad-prediction-ga-svm.git](https://github.com/yourusername/cad-prediction-ga-svm.git)
   cd cad-prediction-ga-svm
   ```
   
2. **Install Dependencies:**
   Ensure you have Python 3.x installed. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. **Data preparation:**
Ensure your Dataset (Cor_data.csv) is placed in the ```EDA_analysis_AND_preproc/data/ folder```.

4. Run the Pipeline:

 * For interactive exploration, open and run the Jupyter Notebooks: CAD_eda.ipynb and CAD_GA.ipynb.
 * For a direct execution of the algorithm, run the testing script:     
   
   ```bash
    python GA_testing.py
   ```
   
## ⚠️ Medical Disclaimer

This software and its underlying models are strictly for research and educational purposes. It does not replace professional medical advice, diagnosis, or treatment. It is intended to support clinical decision-making by providing data-driven insights.
