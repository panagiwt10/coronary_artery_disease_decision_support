# CAD_ML
here is a machine learning model for predicting if someone has Coronary artery disease or not 
by using genetic algorythms for feature selection and a cross valifation (90/10 %).The model which was used was an SVM either with a linear kernel or rbf kernel



# SVM Feature Selection via Genetic Algorithm (CAD Prediction)
Αυτό το έργο υλοποιεί μια μεθοδολογία Μηχανικής Μάθησης για την πρόβλεψη Στεφανιαίας Νόσου (CAD). Χρησιμοποιεί έναν Γενετικό Αλγόριθμο (Genetic Algorithm - GA) για την επιλογή των βέλτιστων χαρακτηριστικών (feature selection) και στη συνέχεια αξιολογεί την απόδοση χρησιμοποιώντας Support Vector Machine (SVM).

# Περιγραφή Διαδικασίας
Ο κώδικας εκτελεί τα παρακάτω βήματα αυτόματα:

# Προεπεξεργασία Δεδομένων (Preprocessing):

Μετατροπή κατηγορικών μεταβλητών (One-Hot Encoding & Binary Mapping).

Κωδικοποίηση της στόχου-μεταβλητής (Cath: 'Cad' -> 1, 'Normal' -> 0).

Κανονικοποίηση δεδομένων με MinMaxScaler (κλίμακα 0-1).

# Επιλογή Χαρακτηριστικών (Feature Selection):

Εφαρμογή Γενετικού Αλγόριθμου για την εύρεση του βέλτιστου υποσυνόλου χαρακτηριστικών.

Συνάρτηση Fitness: Ακρίβεια (Accuracy) ενός Linear SVM με 5-Fold Cross-Validation.

# Τελική Αξιολόγηση (Final Evaluation):

Χρήση μόνο των επιλεγμένων χαρακτηριστικών.

Εκτέλεση 10-Fold Stratified Cross-Validation για αντικειμενική αξιολόγηση.

# Οπτικοποίηση:

Εμφάνιση Heatmap του Confusion Matrix για την οπτική επαλήθευση των σφαλμάτων.

Απαιτήσεις (Requirements)
Για να τρέξει ο κώδικας, απαιτείται Python 3.x και οι παρακάτω βιβλιοθήκες:

Bash

pip install numpy pandas scikit-learn matplotlib seaborn

Δομή Δεδομένων
Ο κώδικας περιμένει ένα αρχείο με το όνομα Cor_data.csv στον ίδιο φάκελο. Το αρχείο πρέπει να περιέχει:

Διάφορα ιατρικά χαρακτηριστικά (στήλες).

Μια στήλη-στόχο με όνομα 'Cath' που περιέχει τις τιμές Cad (ασθενής) και Normal (υγιής).

Πώς να το τρέξετε
Απλά εκτελέστε το script:

Bash

python main.py
(Αντικαταστήστε το main.py με το όνομα που δώσατε στο αρχείο python).

# Ερμηνεία Αποτελεσμάτων
Ο κώδικας θα τυπώσει στην κονσόλα τα αποτελέσματα ανά γενιά του Γενετικού Αλγόριθμου και στο τέλος θα εμφανίσει τα συνολικά Metrics:

Accuracy: Το ποσοστό των συνολικών σωστών προβλέψεων.

Precision: Πόσο αξιόπιστο είναι το μοντέλο όταν προβλέπει "Ασθένεια".

Recall (Sensitivity): Η ικανότητα του μοντέλου να εντοπίζει όλους τους πραγματικούς ασθενείς (Κρίσιμο για ιατρικά δεδομένα).

F1 Score: Ο αρμονικός μέσος όρος Precision και Recall (δείκτης ισορροπίας).

Confusion Matrix: Ένα διάγραμμα που δείχνει τα True Positives, True Negatives, False Positives και False Negatives.

# Παράμετροι (Configuration)
Μπορείτε να αλλάξετε τις παραμέτρους του Γενετικού Αλγόριθμου στην αρχή του κώδικα:

Python

POP_SIZE = 50   # Μέγεθος πληθυσμού
N_GEN = 10      # Αριθμός γενιών (αυξήστε το για καλύτερα αποτελέσματα)
MUT_RATE = 0.1  # Πιθανότητα μετάλλαξης
Σημείωση: Η χρήση του MinMaxScaler και η σωστή διαδικασία Cross-Validation εξασφαλίζουν ότι δεν υπάρχει διαρροή δεδομένων (Data Leakage) και τα αποτελέσματα είναι αξιόπιστα.
