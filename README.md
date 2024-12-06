
# Credit Card Fraud Detection

## Overview

This project focuses on developing a machine learning model to detect fraudulent credit card transactions. The dataset used is highly imbalanced, with fraudulent transactions representing only a small fraction of the total. To address this, techniques such as Synthetic Minority Oversampling Technique (SMOTE) were used to balance the classes. A Random Forest Classifier was chosen for its ability to handle imbalanced datasets and its interpretability. The model was evaluated based on various metrics such as precision, recall, F1-score, and ROC-AUC.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Recommendations](#recommendations)
- [License](#license)

---

## Project Structure

```markdown
├── data/
│   └── creditcard.csv             # The original dataset
├── Notebooks/
│   └── fraud_detection_analysis.ipynb  # Jupyter notebook for data analysis and model building
├── Scripts/
│   ├── data_preprocessing.py      # Data cleaning and preprocessing (SMOTE, scaling)
│   ├── Preprocessing.py           # Preprocessing
│   └── split_data.py              # Data split
│   ├── model_training.py          # Model training (Random Forest)
│   └── evaluate_model.py          # Model evaluation (metrics and confusion matrix)
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and instructions
└── LICENSE                        # Project license


```
## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/dani6566/CODSOFT-Credit-Card-Fraud-Detection.git
   cd CODSOFT-Credit-Card-Fraud-Detection
   ```

2. Install the required dependencies:
   - Create a virtual environment (optional but recommended):
     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## Usage

### 1. Data Preprocessing

To preprocess the data, balance the classes using SMOTE, and scale the numerical features.
This will output the preprocessed training and testing datasets.

### 2. Train the Model

To train the Random Forest model
This script trains the model using the preprocessed data and saves the trained model for future use.

### 3. Evaluate the Model

To evaluate the model, including metrics like precision, recall, F1-score, and ROC-AUC.

This will output the evaluation metrics and confusion matrix.

---

## Data Preprocessing

- **SMOTE (Synthetic Minority Oversampling Technique):**  
  SMOTE was used to balance the class distribution by oversampling the minority class (fraudulent transactions) until it matches the majority class size. This ensures that the model does not become biased toward predicting the majority class (genuine transactions).

- **Normalization:**  
  Features like `Time` and `Amount` were scaled using `StandardScaler` to bring them to a similar range.

---

## Model Training

- **Algorithm Used:** Random Forest Classifier  
  A Random Forest Classifier was chosen because it performs well on imbalanced datasets and has a good balance between interpretability and performance.

- **Training:**  
  The dataset was split into 80% for training and 20% for testing. SMOTE was applied to balance the classes before model training.

---

## Evaluation Metrics

The model was evaluated using the following metrics:

- **Precision:** Measures the accuracy of positive predictions.
- **Recall:** Measures the ability to identify all fraudulent transactions.
- **F1-Score:** A weighted average of precision and recall, providing a balance between the two.
- **Accuracy:** The overall accuracy of the model.
- **ROC-AUC Score:** The area under the ROC curve, measuring the model's ability to discriminate between fraudulent and genuine transactions.

---

## Recommendations

- **Hyperparameter Tuning:**  
  To further improve model performance, consider performing hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.

- **Model Comparison:**  
  Explore other algorithms like **Gradient Boosting Machines (GBM)** or **XGBoost** for potential improvement in recall and precision.

- **Cost-sensitive Learning:**  
  Given the significant cost of false negatives (missed fraudulent transactions), consider experimenting with cost-sensitive learning techniques to penalize misclassifications of fraudulent transactions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

If you have any questions or suggestions, feel free to open an issue or reach out to me at [danielhailay72@gmail.com].

---
```
