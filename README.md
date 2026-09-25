# Machine Learning Algorithms

This repository contains my Machine Learning practice, algorithm implementations, and end-to-end machine learning projects using Python and Scikit-learn.

## Phase 3 - Supervised Learning

The following supervised learning algorithms have been implemented and practiced:

1. Linear Regression
2. Multiple Linear Regression
3. Logistic Regression
4. K-Nearest Neighbors (KNN)
5. Decision Tree
6. Random Forest
7. Naive Bayes
8. Support Vector Machine (SVM)

## Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

### Project Objective

The objective is to build a classification model that can identify customers who are likely to leave a telecom service based on their demographic, service, contract, and billing information.

### Project Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Model Selection
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
New Customer Prediction
      ↓
Streamlit Deployment
```

### Dataset

The project uses the IBM Telco Customer Churn dataset.

* Final dataset size: 7,032 records
* Target variable: `Churn`
* `No` → 0
* `Yes` → 1

### Data Preprocessing

The following preprocessing techniques were applied:

* Missing value handling
* Removal of unnecessary customer ID
* Categorical feature encoding using One-Hot Encoding
* Train/test split with stratification
* Feature scaling for distance-based models such as KNN and SVM

### Feature Engineering

Additional features were explored, including:

* Tenure groups
* Average monthly spending
* Service count
* Month-to-month contract indicator

Feature engineering was evaluated experimentally and compared with the original feature set.

### Models Trained

The following classification models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Naive Bayes
* Support Vector Machine

### Model Comparison

The models were evaluated using:

* Accuracy
* Churn Recall
* Churn F1-Score
* ROC-AUC

The original Logistic Regression model was retained as the final model candidate after comparing baseline and tuned models.

### Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation and ROC-AUC scoring was used for:

* Logistic Regression
* Random Forest

Hyperparameter tuning was performed to evaluate whether model performance could be improved on unseen data.

### Final Model

**Logistic Regression**

Final test-set performance:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 80.38% |
| Precision | 64.67% |
| Recall    | 57.75% |
| F1-Score  | 61.02% |
| ROC-AUC   | 83.59% |

### Model Interpretability

Logistic Regression coefficients were analyzed to identify encoded features associated with higher or lower predicted churn risk.

The coefficient analysis is used for model interpretation and represents model associations rather than causal relationships.

### Model Saving

The trained model and preprocessing pipeline were saved using Joblib:

```text
final_model.pkl
preprocessor.pkl
```

This allows the trained model to be reused without retraining.

### New Customer Prediction

The project includes a reusable prediction pipeline that accepts new customer information and generates:

* Churn prediction
* Churn probability

Example:

```text
Prediction: Churn
Churn Probability: 77.58%
```

### Streamlit Deployment

A Streamlit web application was developed to provide an interactive interface for customer churn prediction.

```text
Customer Input
      ↓
Streamlit Interface
      ↓
Preprocessing Pipeline
      ↓
Logistic Regression Model
      ↓
Prediction + Churn Probability
```

### How to Run

Clone the repository and navigate to the project directory.

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib jupyter
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser and allow users to enter customer information and generate churn predictions.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook
* Git & GitHub

## Learning Approach

For each machine learning algorithm, the following workflow was followed:

* Concept
* Intuition
* Mathematics
* Dataset
* Features and Target
* Train/Test Split
* Model Training
* Prediction
* Model Evaluation
* Hyperparameter Tuning
* New Data Prediction
* Deployment

## Project Structure

```text
Machine-Learning-Algorithms/
│
├── Projects/
│   └── Customer-Churn-Prediction/
│       ├── data/
│       ├── Customer_Churn_Prediction.ipynb
│       ├── app.py
│       ├── final_model.pkl
│       ├── preprocessor.pkl
│       ├── model_comparison.csv
│       ├── hyperparameter_tuning_results.csv
│       ├── README.md
│       └── .gitignore
│
└── README.md
```

## Status

### Phase 3 - Supervised Learning: Completed ✅

### Customer Churn Prediction Project: Completed ✅
