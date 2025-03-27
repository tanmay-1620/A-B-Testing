 A/B Testing for Optimizing Conversion Rates

 Project Overview
This project focuses on implementing A/B Testing to analyze the effectiveness of various changes made to a website. The goal is to identify which version of a webpage leads to higher conversion rates, user engagement, and overall improved customer experience.



 Objectives
- Optimize website conversion rates by analyzing customer behavior.
- Identify the most impactful features that contribute to higher conversions.
- Use machine learning models (XGBoost) to predict outcomes and validate A/B testing results.
- Leverage hyperparameter tuning and SMOTE to handle class imbalance and ensure robust model performance.



 Key Features
- **Data Preprocessing:** Cleans and prepares data by handling missing values, encoding categorical variables, and converting non-numeric columns.
- **Feature Engineering:** Identifies and selects the top 10 important features based on model importance.
- **Hyperparameter Tuning:** Implements RandomizedSearchCV to optimize model performance.
- **Model Training & Evaluation:** Trains an XGBoost classifier and evaluates it using accuracy and classification reports.
- **SMOTE Implementation:** Balances class distribution to avoid model bias.
- **Result Interpretation:** Generates insights based on A/B test results and conversion rate improvements.
- **Model Persistence:** Saves the best model to a `.pkl` file for future use.



 Technologies Used
- Python (Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn)
- Jupyter Notebook / Python Script
- GitHub for version control
- Matplotlib / Seaborn for data visualization

---

 Project Structure

A-B-Testing ├──  data │ ├── half_of_data.csv # Input dataset ├──  models │ └── best_xgb_model.pkl # Saved XGBoost model ├──  notebooks │ └── ab_testing_analysis.ipynb # Jupyter Notebook for analysis ├──  scripts │ └── ab_testing.py # Main Python script ├── 📄 README.md # Project Documentation └── 📄 requirements.txt # Required dependencies

Results
Best Parameters Found:-
{'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 500, 'subsample': 0.8}

 Top 10 Important Features:

1. app_score
2. up_membership_grade
3. up_life_duration
4. app_first_class
5. device_price
6. career
7. device_size
8. age
9. inter_type_cd
10. residence

Model Evaluation

Accuracy: 99.82%
Classification Report: Precision, Recall, F1-score with balanced performance across classes.
Class Imbalance Handling: SMOTE was used to balance the dataset effectively