Loan Default Prediction Project

Overview:
This project aims to develop machine learning models for predicting loan defaults using historical data from a German bank. 
The objective is to assist financial institutions in identifying customers at higher risk of defaulting on their loans, thereby improving risk assessment and decision-making processes.

Organization:
The project is organized into several key sections:

1. Introduction: Provides an overview of the problem context and project objectives.
2. Methods and Materials: Describes data preprocessing steps, exploratory data analysis (EDA), and machine learning models used.
3. Results: Reports key findings from the analysis, including model performance metrics.
4. Discussion: Provides interpretation of results, discusses implications, limitations, and future directions.
5. Conclusions: Summarizes main findings and conclusions drawn from the study.

Usage:
To replicate the analysis:
1. Obtain the dataset (German_bank.csv) containing historical customer data.
2. Preprocess the dataset, conduct EDA, and train machine learning models.
3. Fine-tune model hyperparameters and evaluate model performance.
4. Interpret results, discuss findings, and draw conclusions.

Dependencies:
- Python
- pandas
- scikit-learn
- matplotlib
- seaborn


**Script**
1. Python Code for Data Reading, Exploration, and Cleaning
The code uses pandas for data reading, exploration, and cleaning. 

import pandas as pd

# Read the raw data
data = pd.read_csv('German_bank.csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Clean the data
# Code to handle missing values, encode categorical variables, and scale numerical features
print(df.isnull().sum())

# Check the distribution of the target variable
print(df['default'].value_counts())

2. Code for Data Manipulation, Wrangling, and Visualization

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
cat_columns = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 
               'employment_duration', 'other_credit', 'housing', 'job', 'phone', 'default']

le = LabelEncoder()
df_encoded = df.copy()
for col in cat_columns:
    df_encoded[col] = le.fit_transform(df[col])

# Scale numerical features
num_columns = ['months_loan_duration', 'amount', 'percent_of_income', 
               'years_at_residence', 'age', 'existing_loans_count', 'dependents']

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[num_columns] = scaler.fit_transform(df_encoded[num_columns])

# Display the preprocessed dataset
print(df_scaled.head())

#Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='default', data=df_scaled)
plt.title('Distribution of Default Status')
plt.xlabel('Default')
plt.ylabel('Count')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

3. Code for Model Training and Evaluation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['default'])
y = data['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

# Initialize the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Perform Grid Search Cross Validation
grid_search = GridSearchCV(gb_classifier, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Hyperparameters:")
print(best_params)
print("Best Accuracy:", best_accuracy)

# Evaluate the tuned Gradient Boosting model on the test set
best_gb_model = grid_search.best_estimator_
y_pred_tuned = best_gb_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print("Tuned Gradient Boosting Model Accuracy:", accuracy_tuned)
print(classification_report(y_test, y_pred_tuned))

