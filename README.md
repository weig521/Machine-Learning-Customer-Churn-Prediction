# Machine-Learning-Customer-Churn-Prediction
🏦 Bank Customer Churn Prediction: A Comparative Modeling Analysis

Project OverviewThis project focuses on developing a robust Machine Learning model to accurately predict bank customer churn, a critical factor for profitability in the financial services industry1. Customer retention is significantly more cost-effective than customer acquisition, making the ability to predict churn before it occurs paramount.

A comprehensive comparative analysis was conducted on a diverse set of classifiers, with a rigorous focus on evaluation metrics suitable for imbalanced datasets. The final model is ready for deployment to score the customer base, enabling targeted retention campaigns and proactive risk mitigation

4.🎯 Goal and ObjectivesThe primary objective was to establish an optimal predictive framework for identifying high-risk customer accounts.

The project involved the following key steps:
  - Data Preprocessing: Cleaning and transforming raw customer data, including handling categorical features and scaling numerical features
  - Comparative Analysis: Systematically testing and comparing the performance of ten popular classification models
  - Optimization: Using Stratified K-Fold Cross-Validation and Grid Search to ensure model generalization and avoid overfitting
  - Final Prediction: Generating a binary prediction (0 or 1) indicating the likelihood of customer churn.
  
🧠 Final Model SelectionThe comparative analysis clearly indicated that Ensemble Methods provided the highest predictive accuracy

Selected Model: AdaBoost Classifier

The AdaBoost Classifier (Adaptive Boosting) was selected as the final production model due to its superior performance in maximizing key metrics.
|Metric |   Value    |    Rationale|
|:--------|:-----------|:------------------------------------------------------------------------------------------------|
|ROC AUC  | High 0.80  | Maximized discrimination ability between churn and non-churn across all thresholds.|
|F1 Score|  Highest Achieved   |  Balanced the trade-off between Precision and Recall, crucial for imbalanced classification problems.|
              
Boosting methods consistently outperformed simpler models, confirming the non-linear and intricate nature of the customer churn problem.

📊 Feature ImportanceThe final AdaBoost model was analyzed to determine the most significant features driving customer churn. The key drivers are:

  - Age: The single most important feature, suggesting life-stage factors strongly influence a customer's decision to exit.
  - NumOfProducts: The number of bank products held, indicating customer dependence or reliance on the bank.
  - Geography: The customer's country, suggesting regulatory or economic factors play a role.
  - IsActiveMember: Whether the customer is an active user, reflecting current engagement.
  
Feature Importance Visualization

💾 Data Source

The analysis utilized the train.csv dataset from kaggle.com, which contains 15,000 records detailing various customer attributes, here are some examples:


| Metric | Value | Rationale |
| :------------------ | :-------------------:|  -------------------|
|CreditScore  |  customer's credit worthiness  |  High (financial responsibility)|
| Age       |      customer's age          |         High (life stage factor)| 
| Blance     |     account balance          |        High (financial commitment)| 
| Exited(target) | 1 if churned; 0 otherwise  |      Target variable | 

Note on Imbalance: The target variable, Exited, exhibits class imbalance, with a ratio of approximately 4:1 (non-churn to churn)24. This necessitated the use of ROC AUC and F1 Score over simple Accuracy.

🚀 Getting StartedPrerequisites
  - Python (3.x)
  - Jupyter Notebook or an IDE

Installation
```Bash
pip install pandas numpy scikit-learn matplotlib seaborn
```


Key Files in Repository
  - bank_churn_prediction.ipynb: Main notebook containing data exploration, preprocessing, comparative modeling, hyperparameter tuning, and final model selection.
  - train.csv: The primary dataset used for training and validation.
  - test.csv: The dataset used for prediction with the chosen model
  - README.md: This file.
  
📈 Future Work

While the AdaBoost Classifier achieved high performance, future avenues for exploration include:
  - Deployment: Moving the final model and preprocessing pipeline into a production environment (e.g., a real-time scoring API).
  - Advanced Feature Engineering: Creating interaction terms (e.g., Balance / EstimatedSalary) to potentially capture more complex patterns.
  - Cost-Sensitive Learning: Incorporating a cost-sensitive loss function during tuning to penalize False Negatives (predicting no churn when the customer does churn) more heavily, as this error is often more costly.
