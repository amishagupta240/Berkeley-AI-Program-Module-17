**Comparing Classifiers: Bank Marketing Campaign Data**

Overview

This project compares the performance of four classification algorithms — K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM) — on a real-world dataset of telephone-based marketing campaigns for term deposit products, run by a Portuguese retail bank.

The goal: predict whether a client will subscribe to a term deposit (y = yes/no), using only basic bank-client demographic information, so the bank can prioritize outreach more efficiently.

Data Source


Dataset: bank-additional-full.csv (41,188 records, 21 columns)
Source: UCI Machine Learning Repository — Bank Marketing Data Set
Reference: Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems. The data represents 17 marketing campaigns conducted between May 2008 and November 2010.


Files

FileDescriptionprompt_III.ipynbFull analysis notebook — data understanding, feature engineering, model training, tuning, and comparisondata/bank-additional/bank-additional-full.csvRaw dataset (expected at this relative path for the notebook to run)

Methodology


Data Understanding: Checked for missing values and data type issues. Found no literal nulls, but several categorical fields use "unknown" as a disguised missing value (default: 20.9%, education: 4.2%, housing/loan: 2.4%, job: 0.8%, marital: 0.2%). The pdays field also uses a sentinel value of 999 to mean "never previously contacted" (96.3% of rows).
Business Objective: Predict, before a call is made, whether a client is likely to subscribe — so the bank can prioritize high-probability leads, reduce wasted calls, and improve campaign ROI. (Note: duration is excluded from modeling since it's only known after a call ends and leaks the outcome.)
Feature Engineering: Used only the 7 bank-client demographic features (age, job, marital, education, default, housing, loan), one-hot encoded the categoricals.
Train/Test Split: 75/25 stratified split.
Baseline: Majority-class predictor achieves ~88.7% accuracy (since ~88.7% of clients did not subscribe) — this is the bar every model needs to clear meaningfully.
Modeling: Trained Logistic Regression, KNN, Decision Tree, and SVM with default settings; compared fit time, train accuracy, and test accuracy.
Hyperparameter Tuning: Used GridSearchCV to tune each model (C for Logistic Regression/SVM, n_neighbors for KNN, max_depth for Decision Tree), switching the evaluation metric from accuracy to ROC-AUC since accuracy is uninformative under the dataset's ~89/11 class imbalance.


Key Findings


With only demographic features, all models perform close to the 88.7% accuracy baseline — these 7 features alone carry limited predictive signal for subscription behavior.
On ROC-AUC (a better metric given the class imbalance), Logistic Regression and the Decision Tree performed best (~0.64–0.65), modestly ahead of KNN (~0.62–0.64) and SVM (~0.53–0.55, near random).
SVM was by far the slowest to train, especially during hyperparameter search — its grid search was run on a 4,000-row subsample of the training data purely for computational feasibility.
The Decision Tree showed signs of overfitting at greater depths (high train accuracy, lower test accuracy), which max_depth tuning helps mitigate.


Recommendations / Next Steps


Incorporate additional contact and economic-context features (contact, month, campaign, poutcome, emp.var.rate, cons.price.idx, euribor3m, nr.employed, etc.) — excluding duration — to substantially improve predictive power.
Address class imbalance directly via class_weight='balanced', SMOTE, or threshold tuning.
Expand hyperparameter search ranges and use the full training set for SVM tuning if compute time allows.
Engineer derived features (e.g., age bins, a binary "previously contacted" flag from pdays).
