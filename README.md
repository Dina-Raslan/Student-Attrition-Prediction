# Student Attrition Prediction

A machine learning pipeline to predict whether a university student 
will return for their second academic year, trained on a dataset of 
3,400 students across 56 features.

## Dataset
- 3,400 students | 56 raw features
- Feature categories: academic performance, course grades, demographics, 
  housing, financial aid, and parental education
- Class distribution: 78.7% returned (2,677) | 21.3% did not return (723)

## Pipeline

**Preprocessing**  
Standardized academic term formats, engineered credit hour ratio features 
(earned/attempted hours per term), imputed missing values using column means 
for numerical fields and zero-fill for missing grades, one-hot encoded 
categorical variables — expanding to 165 features.

**Feature Selection**  
RandomForest importance ranking for initial analysis, followed by 
SelectKBest (chi-squared) to select the top 50 features from the 
expanded feature space. Applied MinMaxScaler normalization prior to selection.

**Data Splitting**  
80/20 train-test split (2,720 train | 680 test), stratified by target class.  
SMOTEENN applied to address class imbalance (original: 2,141 vs 579 → 
resampled: 1,572 vs 1,058).

## Results

All models tuned via GridSearchCV with StratifiedKFold (5-fold CV).

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | 82.06% |
| KNN (k=7) | 80.88% |
| Naive Bayes | 80.59% |
| SVM (linear, C=0.1) | 82.06% |
| Decision Tree (max_depth=5) | 82.35% |
| Random Forest (n=200) | 80.29% |
| Extra Trees (n=200) | 78.38% |
| Bagging (n=30) | 79.85% |
| AdaBoost (n=100) | **82.35%** |
| Gradient Boosting (n=50) | 82.35% |
| XGBoost (lr=0.01, n=100) | 82.65% |
| Voting (RF + XGBoost + Bagging) | 81.18% |
| Stacking (RF + XGBoost + Bagging → LR) | 82.21% |

**Selected model: AdaBoost** — best balance between accuracy and 
precision on the minority class (students at risk of not returning).

## Stack
Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib
