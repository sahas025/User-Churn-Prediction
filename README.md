# User Churn Prediction Project

## Project Overview
This project focuses on predicting user churn for a mobile application. Using a dataset of user behavior, built and evaluated machine learning models to identify users at risk of churning. The goal is to enable proactive retention strategies and reduce customer loss.

## Business Problem
Customer churn is a critical challenge for digital platforms. Losing active users directly impacts revenue and growth. By building a predictive model, the business can identify at-risk users and implement targeted interventions to improve retention and maximize customer lifetime value.

## Dataset
The dataset contains 14,999 entries with features related to user activity, engagement, and device information.

**Key Variables:**
- `label`: The target variable ("retained" or "churned")
- `sessions`: The number of times a user started a session
- `drives`: The number of completed user sessions
- `device`: The type of device used by the user (Android or iPhone)

## Methodology

### 1. Feature Engineering
New features were created to better capture user behavior patterns:
- `km_per_driving_day`: Mean kilometers per driving day
- `percent_sessions_in_last_month`: Proportion of total sessions in last month
- `professional_driver`: Binary feature for highly active users
- `km_per_drive`: Mean kilometers per drive
- `percent_of_drives_to_favorite`: Proportion of sessions to favorite locations

### 2. Data Preprocessing
- Handled infinite values from division operations
- Mapped categorical variables to numerical values
- Split data into Training (60%), Validation (20%), and Test (20%) sets

### 3. Modeling & Hyperparameter Tuning
Used `GridSearchCV` with 4-fold cross-validation to tune:
- **Random Forest Classifier**
- **XGBoost Classifier**

**Primary Metric: Recall** - Optimized to identify as many potential churners as possible.

### Key Findings:
- **XGBoost demonstrated 31% higher recall** than Random Forest in cross-validation
- Both models showed signs of overfitting, particularly Random Forest
- **Feature importance analysis** revealed driving behavior metrics as most predictive
- Final XGBoost model provides realistic baseline for churn prediction

### Top Predictive Features:
1. `km_per_hour` (Average speed)
2. `total_navigations_fav1` (Favorite location usage)
3. `n_days_after_onboarding` (User tenure)
4. `percent_sessions_in_last_month` (Recent activity)

## Business Implications
The model can identify approximately 15% of actual churners while maintaining reasonable precision. This enables targeted interventions for highest-risk users, potentially saving 15% of at-risk customers with minimal false positives.

## Libraries Used
- **Python** (Pandas, NumPy)
- **Scikit-learn** (Random Forest, model evaluation)
- **XGBoost** (Gradient Boosting)
- **Matplotlib/Seaborn** (Visualization)

## Next Steps
1. **Address class imbalance** with SMOTE or class weights
2. **Experiment with additional features** based on temporal patterns
3. **Implement cost-sensitive learning** to optimize business value
4. **Develop deployment pipeline** for real-time predictions

## How to Run
1. Clone the repository
2. Install required packages: `pip install pandas scikit-learn xgboost matplotlib seaborn`
3. Run the Jupyter notebook sequentially
4. Review results and model insights
