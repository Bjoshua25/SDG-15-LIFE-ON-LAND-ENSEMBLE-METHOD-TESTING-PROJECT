# **SDG 15 LIFE ON LAND | ENSEMBLE METHOD TESTING PROJECT**  
*Biodiversity Prediction using Ensemble Learning*  

## **INTRODUCTION**  
Sustainable Development Goal 15 (SDG 15) focuses on **protecting, restoring, and promoting sustainable ecosystems**. This project applies **ensemble learning methods** to predict the **Biodiversity Health Index (BHI)** using environmental data.  

By leveraging **Bagging, Boosting, and Stacking** techniques, we aim to identify the best predictive model for biodiversity conservation efforts.  

---

## **PROBLEM STATEMENT**  
Biodiversity loss is a global challenge, but **predicting ecosystem health** remains complex. This project seeks to:  
- **Analyze biodiversity data** to identify key environmental predictors.  
- **Test multiple ensemble learning techniques** to enhance prediction accuracy.  
- **Compare model performances** to determine the best approach for ecosystem monitoring.  

---

## **SKILL DEMONSTRATION**  
- **Exploratory Data Analysis (EDA) & Correlation Analysis**  
- **Feature Engineering & Selection**  
- **Bagging, Boosting, and Stacking Models**  
- **Model Performance Evaluation (MSE, RMSE, R² Score)**  
- **Data Visualization & Model Interpretation**  

---

## **DATA SOURCING**  
The dataset is sourced from [Explore-AI Public Data](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/SDG_15_Life_on_Land_Dataset.csv) and includes:  

### **1. Biodiversity Indicator** (Target Variable)  
- **Biodiversity Health Index (BHI)** – A measure of ecosystem health (0-1 scale).  

### **2. Environmental Factors**  
- **Forest Coverage (%)**  
- **Protected Areas (%)**  
- **Deforestation Rate**  
- **Carbon Sequestration**  
- **Soil Erosion & Land Degradation**  
- **Population Density & Rural Population Share**  

---

## **EXPLORATORY DATA ANALYSIS (EDA)**  
EDA was performed to uncover patterns in biodiversity and environmental factors.  

### **1. Data Overview**  
- **Loaded dataset and checked missing values.**  
- **Summary statistics** using `.describe()`.  
- **Pairplot visualization** to explore relationships.  

### **2. Feature Distributions & Correlations**  
- **Histogram & Density Plots** for feature distributions.  
- **Heatmap** to visualize correlation strengths.  
- **Key Insight:** Certain factors (deforestation, land degradation) strongly correlate with biodiversity loss.  

---

## **ENSEMBLE METHOD TESTING**  
This project evaluates **three ensemble techniques** for predicting biodiversity health.  

### **1. Bagging (Bootstrap Aggregation)**  
- Trains multiple **Decision Trees** on different random subsets of data.  
```python
from sklearn.ensemble import BaggingRegressor
ensemble = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, random_state=42)
ensemble.fit(X_train, y_train)
```
### **2. Boosting (Gradient Boosting & AdaBoost)**  
- **Boosting corrects errors** from previous models by adjusting weights.  
```python
from sklearn.ensemble import GradientBoostingRegressor
ensemble = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
ensemble.fit(X_train, y_train)
```
### **3. Stacking Regressor**  
- Uses multiple base models & a **meta-model** for final predictions.  
```python
from sklearn.ensemble import StackingRegressor
meta_model = RandomForestRegressor()
ensemble = StackingRegressor(estimators=[('rf', rf), ('dt', tree), ('gb', gb)], final_estimator=meta_model)
ensemble.fit(X_train, y_train)
```

---

## **MODEL EVALUATION**  
Each model was assessed using:  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score (Explained Variance)**  
- **Cross-validation** for performance consistency.  

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = ensemble.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## **KEY FINDINGS**  
- **Boosting (Gradient Boosting) outperformed Bagging & Stacking**, achieving the lowest MSE.  
- **Bagging was effective but slightly less accurate**, highlighting decision tree stability.  
- **Stacking showed mixed results**, suggesting model selection optimization is needed.  

---

## **HOW TO RUN THE PROJECT**  
### **1. Prerequisites**  
Ensure you have Python installed along with required libraries:  
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```
### **2. Clone the Repository**  
```bash
git clone https://github.com/yourusername/SDG15-Biodiversity-Prediction.git
cd SDG15-Biodiversity-Prediction
```
### **3. Run the Jupyter Notebook**  
```bash
jupyter notebook Ensemble_methods_exercise.ipynb
```
