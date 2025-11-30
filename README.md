```markdown
# ✈️ Flight Price Prediction — Machine Learning Model Comparison

This project builds and compares machine learning models for predicting **flight ticket prices**.  
The workflow includes Exploratory Data Analysis (EDA), feature engineering, preprocessing, model benchmarking, and learning-curve analysis.

---

# 📦 Project Structure

```
Flightprice_EDA.ipynb               # Exploratory analysis
Flightprice_preparation.ipynb       # Cleaning + feature engineering
Flightprice_process_data.ipynb      # Encoding + scaling + data split
Flightprice_alg_investigation.ipynb # Algorithm selection
Flightprice_DecisionTreeRegressor.ipynb # Decision Tree modeling + curves
Flightprice_SGDRegressor.ipynb      # SGDRegressor + tuned models
Comparision.ipynb                   # Model comparison table & plots
```

---

# 1. 🎯 Project Goal

To predict flight prices using:

- Airline  
- Flight duration  
- Number of stops  
- Origin/destination  
- Departure/arrival times  
- Engineered datetime features  

The objective is to find a **generalizable model** with strong predictive performance and stable learning behavior.

---

# 2. 🔍 Exploratory Data Analysis

The EDA notebook investigates:

- Price distribution  
- Average price by airline  
- Effect of stops on price  
- Duration vs price correlation  
- Outlier detection  
- Monthly/weekly/hourly demand trends  

### Key Findings
- **Stops strongly affect price** (nonstop flights cheapest).  
- **Duration correlates with price.**  
- Some airlines consistently charge higher fares.  
- Outliers needed cleaning for stable modeling.

---

# 3. 🧹 Data Preparation

### ✔ Cleaning  
- Removed missing timestamps  
- Handled extreme outliers in duration and fare  

### ✔ Feature Engineering  
Extracted new variables:
- `day`, `month`, `weekday`, `hour`  
- Total duration in minutes  
- Number of stops  
- Airline / route encoding  

### ✔ Preprocessing  
- One-hot encoding for categorical features  
- StandardScaler for numeric features (needed for SGD)  
- Train/dev/test split using `split_train_test_dev`

---

# 4. 🤖 Models Evaluated

### Baseline Models
- Linear Regression  
- Ridge / Lasso (tested but not retained)  

### Nonlinear Models
- **KNeighborsRegressor**  
- **DecisionTreeRegressor**  

### Gradient-based Model
- **SGDRegressor** (primary model with learning curves)

---

# 5. 📊 Model Performance Summary

Extracted from your notebook:

| Model | RMSE | MAE |
|-------|----------------|-----------|
| **SGDRegressor** | **92.4097** | **33.9834** |
| **LinearRegression** | 11295813625.9657 | 3560533347.0394 |
| **KNeighborsRegressor** | **164.6169** | **61.5052** |
| **DecisionTreeRegressor** | **142.6257** | **21.8568** |

### Interpretation
- Linear Regression failed due to poor scaling / high dimensionality  
- **SGDRegressor has the best overall RMSE → strongest global accuracy**  
- Decision Tree has the lowest MAE but significantly worse RMSE → overfitting  
- KNN performs poorly in high-dimensional categorical space  

---

# 6. 📉 Learning Curve Findings

## 🌳 **Decision Tree Learning Curve**
(From your plot)

- Training score ≈ **1.00**  
- CV score ≈ **0.9991** at ~4000 samples  
- **Error rate: 17.06%**

### Interpretation  
- Perfect training score → **severe overfitting**  
- Validation curve converges early → more data won’t help  
- Limitation is the model, not the data  
- Additional regularization or feature simplification required  

---

## ⚡ **SGD Regressor Learning Curves**

### **Initial SGD Model**
- RMSE: **142.39**  
- Test RMSE: **146.61**  
- Error rate: **34.93%**  
- **High variance** (large train–CV gap)

### **Improved SGD Model (Final)**
- Train RMSE: **73.40**  
- Test RMSE: **71.59**  
- Error rate: **17.06%**  
- Tight convergence of curves → **low variance**  
- Strong generalization  

---

# 7. 🏆 Final Model Selection

### 🎖 **Chosen Final Model: Tuned SGDRegressor**

Selected because:

- Best **overall RMSE (71–92 range depending on dataset split)**  
- Lowest **error rate (17.06%)**  
- Most stable **learning curve**  
- Handles high-dimensional, sparse one-hot encoded features  
- Regularization prevents overfitting  

Given the dataset size (~4k training examples), SGD provides the best tradeoff of:

- Speed  
- Stability  
- Generalization  

---

# 8. 🚀 Future Improvements

Potential next steps:

- Add Gradient Boosting models: **XGBoost**, **LightGBM**, **CatBoost**  
- Hyperparameter search: GridSearchCV / RandomizedSearchCV  
- Add holiday / seasonal features  
- Try embedding categorical variables (target encoding, embeddings)  
- Train Random Forest + reduce depth to avoid overfitting  
- Evaluate R² for interpretability  

---

# 9. 🛠 Tech Stack

- Python  
- pandas, NumPy  
- scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook  

